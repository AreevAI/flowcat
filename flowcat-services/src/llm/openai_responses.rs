// SPDX-License-Identifier: Apache-2.0
//
//! **OpenAI Responses API** LLM — the `/responses` streaming client.
//!
//! A **(D)istinct** client (PROVIDERS.md §1): OpenAI's newer **Responses API**
//! (`POST {base}/responses`) has a different request body + a different streamed
//! event schema (`response.output_text.delta`, `response.function_call_arguments.*`)
//! from chat-completions, so it is its own decode (not a `base_url` wrapper over the
//! chat-completions [`OpenAiLlm`](super::OpenAiLlm)). Behind `llm-openai-responses`.
//!
//! ## Wire protocol (cross-checked against pipecat `services/openai/llm.py`)
//!
//! Request body: `{ model, input, instructions?, tools?, stream: true }`. The
//! Responses API renames `messages` → `input` and lifts the system prompt to a
//! top-level `instructions` field; tools are a **flattened** `{ type:"function",
//! name, description, parameters }` (no nested `function` object like
//! chat-completions).
//!
//! Response SSE events are typed (`data: {json}` whose `"type"` discriminates):
//! - `response.output_text.delta` → `delta` text (→ [`Frame::LlmText`]).
//! - `response.output_item.added` with `item.type == "function_call"` opens a call
//!   (carrying `call_id` + `name`), keyed by `output_index`.
//! - `response.function_call_arguments.delta` → `delta` argument fragment, appended
//!   to the open call at `output_index`.
//! - `response.completed` ends the response.
//!
//! The SSE→frame mapping is split into pure functions ([`parse_sse_line`],
//! [`accumulate`], [`drain_tool_calls`]) so the wire format is unit-tested without a
//! network call.

use std::collections::{BTreeMap, VecDeque};

use async_trait::async_trait;
use futures::stream::{self, BoxStream};
use futures::StreamExt;
use serde_json::{json, Value};

use flowcat_core::error::{FlowcatError, Result};
use flowcat_core::processor::frame::{Frame, FunctionCall, LlmContext, StartParams};
use flowcat_core::service::{LlmService, Tool};

/// OpenAI's default API base for the Responses API.
pub const OPENAI_API_BASE: &str = "https://api.openai.com/v1";
/// Default model. Override with [`OpenAiResponsesLlm::model`].
const DEFAULT_MODEL: &str = "gpt-4o";

/// OpenAI Responses-API LLM service (streaming).
pub struct OpenAiResponsesLlm {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
    tools: Vec<Tool>,
}

impl OpenAiResponsesLlm {
    /// Construct bound to `api_key` (default base + a Responses-capable model).
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: OPENAI_API_BASE.to_string(),
            model: DEFAULT_MODEL.to_string(),
            tools: Vec::new(),
        }
    }

    /// Override the API base (a compatible gateway). Trailing slashes are trimmed.
    /// Only an operator-supplied base is honoured (config, never request-derived), so
    /// there is no request-controlled SSRF surface.
    pub fn base_url(mut self, base: impl Into<String>) -> Self {
        self.base_url = base.into().trim_end_matches('/').to_string();
        self
    }

    /// Override the model (default `gpt-4o`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Build the request body for `ctx` (pure — the seam the body test drives).
    ///
    /// The Responses API uses `input` (not `messages`) and lifts the system prompt
    /// to a top-level `instructions` field.
    fn request_body(&self, ctx: &LlmContext) -> Value {
        let mut instructions = String::new();
        let mut input: Vec<Value> = Vec::with_capacity(ctx.messages.len());
        for m in &ctx.messages {
            if m.get("role").and_then(|r| r.as_str()) == Some("system") {
                if let Some(text) = m.get("content").and_then(|c| c.as_str()) {
                    if !instructions.is_empty() {
                        instructions.push_str("\n\n");
                    }
                    instructions.push_str(text);
                }
            } else {
                input.push(m.clone());
            }
        }

        let mut body = json!({
            "model": self.model,
            "input": input,
            "stream": true,
        });
        if !instructions.is_empty() {
            body["instructions"] = Value::String(instructions);
        }
        // Tools: prefer the context's, else the service-level set. Responses flattens
        // each tool to `{ type:"function", name, description, parameters }`.
        let tools: Vec<Value> = if !ctx.tools.is_empty() {
            ctx.tools.clone()
        } else {
            self.tools.iter().map(tool_to_responses).collect()
        };
        if !tools.is_empty() {
            body["tools"] = Value::Array(tools);
        }
        body
    }
}

/// Map a flowcat [`Tool`] to the Responses-API (flattened) tool schema.
fn tool_to_responses(t: &Tool) -> Value {
    json!({
        "type": "function",
        "name": t.name,
        "description": t.description,
        "parameters": t.params,
    })
}

#[async_trait]
impl LlmService for OpenAiResponsesLlm {
    fn name(&self) -> &str {
        "openai_responses"
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        Ok(())
    }

    async fn run_llm<'a>(&'a mut self, ctx: &'a LlmContext) -> Result<BoxStream<'a, Frame>> {
        let body = self.request_body(ctx);
        let url = format!("{}/responses", self.base_url);
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.api_key)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| FlowcatError::Network(format!("openai_responses send: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(FlowcatError::Network(format!(
                "openai_responses {status}: {text}"
            )));
        }

        Ok(sse_to_frames(resp.bytes_stream()))
    }

    fn set_tools(&mut self, tools: Vec<Tool>) {
        self.tools = tools;
    }
}

/// One accumulating function call (opened by `output_item.added`, its args appended
/// by `function_call_arguments.delta`), keyed by `output_index`.
struct ToolAcc {
    id: String,
    name: String,
    args: String,
}

/// State carried across SSE chunks by the [`sse_to_frames`] unfold.
struct SseState {
    buf: String,
    started: bool,
    finished: bool,
    pending: VecDeque<Frame>,
    tool_acc: BTreeMap<u64, ToolAcc>,
}

impl SseState {
    /// Flush end-of-response frames once: ensure `LlmResponseStart` was emitted, then
    /// any assembled tool calls (`FunctionCallsStarted`), then `LlmResponseEnd`.
    fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        if !self.started {
            self.started = true;
            self.pending.push_back(Frame::LlmResponseStart);
        }
        if let Some(f) = drain_tool_calls(&mut self.tool_acc) {
            self.pending.push_back(f);
        }
        self.pending.push_back(Frame::LlmResponseEnd);
    }
}

/// Turn a reqwest byte stream of Responses SSE into a [`Frame`] stream. Owns the body
/// stream so it doesn't borrow the service.
fn sse_to_frames<S, B, E>(byte_stream: S) -> BoxStream<'static, Frame>
where
    S: futures::Stream<Item = std::result::Result<B, E>> + Send + 'static,
    B: AsRef<[u8]> + Send + 'static,
    E: Send + 'static,
{
    let inner = Box::pin(byte_stream);
    let st = SseState {
        buf: String::new(),
        started: false,
        finished: false,
        pending: VecDeque::new(),
        tool_acc: BTreeMap::new(),
    };
    stream::unfold((inner, st), |(mut inner, mut st)| async move {
        loop {
            if let Some(f) = st.pending.pop_front() {
                return Some((f, (inner, st)));
            }
            if st.finished {
                return None;
            }
            match inner.next().await {
                Some(Ok(bytes)) => {
                    st.buf.push_str(&String::from_utf8_lossy(bytes.as_ref()));
                    while let Some(nl) = st.buf.find('\n') {
                        let line: String = st.buf.drain(..=nl).collect();
                        match parse_sse_line(line.trim_end()) {
                            SseEvent::None => {}
                            SseEvent::Done => st.finish(),
                            SseEvent::Chunk(v) => {
                                if !st.started {
                                    st.started = true;
                                    st.pending.push_back(Frame::LlmResponseStart);
                                }
                                accumulate(&v, &mut st.tool_acc, &mut st.pending);
                            }
                        }
                    }
                    if let Some(f) = st.pending.pop_front() {
                        return Some((f, (inner, st)));
                    }
                }
                Some(Err(_)) | None => {
                    st.finish();
                    if let Some(f) = st.pending.pop_front() {
                        return Some((f, (inner, st)));
                    }
                    return None;
                }
            }
        }
    })
    .boxed()
}

/// One parsed SSE line.
enum SseEvent {
    /// A `data: {json}` typed event (its `"type"` field discriminates it).
    Chunk(Value),
    /// A `response.completed` (or `response.done`) terminal event.
    Done,
    /// A blank line, an `event:` name line, a comment, or unrelated noise.
    None,
}

/// Parse one Responses SSE line into an [`SseEvent`] (pure — the wire-fixture seam).
/// The `event:` line is redundant — we key off the `data:` payload's `"type"`.
fn parse_sse_line(line: &str) -> SseEvent {
    let line = line.trim();
    let Some(data) = line.strip_prefix("data:") else {
        return SseEvent::None;
    };
    let data = data.trim();
    if data.is_empty() {
        return SseEvent::None;
    }
    match serde_json::from_str::<Value>(data) {
        Ok(v) => {
            let t = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
            if t == "response.completed" || t == "response.done" {
                SseEvent::Done
            } else {
                SseEvent::Chunk(v)
            }
        }
        Err(_) => SseEvent::None,
    }
}

/// Fold one Responses SSE event into the running state (pure — the wire-fixture
/// seam). Text deltas push [`Frame::LlmText`]; function-call items are opened on
/// `output_item.added` and their argument deltas accumulated by `output_index`.
fn accumulate(event: &Value, tool_acc: &mut BTreeMap<u64, ToolAcc>, pending: &mut VecDeque<Frame>) {
    let etype = event.get("type").and_then(|t| t.as_str()).unwrap_or("");
    let index = event
        .get("output_index")
        .and_then(|i| i.as_u64())
        .unwrap_or(0);
    match etype {
        "response.output_text.delta" => {
            if let Some(text) = event.get("delta").and_then(|d| d.as_str()) {
                if !text.is_empty() {
                    pending.push_back(Frame::LlmText(text.to_string()));
                }
            }
        }
        "response.output_item.added" => {
            let item = event.get("item");
            if item.and_then(|i| i.get("type")).and_then(|t| t.as_str()) == Some("function_call") {
                let id = item
                    .and_then(|i| i.get("call_id"))
                    .and_then(|c| c.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = item
                    .and_then(|i| i.get("name"))
                    .and_then(|n| n.as_str())
                    .unwrap_or("")
                    .to_string();
                tool_acc.insert(
                    index,
                    ToolAcc {
                        id,
                        name,
                        args: String::new(),
                    },
                );
            }
        }
        "response.function_call_arguments.delta" => {
            if let Some(frag) = event.get("delta").and_then(|d| d.as_str()) {
                if let Some(entry) = tool_acc.get_mut(&index) {
                    entry.args.push_str(frag);
                }
            }
        }
        _ => {}
    }
}

/// Assemble the accumulated tool calls into a single [`Frame::FunctionCallsStarted`],
/// if any (pure — the wire-fixture seam). An empty argument run decodes to `{}`.
fn drain_tool_calls(tool_acc: &mut BTreeMap<u64, ToolAcc>) -> Option<Frame> {
    if tool_acc.is_empty() {
        return None;
    }
    let calls: Vec<FunctionCall> = std::mem::take(tool_acc)
        .into_values()
        .map(|t| FunctionCall {
            function_name: t.name,
            tool_call_id: t.id,
            arguments: if t.args.trim().is_empty() {
                json!({})
            } else {
                serde_json::from_str(&t.args).unwrap_or(Value::Null)
            },
        })
        .collect();
    Some(Frame::FunctionCallsStarted(calls))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_body_uses_input_instructions_and_flat_tools() {
        let mut llm = OpenAiResponsesLlm::new("k");
        llm.set_tools(vec![Tool {
            name: "end_call".into(),
            description: "end".into(),
            params: json!({"type": "object"}),
        }]);
        let ctx = LlmContext {
            messages: vec![
                json!({"role": "system", "content": "be brief"}),
                json!({"role": "user", "content": "hi"}),
            ],
            tools: vec![],
        };
        let body = llm.request_body(&ctx);
        assert_eq!(body["model"], DEFAULT_MODEL);
        assert_eq!(body["stream"], true);
        // Responses: messages → input, system → instructions.
        assert!(body.get("messages").is_none());
        assert_eq!(body["instructions"], "be brief");
        assert_eq!(body["input"].as_array().unwrap().len(), 1);
        assert_eq!(body["input"][0]["content"], "hi");
        // Flattened tool shape: { type, name, description, parameters }.
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["name"], "end_call");
        assert!(body["tools"][0].get("function").is_none());
    }

    #[test]
    fn base_url_and_model_overrides_apply() {
        let llm = OpenAiResponsesLlm::new("k")
            .base_url("https://gw.example.com/")
            .model("gpt-4o-mini");
        assert_eq!(llm.base_url, "https://gw.example.com");
        assert_eq!(llm.model, "gpt-4o-mini");
    }

    #[test]
    fn parse_sse_line_classifies_delta_done_and_noise() {
        assert!(matches!(
            parse_sse_line(r#"data: {"type":"response.completed"}"#),
            SseEvent::Done
        ));
        assert!(matches!(parse_sse_line(""), SseEvent::None));
        assert!(matches!(
            parse_sse_line("event: response.output_text.delta"),
            SseEvent::None
        ));
        match parse_sse_line(
            r#"data: {"type":"response.output_text.delta","output_index":0,"delta":"hi"}"#,
        ) {
            SseEvent::Chunk(v) => assert_eq!(v["delta"], "hi"),
            _ => panic!("expected chunk"),
        }
    }

    #[test]
    fn accumulate_emits_text_deltas_in_order() {
        let mut acc = BTreeMap::new();
        let mut pending = VecDeque::new();
        for token in ["Hello", " ", "world"] {
            let ev = json!({
                "type": "response.output_text.delta",
                "output_index": 0,
                "delta": token
            });
            accumulate(&ev, &mut acc, &mut pending);
        }
        let texts: Vec<String> = pending
            .into_iter()
            .filter_map(|f| match f {
                Frame::LlmText(t) => Some(t),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["Hello", " ", "world"]);
    }

    #[test]
    fn accumulate_assembles_streamed_function_call() {
        let mut acc = BTreeMap::new();
        let mut pending = VecDeque::new();
        // Responses: item.added opens the call (call_id+name), then args deltas.
        let added = json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "function_call", "call_id": "call_1", "name": "book"}
        });
        let d1 = json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "delta": "{\"da"
        });
        let d2 = json!({
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "delta": "y\":\"mon\"}"
        });
        accumulate(&added, &mut acc, &mut pending);
        accumulate(&d1, &mut acc, &mut pending);
        accumulate(&d2, &mut acc, &mut pending);
        assert!(
            pending.is_empty(),
            "tool calls assemble, not stream as text"
        );
        let frame = drain_tool_calls(&mut acc).expect("a tool call");
        match frame {
            Frame::FunctionCallsStarted(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function_name, "book");
                assert_eq!(calls[0].tool_call_id, "call_1");
                assert_eq!(calls[0].arguments["day"], "mon");
            }
            other => panic!("expected FunctionCallsStarted, got {}", other.name()),
        }
    }

    #[tokio::test]
    async fn sse_to_frames_decodes_a_full_text_response() {
        let fixture = concat!(
            "event: response.created\n",
            "data: {\"type\":\"response.created\"}\n\n",
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"Hello\"}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\" there\"}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\"}\n\n",
        );
        let chunks: Vec<std::result::Result<Vec<u8>, std::io::Error>> =
            vec![Ok(fixture.as_bytes().to_vec())];
        let mut stream = sse_to_frames(stream::iter(chunks));
        let mut names = Vec::new();
        let mut text = String::new();
        while let Some(f) = stream.next().await {
            names.push(f.name());
            if let Frame::LlmText(t) = f {
                text.push_str(&t);
            }
        }
        assert_eq!(names.first(), Some(&"LlmResponseStart"));
        assert_eq!(names.last(), Some(&"LlmResponseEnd"));
        assert_eq!(text, "Hello there");
    }

    /// Live smoke (requires `OPENAI_API_KEY`): stream a one-token reply. Run:
    /// `OPENAI_API_KEY=… cargo test -p flowcat-services --features llm-openai-responses -- --ignored openai_responses_live`
    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY"]
    async fn openai_responses_live_streams_a_reply() {
        let key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY");
        let mut llm = OpenAiResponsesLlm::new(key).model("gpt-4o-mini");
        let ctx = LlmContext {
            messages: vec![json!({"role": "user", "content": "Say 'hi' and nothing else."})],
            tools: vec![],
        };
        let mut stream = llm.run_llm(&ctx).await.expect("run_llm");
        let mut saw_text = false;
        while let Some(f) = stream.next().await {
            if matches!(f, Frame::LlmText(_)) {
                saw_text = true;
            }
        }
        assert!(saw_text, "expected at least one LlmText");
    }
}
