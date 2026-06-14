// SPDX-License-Identifier: Apache-2.0
//
//! **Anthropic** (Claude) LLM — the Messages API streaming client.
//!
//! A **(D)istinct** client (PROVIDERS.md §1): the Anthropic Messages API
//! (`POST {base}/v1/messages`, `x-api-key` + `anthropic-version` headers, an SSE
//! stream of `message_start` / `content_block_start` / `content_block_delta` /
//! `content_block_stop` / `message_delta` / `message_stop` events — **not**
//! OpenAI-shaped). The [`OpenAiLlm`](super::OpenAiLlm) ref impl is the structural
//! template (owned byte-stream → [`Frame`] unfold + a pure SSE-decode seam).
//! Behind the `llm-anthropic` feature.
//!
//! ## Wire protocol (cross-checked against pipecat `services/anthropic/llm.py`)
//!
//! Request body: `{ model, max_tokens, system?, messages, tools?, stream: true }`.
//! Unlike OpenAI the **system prompt is a top-level `system` field** (lifted out of
//! the message list), and tools use `{ name, description, input_schema }` (not the
//! `{type:"function", function:{…}}` envelope).
//!
//! Response SSE events (each `event: <name>\n` then `data: {json}\n`):
//! - `content_block_start` with `content_block.type == "tool_use"` opens a tool
//!   call (carrying `id` + `name`); a `"text"` block opens a text run.
//! - `content_block_delta` carries either `delta.text` (→ [`Frame::LlmText`]) or
//!   `delta.partial_json` (streamed tool-call arguments, accumulated per block).
//! - `content_block_stop` closes the current block (finalising a tool call's args).
//! - `message_stop` ends the response.
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

/// Anthropic's default API base. A `base_url` override points at a gateway.
pub const ANTHROPIC_API_BASE: &str = "https://api.anthropic.com";
/// The Anthropic API version header value this client targets.
pub const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Default model. Override with [`AnthropicLlm::model`].
const DEFAULT_MODEL: &str = "claude-3-5-sonnet-latest";
/// Anthropic requires `max_tokens` on every request; this is the default cap.
const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Anthropic Claude LLM service (Messages API, streaming).
pub struct AnthropicLlm {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
    max_tokens: u32,
    tools: Vec<Tool>,
}

impl AnthropicLlm {
    /// Construct bound to `api_key` (default base + a Claude model).
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: ANTHROPIC_API_BASE.to_string(),
            model: DEFAULT_MODEL.to_string(),
            max_tokens: DEFAULT_MAX_TOKENS,
            tools: Vec::new(),
        }
    }

    /// Override the API base (a gateway). Trailing slashes are trimmed. Only an
    /// operator-supplied base is honoured (config, never request-derived), so there
    /// is no request-controlled SSRF surface.
    pub fn base_url(mut self, base: impl Into<String>) -> Self {
        self.base_url = base.into().trim_end_matches('/').to_string();
        self
    }

    /// Override the model (default `claude-3-5-sonnet-latest`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the `max_tokens` cap (default 4096).
    pub fn max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = n;
        self
    }

    /// Build the request body for `ctx` (pure — the seam the body test drives).
    ///
    /// Anthropic wants the system prompt as a **top-level `system` field**, so any
    /// `{"role":"system"}` messages are lifted out of the message list and joined.
    fn request_body(&self, ctx: &LlmContext) -> Value {
        let mut system = String::new();
        let mut messages: Vec<Value> = Vec::with_capacity(ctx.messages.len());
        for m in &ctx.messages {
            if m.get("role").and_then(|r| r.as_str()) == Some("system") {
                if let Some(text) = m.get("content").and_then(|c| c.as_str()) {
                    if !system.is_empty() {
                        system.push_str("\n\n");
                    }
                    system.push_str(text);
                }
            } else {
                messages.push(m.clone());
            }
        }

        let mut body = json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "stream": true,
        });
        if !system.is_empty() {
            body["system"] = Value::String(system);
        }
        // Tools: prefer the context's, else the service-level set. Anthropic wants
        // each tool as `{ name, description, input_schema }`.
        let tools: Vec<Value> = if !ctx.tools.is_empty() {
            ctx.tools.clone()
        } else {
            self.tools.iter().map(tool_to_anthropic).collect()
        };
        if !tools.is_empty() {
            body["tools"] = Value::Array(tools);
        }
        body
    }
}

/// Map a flowcat [`Tool`] to the Anthropic tool schema.
fn tool_to_anthropic(t: &Tool) -> Value {
    json!({
        "name": t.name,
        "description": t.description,
        "input_schema": t.params,
    })
}

#[async_trait]
impl LlmService for AnthropicLlm {
    fn name(&self) -> &str {
        "anthropic"
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        Ok(())
    }

    async fn run_llm<'a>(&'a mut self, ctx: &'a LlmContext) -> Result<BoxStream<'a, Frame>> {
        let body = self.request_body(ctx);
        let url = format!("{}/v1/messages", self.base_url);
        let resp = self
            .http
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| FlowcatError::Network(format!("anthropic send: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(FlowcatError::Network(format!("anthropic {status}: {text}")));
        }

        Ok(sse_to_frames(resp.bytes_stream()))
    }

    fn set_tools(&mut self, tools: Vec<Tool>) {
        self.tools = tools;
    }
}

/// One accumulating tool-call block (Anthropic streams `name`+`id` at the block
/// start, then `input_json` argument fragments via `partial_json` deltas).
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
    /// Open content blocks keyed by their `index` (Anthropic numbers them).
    tool_acc: BTreeMap<u64, ToolAcc>,
}

impl SseState {
    /// Flush end-of-response frames once: ensure a `LlmResponseStart` was emitted,
    /// then any assembled tool calls (`FunctionCallsStarted`), then `LlmResponseEnd`.
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

/// Turn a reqwest byte stream of Anthropic SSE into a [`Frame`] stream:
/// `LlmResponseStart`, then `LlmText`* / `FunctionCallsStarted`, then
/// `LlmResponseEnd`. Owns the body stream so it doesn't borrow the service.
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
    /// A `data: {json}` event payload (its `"type"` field discriminates it).
    Chunk(Value),
    /// The terminal `message_stop` event.
    Done,
    /// A blank line, an `event:` name line, a comment, or unrelated noise.
    None,
}

/// Parse one SSE line into an [`SseEvent`] (pure — the wire-fixture seam).
///
/// Anthropic interleaves `event: <name>` and `data: {json}` lines; we key off the
/// `data:` payload's own `"type"` field (the `event:` line is redundant), and treat
/// a `message_stop`-typed data payload as the terminal event.
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
            if v.get("type").and_then(|t| t.as_str()) == Some("message_stop") {
                SseEvent::Done
            } else {
                SseEvent::Chunk(v)
            }
        }
        Err(_) => SseEvent::None,
    }
}

/// Fold one Anthropic SSE event into the running state (pure — the wire-fixture
/// seam). Text deltas push [`Frame::LlmText`]; tool-call blocks are opened on
/// `content_block_start` and their `partial_json` argument fragments accumulated.
fn accumulate(event: &Value, tool_acc: &mut BTreeMap<u64, ToolAcc>, pending: &mut VecDeque<Frame>) {
    let etype = event.get("type").and_then(|t| t.as_str()).unwrap_or("");
    let index = event.get("index").and_then(|i| i.as_u64()).unwrap_or(0);
    match etype {
        "content_block_start" => {
            let block = event.get("content_block");
            if block.and_then(|b| b.get("type")).and_then(|t| t.as_str()) == Some("tool_use") {
                let id = block
                    .and_then(|b| b.get("id"))
                    .and_then(|i| i.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = block
                    .and_then(|b| b.get("name"))
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
        "content_block_delta" => {
            let delta = event.get("delta");
            // Text delta → LlmText.
            if let Some(text) = delta.and_then(|d| d.get("text")).and_then(|t| t.as_str()) {
                if !text.is_empty() {
                    pending.push_back(Frame::LlmText(text.to_string()));
                }
            }
            // Tool-call argument fragment → accumulate by block index.
            if let Some(frag) = delta
                .and_then(|d| d.get("partial_json"))
                .and_then(|p| p.as_str())
            {
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
    fn request_body_lifts_system_and_maps_tools() {
        let mut llm = AnthropicLlm::new("k");
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
        assert_eq!(body["max_tokens"], DEFAULT_MAX_TOKENS);
        // System lifted to a top-level field, not left in messages.
        assert_eq!(body["system"], "be brief");
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
        assert_eq!(body["messages"][0]["content"], "hi");
        // Anthropic tool shape: { name, description, input_schema }.
        assert_eq!(body["tools"][0]["name"], "end_call");
        assert_eq!(body["tools"][0]["input_schema"]["type"], "object");
        assert!(body["tools"][0].get("function").is_none());
    }

    #[test]
    fn base_url_and_model_overrides_apply() {
        let llm = AnthropicLlm::new("k")
            .base_url("https://gw.example.com/")
            .model("claude-3-haiku")
            .max_tokens(256);
        assert_eq!(llm.base_url, "https://gw.example.com");
        assert_eq!(llm.model, "claude-3-haiku");
        assert_eq!(llm.max_tokens, 256);
    }

    #[test]
    fn parse_sse_line_classifies_chunk_stop_and_noise() {
        assert!(matches!(
            parse_sse_line(r#"data: {"type":"message_stop"}"#),
            SseEvent::Done
        ));
        assert!(matches!(parse_sse_line(""), SseEvent::None));
        assert!(matches!(
            parse_sse_line("event: content_block_delta"),
            SseEvent::None
        ));
        match parse_sse_line(
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#,
        ) {
            SseEvent::Chunk(v) => assert_eq!(v["delta"]["text"], "hi"),
            _ => panic!("expected chunk"),
        }
    }

    #[test]
    fn accumulate_emits_text_deltas_in_order() {
        let mut acc = BTreeMap::new();
        let mut pending = VecDeque::new();
        for token in ["Hello", " ", "world"] {
            let ev = json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": token}
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
    fn accumulate_assembles_streamed_tool_use_block() {
        let mut acc = BTreeMap::new();
        let mut pending = VecDeque::new();
        // Anthropic: block start carries id+name, then partial_json arg fragments.
        let start = json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "book"}
        });
        let d1 = json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": "{\"da"}
        });
        let d2 = json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": "y\":\"mon\"}"}
        });
        accumulate(&start, &mut acc, &mut pending);
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
                assert_eq!(calls[0].tool_call_id, "toolu_1");
                assert_eq!(calls[0].arguments["day"], "mon");
            }
            other => panic!("expected FunctionCallsStarted, got {}", other.name()),
        }
    }

    #[tokio::test]
    async fn sse_to_frames_decodes_a_full_text_response() {
        // A hand-written Anthropic SSE fixture: start → two text deltas → stop.
        let fixture = concat!(
            "event: message_start\n",
            "data: {\"type\":\"message_start\"}\n\n",
            "event: content_block_start\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n",
            "event: content_block_delta\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" there\"}}\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n",
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

    /// Live smoke (requires `ANTHROPIC_API_KEY`): stream a one-token reply. Run:
    /// `ANTHROPIC_API_KEY=… cargo test -p flowcat-services --features llm-anthropic -- --ignored anthropic_live`
    #[tokio::test]
    #[ignore = "requires ANTHROPIC_API_KEY"]
    async fn anthropic_live_streams_a_reply() {
        let key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY");
        let mut llm = AnthropicLlm::new(key).model("claude-3-5-haiku-latest");
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
