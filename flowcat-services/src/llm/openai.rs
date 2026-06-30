// SPDX-License-Identifier: Apache-2.0
//
//! OpenAI (and OpenRouter / OpenAI-compatible) streaming chat completions.
//!
//! Wire protocol (cross-checked against pipecat `services/openai/llm.py`): POST
//! `{base}/chat/completions` with `Authorization: Bearer <key>` and a body
//! `{ model, messages, tools?, stream: true }`. The response is **Server-Sent
//! Events** — `data: {json}` lines, each a chat-completion chunk
//! `{ "choices": [ { "delta": { "content": "…" } } ] }`, terminated by
//! `data: [DONE]`. Text deltas become [`Frame::LlmText`]; a `tool_calls` delta is
//! assembled into a [`Frame::FunctionCallsStarted`]. The whole response is framed
//! by [`Frame::LlmResponseStart`] / [`Frame::LlmResponseEnd`].
//!
//! The SSE→frame mapping is split into pure functions ([`parse_sse_line`],
//! [`accumulate`]) so the wire format is unit-tested without a network call.

use std::collections::BTreeMap;

use async_trait::async_trait;
use futures::stream::{self, BoxStream};
use futures::StreamExt;
use serde_json::{json, Value};

use flowcat_core::error::{FlowcatError, Result};
use flowcat_core::processor::frame::{Frame, FunctionCall, LlmContext, StartParams};
use flowcat_core::processor::metrics::{LlmTokenUsage, MetricsData};
use flowcat_core::service::{LlmService, Tool};

/// OpenAI's default API base. A `base_url` override points the same client at
/// OpenRouter (`https://openrouter.ai/api/v1`) or a self-hosted gateway. Only an
/// operator-supplied base is honoured (it is config, never request-derived), so
/// there is no request-controlled SSRF surface.
pub const OPENAI_API_BASE: &str = "https://api.openai.com/v1";

/// Builder for [`OpenAiLlm`].
#[derive(Debug, Clone)]
pub struct OpenAiLlmBuilder {
    api_key: String,
    base_url: String,
    model: String,
}

impl OpenAiLlmBuilder {
    /// Start a builder bound to `api_key` (default base = OpenAI, model gpt-4o).
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: OPENAI_API_BASE.to_string(),
            model: "gpt-4o".to_string(),
        }
    }

    /// Override the API base (OpenRouter / a compatible gateway). Trailing
    /// slashes are trimmed.
    pub fn base_url(mut self, base: impl Into<String>) -> Self {
        self.base_url = base.into().trim_end_matches('/').to_string();
        self
    }

    /// Override the model (default `gpt-4o`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Build the client.
    pub fn build(self) -> OpenAiLlm {
        OpenAiLlm {
            http: reqwest::Client::new(),
            api_key: self.api_key,
            base_url: self.base_url,
            model: self.model,
            tools: Vec::new(),
        }
    }
}

/// A streaming OpenAI chat-completions LLM service.
pub struct OpenAiLlm {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
    tools: Vec<Tool>,
}

impl OpenAiLlm {
    /// Construct with default base + model. Use [`OpenAiLlmBuilder`] otherwise.
    pub fn new(api_key: impl Into<String>) -> Self {
        OpenAiLlmBuilder::new(api_key).build()
    }

    /// Build the request body for `ctx` (pure — the seam the body test drives).
    fn request_body(&self, ctx: &LlmContext) -> Value {
        let mut body = json!({
            "model": self.model,
            "messages": ctx.messages,
            "stream": true,
            // Ask for the trailing usage chunk so the cascaded leg can report LLM
            // token usage (parsed in `accumulate`, emitted as `Frame::Metrics`).
            "stream_options": { "include_usage": true },
        });
        // Tools: prefer the context's, else the service-level set. OpenAI wants
        // each tool as `{ "type": "function", "function": { name, description,
        // parameters } }`. `ctx.tools` are the brain's provider-agnostic
        // `ToolDecl` JSON, so they MUST go through `tool_to_openai` too — sending
        // them verbatim 400s with "Missing required parameter: 'tools[0].type'".
        let tools: Vec<Value> = if !ctx.tools.is_empty() {
            ctx.tools
                .iter()
                .filter_map(|t| serde_json::from_value::<Tool>(t.clone()).ok())
                .map(|t| tool_to_openai(&t))
                .collect()
        } else {
            self.tools.iter().map(tool_to_openai).collect()
        };
        if !tools.is_empty() {
            body["tools"] = Value::Array(tools);
        }
        body
    }
}

/// Map a flowcat [`Tool`] to the OpenAI tool schema.
fn tool_to_openai(t: &Tool) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.params,
        }
    })
}

#[async_trait]
impl LlmService for OpenAiLlm {
    fn name(&self) -> &str {
        "openai"
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        Ok(())
    }

    async fn run_llm<'a>(&'a mut self, ctx: &'a LlmContext) -> Result<BoxStream<'a, Frame>> {
        let body = self.request_body(ctx);
        let url = format!("{}/chat/completions", self.base_url);
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| FlowcatError::Network(format!("openai send: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(FlowcatError::Network(format!("openai {status}: {text}")));
        }

        // Build an owned byte stream so the returned frame stream does not borrow
        // `self` (only the connection/body, which the stream owns). The SSE
        // bytes are buffered line-by-line and mapped to frames as they arrive.
        // The model name is cloned in so the trailing usage chunk can be reported
        // as `Frame::Metrics(LlmUsage { model, … })`.
        Ok(sse_to_frames(resp.bytes_stream(), self.model.clone()))
    }

    fn set_tools(&mut self, tools: Vec<Tool>) {
        self.tools = tools;
    }
}

/// State carried across SSE chunks by the [`sse_to_frames`] unfold.
struct SseState {
    buf: String,
    started: bool,
    finished: bool,
    pending: std::collections::VecDeque<Frame>,
    tool_acc: BTreeMap<u64, ToolAcc>,
    /// The model name, echoed into the emitted `LlmUsage` metric.
    model: String,
    /// Token usage from the trailing `{"usage": …}` chunk (present only when the
    /// request set `stream_options.include_usage`). Emitted on [`finish`].
    usage: Option<LlmTokenUsage>,
}

impl SseState {
    /// Flush the end-of-response frames into `pending` once: emit
    /// `LlmResponseStart` first if the stream produced nothing, then any
    /// assembled tool calls (`FunctionCallsStarted`), then the LLM token-usage
    /// metric (if the trailing usage chunk arrived), then `LlmResponseEnd`.
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
        if let Some(tokens) = self.usage.take() {
            self.pending
                .push_back(Frame::Metrics(vec![MetricsData::LlmUsage {
                    processor: "openai".to_string(),
                    model: Some(self.model.clone()),
                    tokens,
                }]));
        }
        self.pending.push_back(Frame::LlmResponseEnd);
    }
}

/// Parse an OpenAI streaming `usage` object into [`LlmTokenUsage`]. The trailing
/// chunk (when `stream_options.include_usage` is set) carries
/// `{prompt_tokens, completion_tokens, total_tokens, …}` with an empty `choices`.
fn parse_usage(usage: &Value) -> LlmTokenUsage {
    let get = |k: &str| usage.get(k).and_then(Value::as_u64).unwrap_or(0);
    let opt = |path: &[&str]| -> Option<u64> {
        let mut cur = usage;
        for k in path {
            cur = cur.get(k)?;
        }
        cur.as_u64()
    };
    LlmTokenUsage {
        prompt_tokens: get("prompt_tokens"),
        completion_tokens: get("completion_tokens"),
        total_tokens: get("total_tokens"),
        cache_read_input_tokens: opt(&["prompt_tokens_details", "cached_tokens"]),
        cache_creation_input_tokens: None,
        reasoning_tokens: opt(&["completion_tokens_details", "reasoning_tokens"]),
    }
}

/// Turn a reqwest byte stream of SSE into a [`Frame`] stream:
/// `LlmResponseStart`, then `LlmText`* / `FunctionCallsStarted`, then
/// `LlmResponseEnd`. Owns the body stream so it doesn't borrow the service.
/// Generic over the chunk type (`AsRef<[u8]>`) so we don't name `bytes::Bytes`
/// (no extra dep).
fn sse_to_frames<S, B, E>(byte_stream: S, model: String) -> BoxStream<'static, Frame>
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
        pending: std::collections::VecDeque::new(),
        tool_acc: BTreeMap::new(),
        model,
        usage: None,
    };
    stream::unfold((inner, st), |(mut inner, mut st)| async move {
        loop {
            // Drain any frames queued from the last parsed chunk first.
            if let Some(f) = st.pending.pop_front() {
                return Some((f, (inner, st)));
            }
            if st.finished {
                return None;
            }
            match inner.next().await {
                Some(Ok(bytes)) => {
                    st.buf.push_str(&String::from_utf8_lossy(bytes.as_ref()));
                    // Process complete lines.
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
                                // The trailing `include_usage` chunk carries a
                                // top-level `usage` (and an empty `choices`); stash
                                // it to emit on finish.
                                if let Some(u) = v.get("usage").filter(|u| !u.is_null()) {
                                    st.usage = Some(parse_usage(u));
                                }
                                accumulate(&v, &mut st.tool_acc, &mut st.pending);
                            }
                        }
                    }
                    if let Some(f) = st.pending.pop_front() {
                        return Some((f, (inner, st)));
                    }
                    // else loop to read more bytes
                }
                // Stream ended (or errored) without an explicit [DONE].
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

struct ToolAcc {
    id: String,
    name: String,
    args: String,
}

/// One parsed SSE line.
enum SseEvent {
    /// A `data: {json}` chat-completion chunk.
    Chunk(Value),
    /// The terminal `data: [DONE]`.
    Done,
    /// A blank line / comment / unrelated line.
    None,
}

/// Parse one SSE line into an [`SseEvent`] (pure — the wire-fixture seam).
fn parse_sse_line(line: &str) -> SseEvent {
    let line = line.trim();
    let Some(data) = line.strip_prefix("data:") else {
        return SseEvent::None;
    };
    let data = data.trim();
    if data == "[DONE]" {
        return SseEvent::Done;
    }
    match serde_json::from_str::<Value>(data) {
        Ok(v) => SseEvent::Chunk(v),
        Err(_) => SseEvent::None,
    }
}

/// Fold one chat-completion chunk into the running state, pushing any emitted
/// frames (text deltas) onto `pending` and accumulating tool-call deltas
/// (pure — the wire-fixture seam).
fn accumulate(
    chunk: &Value,
    tool_acc: &mut BTreeMap<u64, ToolAcc>,
    pending: &mut std::collections::VecDeque<Frame>,
) {
    let Some(choice) = chunk
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|c| c.first())
    else {
        return;
    };
    let delta = choice.get("delta");
    // Text delta → LlmText.
    if let Some(text) = delta
        .and_then(|d| d.get("content"))
        .and_then(|c| c.as_str())
    {
        if !text.is_empty() {
            pending.push_back(Frame::LlmText(text.to_string()));
        }
    }
    // Tool-call deltas → accumulate by index (OpenAI streams name then args).
    if let Some(calls) = delta
        .and_then(|d| d.get("tool_calls"))
        .and_then(|t| t.as_array())
    {
        for call in calls {
            let idx = call.get("index").and_then(|i| i.as_u64()).unwrap_or(0);
            let entry = tool_acc.entry(idx).or_insert_with(|| ToolAcc {
                id: String::new(),
                name: String::new(),
                args: String::new(),
            });
            if let Some(id) = call.get("id").and_then(|i| i.as_str()) {
                if !id.is_empty() {
                    entry.id = id.to_string();
                }
            }
            if let Some(func) = call.get("function") {
                if let Some(name) = func.get("name").and_then(|n| n.as_str()) {
                    entry.name.push_str(name);
                }
                if let Some(args) = func.get("arguments").and_then(|a| a.as_str()) {
                    entry.args.push_str(args);
                }
            }
        }
    }
}

/// Assemble the accumulated tool calls into a single [`Frame::FunctionCallsStarted`],
/// if any (pure — the wire-fixture seam).
fn drain_tool_calls(tool_acc: &mut BTreeMap<u64, ToolAcc>) -> Option<Frame> {
    if tool_acc.is_empty() {
        return None;
    }
    let calls: Vec<FunctionCall> = std::mem::take(tool_acc)
        .into_values()
        .map(|t| FunctionCall {
            function_name: t.name,
            tool_call_id: t.id,
            arguments: serde_json::from_str(&t.args).unwrap_or(Value::Null),
        })
        .collect();
    Some(Frame::FunctionCallsStarted(calls))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_body_carries_model_messages_stream_and_tools() {
        let mut llm = OpenAiLlm::new("k");
        llm.set_tools(vec![Tool {
            name: "end_call".into(),
            description: "end".into(),
            params: json!({"type": "object"}),
        }]);
        let ctx = LlmContext {
            messages: vec![json!({"role": "user", "content": "hi"})],
            tools: vec![],
        };
        let body = llm.request_body(&ctx);
        assert_eq!(body["model"], "gpt-4o");
        assert_eq!(body["stream"], true);
        assert_eq!(body["messages"][0]["content"], "hi");
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["function"]["name"], "end_call");
    }

    #[test]
    fn ctx_tools_are_wrapped_in_openai_shape() {
        // Regression: the cascaded path puts the brain's `ToolDecl` JSON into
        // `ctx.tools` (no `type` wrapper). `request_body` MUST still wrap each as
        // `{type:function, function:{…}}` — sending them verbatim 400s with
        // "Missing required parameter: 'tools[0].type'" (the live cascaded bug).
        let llm = OpenAiLlm::new("k");
        let ctx = LlmContext {
            messages: vec![json!({"role": "user", "content": "hi"})],
            // Exactly what cascaded.rs seeds via serde_json::to_value(ToolDecl).
            tools: vec![json!({
                "name": "go_book",
                "description": "book it",
                "params": {"type": "object"}
            })],
        };
        let body = llm.request_body(&ctx);
        assert_eq!(
            body["tools"][0]["type"], "function",
            "ctx tools need type:function"
        );
        assert_eq!(body["tools"][0]["function"]["name"], "go_book");
        assert_eq!(body["tools"][0]["function"]["parameters"]["type"], "object");
    }

    #[test]
    fn base_url_override_points_at_openrouter() {
        let llm = OpenAiLlmBuilder::new("k")
            .base_url("https://openrouter.ai/api/v1/")
            .model("anthropic/claude-3.5")
            .build();
        assert_eq!(llm.base_url, "https://openrouter.ai/api/v1");
        assert_eq!(llm.model, "anthropic/claude-3.5");
    }

    #[test]
    fn parse_sse_line_classifies_chunk_done_and_noise() {
        assert!(matches!(parse_sse_line("data: [DONE]"), SseEvent::Done));
        assert!(matches!(parse_sse_line(""), SseEvent::None));
        assert!(matches!(parse_sse_line(": comment"), SseEvent::None));
        match parse_sse_line(r#"data: {"choices":[{"delta":{"content":"hi"}}]}"#) {
            SseEvent::Chunk(v) => assert_eq!(v["choices"][0]["delta"]["content"], "hi"),
            _ => panic!("expected chunk"),
        }
    }

    #[test]
    fn accumulate_emits_text_deltas_in_order() {
        let mut acc = BTreeMap::new();
        let mut pending = std::collections::VecDeque::new();
        for token in ["Hello", " ", "world"] {
            let chunk = json!({"choices":[{"delta":{"content": token}}]});
            accumulate(&chunk, &mut acc, &mut pending);
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
    fn accumulate_assembles_streamed_tool_call() {
        let mut acc = BTreeMap::new();
        let mut pending = std::collections::VecDeque::new();
        // OpenAI streams the tool call across chunks: id+name first, then args.
        let c1 = json!({"choices":[{"delta":{"tool_calls":[
            {"index":0,"id":"call_1","function":{"name":"book","arguments":"{\"da"}}
        ]}}]});
        let c2 = json!({"choices":[{"delta":{"tool_calls":[
            {"index":0,"function":{"arguments":"y\":\"mon\"}"}}
        ]}}]});
        accumulate(&c1, &mut acc, &mut pending);
        accumulate(&c2, &mut acc, &mut pending);
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

    #[test]
    fn request_body_asks_for_include_usage() {
        let llm = OpenAiLlm::new("k");
        let ctx = LlmContext {
            messages: vec![json!({"role": "user", "content": "hi"})],
            tools: vec![],
        };
        let body = llm.request_body(&ctx);
        assert_eq!(body["stream_options"]["include_usage"], true);
    }

    #[test]
    fn parse_usage_maps_openai_token_fields() {
        let u = json!({
            "prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19,
            "prompt_tokens_details": { "cached_tokens": 4 },
            "completion_tokens_details": { "reasoning_tokens": 3 }
        });
        let t = parse_usage(&u);
        assert_eq!(t.prompt_tokens, 12);
        assert_eq!(t.completion_tokens, 7);
        assert_eq!(t.total_tokens, 19);
        assert_eq!(t.cache_read_input_tokens, Some(4));
        assert_eq!(t.reasoning_tokens, Some(3));
    }

    #[tokio::test]
    async fn sse_stream_emits_llm_usage_metric_from_trailing_chunk() {
        // A text delta, then the `include_usage` trailing chunk (empty `choices`,
        // top-level `usage`), then `[DONE]`. The cascaded leg must surface the
        // usage as `Frame::Metrics(LlmUsage)` so the run records tokens.
        let chunks: Vec<std::result::Result<Vec<u8>, std::convert::Infallible>> = vec![
            Ok(b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n".to_vec()),
            Ok(b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":7,\"total_tokens\":19}}\n".to_vec()),
            Ok(b"data: [DONE]\n".to_vec()),
        ];
        let mut frames = sse_to_frames(stream::iter(chunks), "gpt-4o".to_string());
        let mut saw_text = false;
        let mut usage = None;
        while let Some(f) = frames.next().await {
            match f {
                Frame::LlmText(_) => saw_text = true,
                Frame::Metrics(items) => {
                    for m in items {
                        if let MetricsData::LlmUsage { model, tokens, .. } = m {
                            usage = Some((model, tokens));
                        }
                    }
                }
                _ => {}
            }
        }
        assert!(saw_text, "text delta still streams");
        let (model, tokens) = usage.expect("a usage metric is emitted");
        assert_eq!(model.as_deref(), Some("gpt-4o"));
        assert_eq!(tokens.prompt_tokens, 12);
        assert_eq!(tokens.completion_tokens, 7);
        assert_eq!(tokens.total_tokens, 19);
    }

    /// Live smoke (requires `OPENAI_API_KEY`): stream a one-token reply. Run:
    /// `OPENAI_API_KEY=… cargo test -p flowcat-services --features llm-openai -- --ignored openai_live`
    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY"]
    async fn openai_live_streams_a_reply() {
        let key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY");
        let mut llm = OpenAiLlmBuilder::new(key).model("gpt-4o-mini").build();
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
