// SPDX-License-Identifier: Apache-2.0
//
//! **Google Gemini** (text) LLM — the `streamGenerateContent` REST client.
//!
//! A **(D)istinct** client (PROVIDERS.md §1): the Gemini text API
//! (`POST {base}/v1beta/models/{model}:streamGenerateContent?alt=sse`,
//! `x-goog-api-key` auth, a `candidates[].content.parts[]` streamed JSON schema) —
//! distinct from both OpenAI chat-completions and from the **Gemini *realtime***
//! client that already lives in `flowcat-core` (`flowcat_core::GeminiLive`). Behind
//! `llm-google`.
//!
//! ## Wire protocol (cross-checked against pipecat `services/google/llm.py`)
//!
//! Request body: `{ contents, systemInstruction?, tools? }`, where `contents` is a
//! list of `{ role: "user"|"model", parts: [{ text }] }`. flowcat's OpenAI-shaped
//! messages (`{role, content}`) are mapped: `system` → top-level `systemInstruction`,
//! `assistant` → `model`, everything else → `user`. Tools are wrapped as
//! `{ functionDeclarations: [{ name, description, parameters }] }`.
//!
//! Response (with `?alt=sse`) is SSE: `data: {json}` lines, each a
//! `GenerateContentResponse` whose `candidates[].content.parts[]` hold either a
//! `text` fragment (→ [`Frame::LlmText`]) or a whole `functionCall {name, args}`
//! (→ a [`Frame::FunctionCallsStarted`] entry — Gemini does **not** fragment tool-call
//! arguments the way OpenAI/Anthropic do).
//!
//! The decode is split into pure functions ([`parse_sse_line`], [`accumulate`]) so
//! the wire format is unit-tested without a network call.

use std::collections::VecDeque;

use async_trait::async_trait;
use futures::stream::{self, BoxStream};
use futures::StreamExt;
use serde_json::{json, Value};

use flowcat_core::error::{FlowcatError, Result};
use flowcat_core::processor::frame::{Frame, FunctionCall, LlmContext, StartParams};
use flowcat_core::realtime::gemini_schema_subset;
use flowcat_core::service::{LlmService, Tool};

/// Google Generative Language API base.
pub const GOOGLE_API_BASE: &str = "https://generativelanguage.googleapis.com";
/// Default model. Override with [`GoogleLlm::model`].
const DEFAULT_MODEL: &str = "gemini-3.5-flash";

/// Google Gemini (text) LLM service.
pub struct GoogleLlm {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
    tools: Vec<Tool>,
}

impl GoogleLlm {
    /// Construct bound to `api_key` (default base + a Gemini model).
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: GOOGLE_API_BASE.to_string(),
            model: DEFAULT_MODEL.to_string(),
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

    /// Override the model (default [`DEFAULT_MODEL`]).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Build the request body for `ctx` (pure — the seam the body test drives).
    /// Delegates to [`gemini_request_body`] with this service's tool set.
    fn request_body(&self, ctx: &LlmContext) -> Value {
        gemini_request_body(ctx, &self.tools)
    }
}

/// Build a Gemini `generateContent` request body from an [`LlmContext`] and a
/// service-level tool set (pure — the seam the body tests drive; shared by both the
/// AI-Studio [`GoogleLlm`] and the Vertex `GoogleVertexLlm` client, which speak the
/// identical wire format and differ only in URL + auth).
///
/// Maps OpenAI-shaped `{role, content}` messages to Gemini `contents`:
/// `system` → `systemInstruction`, `assistant` → role `model`, else role `user`.
pub(crate) fn gemini_request_body(ctx: &LlmContext, service_tools: &[Tool]) -> Value {
    {
        let mut system_parts: Vec<Value> = Vec::new();
        let mut contents: Vec<Value> = Vec::with_capacity(ctx.messages.len());
        for m in &ctx.messages {
            let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let text = m
                .get("content")
                .and_then(|c| c.as_str())
                .unwrap_or_default();
            if role == "system" {
                system_parts.push(json!({ "text": text }));
            } else {
                let g_role = if role == "assistant" { "model" } else { "user" };
                contents.push(json!({
                    "role": g_role,
                    "parts": [{ "text": text }],
                }));
            }
        }

        // Gemini requires a non-empty `contents`. The kickoff greeting sends only the
        // system prompt (which we lift into `systemInstruction`), leaving `contents`
        // empty → 400 "contents is not specified". Inject a minimal user turn so the
        // model produces the opening per the systemInstruction. Request-only — does not
        // touch the rolling conversation history.
        if contents.is_empty() {
            contents.push(json!({ "role": "user", "parts": [{ "text": "Hello" }] }));
        }
        let mut body = json!({ "contents": contents });
        if !system_parts.is_empty() {
            body["systemInstruction"] = json!({ "parts": system_parts });
        }
        // Tools: prefer the context's, else the service-level set, wrapped as a
        // single `{ functionDeclarations: [...] }`. `ctx.tools` are the brain's
        // provider-agnostic `Tool` JSON (with a `params` field), so they MUST go
        // through `tool_to_gemini` too — sending them verbatim 400s with
        // `Unknown name "params" at 'tools[0].function_declarations[0]'`.
        let mut decls: Vec<Value> = if !ctx.tools.is_empty() {
            ctx.tools
                .iter()
                .filter_map(|t| serde_json::from_value::<Tool>(t.clone()).ok())
                .map(|t| tool_to_gemini(&t))
                .collect()
        } else {
            service_tools.iter().map(tool_to_gemini).collect()
        };
        // Sanitize every declaration's `parameters` to Gemini's `Schema` subset.
        // Raw MCP `input_schema` carries `$schema`/`additionalProperties`/… that
        // Gemini rejects by closing the connection with `1007` — the same failure
        // the realtime path hit (#82); `generateContent` is no different. Applied
        // here so BOTH tool sources (context + service-level) are covered.
        for d in &mut decls {
            if let Some(obj) = d.as_object_mut() {
                if let Some(params) = obj.get("parameters") {
                    let sanitized = gemini_schema_subset(params);
                    obj.insert("parameters".to_string(), sanitized);
                }
            }
        }
        if !decls.is_empty() {
            body["tools"] = json!([{ "functionDeclarations": decls }]);
        }
        body
    }
}

/// Map a flowcat [`Tool`] to a Gemini `functionDeclarations` entry.
fn tool_to_gemini(t: &Tool) -> Value {
    json!({
        "name": t.name,
        "description": t.description,
        "parameters": t.params,
    })
}

#[async_trait]
impl LlmService for GoogleLlm {
    fn name(&self) -> &str {
        "google"
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        Ok(())
    }

    async fn run_llm<'a>(&'a mut self, ctx: &'a LlmContext) -> Result<BoxStream<'a, Frame>> {
        let body = self.request_body(ctx);
        // `?alt=sse` makes Gemini emit a `data: {json}` SSE stream (otherwise it
        // returns a single JSON array). The key rides the `x-goog-api-key` header
        // rather than `?key=` so it never lands in URLs/logs.
        let url = format!(
            "{}/v1beta/models/{}:streamGenerateContent?alt=sse",
            self.base_url, self.model
        );
        let resp = self
            .http
            .post(&url)
            .header("x-goog-api-key", &self.api_key)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| FlowcatError::Network(format!("google send: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(FlowcatError::Network(format!("google {status}: {text}")));
        }

        Ok(sse_to_frames(resp.bytes_stream()))
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
    pending: VecDeque<Frame>,
    /// Whole function calls collected across chunks (Gemini sends each `functionCall`
    /// complete in one part, so we just collect them in arrival order).
    calls: Vec<FunctionCall>,
}

impl SseState {
    /// Flush end-of-response frames once: ensure `LlmResponseStart` was emitted, then
    /// any collected function calls (`FunctionCallsStarted`), then `LlmResponseEnd`.
    fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        if !self.started {
            self.started = true;
            self.pending.push_back(Frame::LlmResponseStart);
        }
        if !self.calls.is_empty() {
            self.pending
                .push_back(Frame::FunctionCallsStarted(std::mem::take(&mut self.calls)));
        }
        self.pending.push_back(Frame::LlmResponseEnd);
    }
}

/// Turn a reqwest byte stream of Gemini SSE into a [`Frame`] stream. Owns the body
/// stream so it doesn't borrow the service. Shared by the AI-Studio [`GoogleLlm`] and
/// the Vertex client (identical `streamGenerateContent` SSE schema).
pub(crate) fn sse_to_frames<S, B, E>(byte_stream: S) -> BoxStream<'static, Frame>
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
        calls: Vec::new(),
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
                        if let Some(v) = parse_sse_line(line.trim_end()) {
                            if !st.started {
                                st.started = true;
                                st.pending.push_back(Frame::LlmResponseStart);
                            }
                            accumulate(&v, &mut st.calls, &mut st.pending);
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

/// Parse one Gemini SSE line into its JSON chunk, if any (pure — the wire-fixture
/// seam). Gemini has no `[DONE]` sentinel; the stream simply ends.
fn parse_sse_line(line: &str) -> Option<Value> {
    let data = line.trim().strip_prefix("data:")?.trim();
    if data.is_empty() {
        return None;
    }
    serde_json::from_str::<Value>(data).ok()
}

/// Fold one Gemini `GenerateContentResponse` chunk into the running state (pure — the
/// wire-fixture seam). Text parts push [`Frame::LlmText`]; whole `functionCall` parts
/// are collected for the terminal `FunctionCallsStarted`.
fn accumulate(chunk: &Value, calls: &mut Vec<FunctionCall>, pending: &mut VecDeque<Frame>) {
    let Some(candidates) = chunk.get("candidates").and_then(|c| c.as_array()) else {
        return;
    };
    for cand in candidates {
        let Some(parts) = cand
            .get("content")
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.as_array())
        else {
            continue;
        };
        for part in parts {
            if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                if !text.is_empty() {
                    pending.push_back(Frame::LlmText(text.to_string()));
                }
            } else if let Some(fc) = part.get("functionCall") {
                let name = fc
                    .get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or("")
                    .to_string();
                let args = fc.get("args").cloned().unwrap_or_else(|| json!({}));
                // Gemini function calls carry no provider id; synthesise a stable one
                // from the call ordinal so downstream result-matching has a handle.
                let id = format!("gemini_call_{}", calls.len());
                calls.push(FunctionCall {
                    function_name: name,
                    tool_call_id: id,
                    arguments: args,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_body_injects_user_turn_when_only_system() {
        // Kickoff greeting: only a system message. Gemini rejects empty `contents`,
        // so we inject a synthetic user turn (the system prompt drives the greeting).
        let llm = GoogleLlm::new("k");
        let ctx = LlmContext {
            messages: vec![json!({"role": "system", "content": "greet briefly"})],
            tools: vec![],
        };
        let body = llm.request_body(&ctx);
        assert_eq!(
            body["systemInstruction"]["parts"][0]["text"],
            "greet briefly"
        );
        let contents = body["contents"].as_array().expect("contents array");
        assert_eq!(contents.len(), 1, "must not be empty");
        assert_eq!(contents[0]["role"], "user");
    }

    #[test]
    fn request_body_maps_ctx_tools_to_gemini_shape() {
        // ctx.tools are brain `Tool` JSON (carry a `params` field). They MUST be
        // mapped to Gemini's `parameters`, not sent verbatim (which 400s on `params`).
        let llm = GoogleLlm::new("k");
        let ctx = LlmContext {
            messages: vec![json!({"role": "user", "content": "hi"})],
            tools: vec![json!({
                "name": "end_call",
                "description": "end the call",
                "params": {"type": "object"}
            })],
        };
        let body = llm.request_body(&ctx);
        let decl = &body["tools"][0]["functionDeclarations"][0];
        assert_eq!(decl["name"], "end_call");
        assert!(decl.get("params").is_none(), "must not carry raw `params`");
        assert!(decl.get("parameters").is_some(), "must expose `parameters`");
    }

    #[test]
    fn request_body_maps_roles_system_and_tools() {
        let mut llm = GoogleLlm::new("k");
        llm.set_tools(vec![Tool {
            name: "end_call".into(),
            description: "end".into(),
            params: json!({"type": "object"}),
        }]);
        let ctx = LlmContext {
            messages: vec![
                json!({"role": "system", "content": "be brief"}),
                json!({"role": "user", "content": "hi"}),
                json!({"role": "assistant", "content": "hello"}),
            ],
            tools: vec![],
        };
        let body = llm.request_body(&ctx);
        // System lifted into systemInstruction.
        assert_eq!(body["systemInstruction"]["parts"][0]["text"], "be brief");
        // user stays user, assistant → model.
        assert_eq!(body["contents"][0]["role"], "user");
        assert_eq!(body["contents"][0]["parts"][0]["text"], "hi");
        assert_eq!(body["contents"][1]["role"], "model");
        // Tool wrapped under functionDeclarations.
        assert_eq!(
            body["tools"][0]["functionDeclarations"][0]["name"],
            "end_call"
        );
        assert_eq!(
            body["tools"][0]["functionDeclarations"][0]["parameters"]["type"],
            "object"
        );
    }

    #[test]
    fn request_body_sanitizes_tool_parameters_to_gemini_subset() {
        // A raw MCP-style schema with keys Gemini rejects (`$schema`,
        // `additionalProperties`, `$ref`) — must be stripped so `generateContent`
        // isn't `1007`-closed, exactly like the realtime path (#82).
        let raw = json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "city": { "type": "string", "$comment": "drop me" }
            },
            "required": ["city"]
        });
        // Cover both tool sources: service-level (tool_to_gemini) …
        let mut llm = GoogleLlm::new("k");
        llm.set_tools(vec![Tool {
            name: "lookup".into(),
            description: "d".into(),
            params: raw.clone(),
        }]);
        let svc_body = llm.request_body(&LlmContext {
            messages: vec![],
            tools: vec![],
        });
        let svc_params = &svc_body["tools"][0]["functionDeclarations"][0]["parameters"];
        assert!(
            svc_params.get("$schema").is_none(),
            "service tool: $schema kept"
        );
        assert!(svc_params.get("additionalProperties").is_none());
        assert_eq!(svc_params["type"], "object");
        assert_eq!(svc_params["required"][0], "city");
        assert!(svc_params["properties"]["city"].get("$comment").is_none());

        // … and context tools (the brain's `Tool` JSON — carries `params`, mapped
        // through `tool_to_gemini` like the service-level set; NOT passed verbatim).
        let ctx_body = llm.request_body(&LlmContext {
            messages: vec![],
            tools: vec![json!({ "name": "ctxtool", "description": "d", "params": raw })],
        });
        let ctx_params = &ctx_body["tools"][0]["functionDeclarations"][0]["parameters"];
        assert_eq!(
            ctx_body["tools"][0]["functionDeclarations"][0]["name"],
            "ctxtool"
        );
        assert!(
            ctx_params.get("$schema").is_none(),
            "context tool: $schema kept"
        );
        assert!(ctx_params.get("additionalProperties").is_none());
        assert_eq!(ctx_params["type"], "object");
    }

    #[test]
    fn base_url_and_model_overrides_apply() {
        let llm = GoogleLlm::new("k")
            .base_url("https://gw.example.com/")
            .model("gemini-1.5-pro");
        assert_eq!(llm.base_url, "https://gw.example.com");
        assert_eq!(llm.model, "gemini-1.5-pro");
    }

    #[test]
    fn parse_sse_line_extracts_chunk_and_skips_noise() {
        assert!(parse_sse_line("").is_none());
        assert!(parse_sse_line(": comment").is_none());
        let v = parse_sse_line(r#"data: {"candidates":[{"content":{"parts":[{"text":"hi"}]}}]}"#)
            .expect("chunk");
        assert_eq!(v["candidates"][0]["content"]["parts"][0]["text"], "hi");
    }

    #[test]
    fn accumulate_emits_text_parts_in_order() {
        let mut calls = Vec::new();
        let mut pending = VecDeque::new();
        for token in ["Hello", " ", "world"] {
            let chunk = json!({
                "candidates": [{ "content": { "parts": [{ "text": token }] } }]
            });
            accumulate(&chunk, &mut calls, &mut pending);
        }
        let texts: Vec<String> = pending
            .into_iter()
            .filter_map(|f| match f {
                Frame::LlmText(t) => Some(t),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["Hello", " ", "world"]);
        assert!(calls.is_empty());
    }

    #[test]
    fn accumulate_collects_a_whole_function_call() {
        let mut calls = Vec::new();
        let mut pending = VecDeque::new();
        let chunk = json!({
            "candidates": [{ "content": { "parts": [{
                "functionCall": { "name": "book", "args": { "day": "mon" } }
            }] } }]
        });
        accumulate(&chunk, &mut calls, &mut pending);
        assert!(
            pending.is_empty(),
            "function calls are not streamed as text"
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function_name, "book");
        assert_eq!(calls[0].arguments["day"], "mon");
        assert_eq!(calls[0].tool_call_id, "gemini_call_0");
    }

    #[tokio::test]
    async fn sse_to_frames_decodes_text_then_emits_framing() {
        let fixture = concat!(
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hello\"}]}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\" there\"}]}}]}\n\n",
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

    /// Live smoke (requires `GOOGLE_API_KEY`): stream a one-token reply. Run:
    /// `GOOGLE_API_KEY=… cargo test -p flowcat-services --features llm-google -- --ignored google_live`
    #[tokio::test]
    #[ignore = "requires GOOGLE_API_KEY"]
    async fn google_live_streams_a_reply() {
        let key = std::env::var("GOOGLE_API_KEY").expect("GOOGLE_API_KEY");
        let mut llm = GoogleLlm::new(key).model("gemini-3.5-flash");
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
