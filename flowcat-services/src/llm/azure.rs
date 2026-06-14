// SPDX-License-Identifier: Apache-2.0
//
//! **Azure OpenAI** LLM — the OpenAI-compatible cohort's **one auth seam**.
//!
//! Azure OpenAI speaks the **same** chat-completions streaming **wire** as
//! [`OpenAiLlm`](super::OpenAiLlm) (verified against pipecat
//! `services/azure/llm.py`, which is `class AzureLLMService(OpenAILLMService)`), but
//! it differs from every other (W) wrapper on **two** axes the shared [`OpenAiLlm`]
//! client cannot express:
//!
//! 1. **Auth header** — Azure uses `api-key: <key>`, *not* `Authorization: Bearer
//!    <key>` (bearer is reserved for AAD tokens). [`OpenAiLlm`] hard-codes
//!    `bearer_auth`, so a key-auth caller cannot reuse it.
//! 2. **URL shape** — the request path is
//!    `{endpoint}/openai/deployments/{deployment}/chat/completions?api-version=<ver>`,
//!    i.e. the model lives in the **path** (the deployment name) plus a required
//!    `?api-version=` query — not `{base}/chat/completions` with the model in the body.
//!
//! Because the shared `OpenAiLlm` impl is intentionally **not modified** by this
//! fan-out, [`AzureLlm`] is a **self-contained**
//! [`LlmService`] that POSTs the identical OpenAI body but with the Azure header +
//! URL, and decodes the **identical** chat-completions SSE (text deltas → `LlmText`,
//! streamed `tool_calls` → `FunctionCallsStarted`). The decode mirrors the OpenAI ref
//! impl exactly — **flagged for review** as the cohort's single bespoke seam.
//! Behind the `llm-azure` feature (which enables `llm-openai`).

use std::collections::BTreeMap;

use async_trait::async_trait;
use futures::stream::{self, BoxStream};
use futures::StreamExt;
use serde_json::{json, Value};

use flowcat_core::error::{FlowcatError, Result};
use flowcat_core::processor::frame::{Frame, FunctionCall, LlmContext, StartParams};
use flowcat_core::service::{LlmService, Tool};

/// Default Azure OpenAI REST API version (matches pipecat's default).
pub const AZURE_DEFAULT_API_VERSION: &str = "2024-09-01-preview";
/// Default Azure deployment / model name (a placeholder — Azure deployments are
/// operator-named, so this is expected to be set explicitly via [`AzureLlm::new`]).
pub const AZURE_DEFAULT_DEPLOYMENT: &str = "gpt-4o";

/// Azure OpenAI LLM service.
///
/// Unlike the other (W) wrappers it does **not** hold an [`OpenAiLlm`]: Azure's
/// `api-key` header + deployment-path URL cannot be expressed through that client's
/// `bearer_auth` + `{base}/chat/completions` shape, so this is a thin standalone
/// client over the same wire.
pub struct AzureLlm {
    http: reqwest::Client,
    api_key: String,
    /// Resource endpoint, e.g. `https://my-resource.openai.azure.com` (no trailing slash).
    endpoint: String,
    /// Azure deployment name (stands in for the model).
    deployment: String,
    api_version: String,
    tools: Vec<Tool>,
}

impl AzureLlm {
    /// Construct against `endpoint` (e.g. `https://my-resource.openai.azure.com`)
    /// with `api_key` and the `deployment` name. Uses [`AZURE_DEFAULT_API_VERSION`].
    pub fn new(
        api_key: impl Into<String>,
        endpoint: impl Into<String>,
        deployment: impl Into<String>,
    ) -> Self {
        Self::with_api_version(api_key, endpoint, deployment, AZURE_DEFAULT_API_VERSION)
    }

    /// Construct with an explicit `api_version`.
    pub fn with_api_version(
        api_key: impl Into<String>,
        endpoint: impl Into<String>,
        deployment: impl Into<String>,
        api_version: impl Into<String>,
    ) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: api_key.into(),
            endpoint: endpoint.into().trim_end_matches('/').to_string(),
            deployment: deployment.into(),
            api_version: api_version.into(),
            tools: Vec::new(),
        }
    }

    /// The full chat-completions URL for this deployment (pure — the URL-seam test
    /// drives it). Azure puts the deployment in the path + an `api-version` query.
    fn chat_url(&self) -> String {
        format!(
            "{}/openai/deployments/{}/chat/completions?api-version={}",
            self.endpoint, self.deployment, self.api_version
        )
    }

    /// Build the request body for `ctx` (pure — identical OpenAI chat-completions
    /// shape; the model lives in the URL, so it is not repeated here).
    fn request_body(&self, ctx: &LlmContext) -> Value {
        let mut body = json!({
            "messages": ctx.messages,
            "stream": true,
        });
        let tools: Vec<Value> = if !ctx.tools.is_empty() {
            ctx.tools.clone()
        } else {
            self.tools.iter().map(tool_to_openai).collect()
        };
        if !tools.is_empty() {
            body["tools"] = Value::Array(tools);
        }
        body
    }
}

/// Map a flowcat [`Tool`] to the OpenAI tool schema (Azure shares it).
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
impl LlmService for AzureLlm {
    fn name(&self) -> &str {
        "azure"
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        Ok(())
    }

    async fn run_llm<'a>(&'a mut self, ctx: &'a LlmContext) -> Result<BoxStream<'a, Frame>> {
        let body = self.request_body(ctx);
        let resp = self
            .http
            .post(self.chat_url())
            // THE SEAM: Azure key auth is the `api-key` header, not bearer.
            .header("api-key", &self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| FlowcatError::Network(format!("azure send: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(FlowcatError::Network(format!("azure {status}: {text}")));
        }
        Ok(sse_to_frames(resp.bytes_stream()))
    }

    fn set_tools(&mut self, tools: Vec<Tool>) {
        self.tools = tools;
    }
}

// ---------------------------------------------------------------------------
// SSE → Frame decode. Azure streams the identical chat-completions SSE as OpenAI;
// this mirrors the OpenAI ref impl (text deltas + assembled tool calls). The pure
// `parse_sse_line` / `accumulate` seam is unit-tested without a network call.
// ---------------------------------------------------------------------------

struct ToolAcc {
    id: String,
    name: String,
    args: String,
}

struct SseState {
    started: bool,
    finished: bool,
    pending: std::collections::VecDeque<Frame>,
    tool_acc: BTreeMap<u64, ToolAcc>,
}

impl SseState {
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

/// One parsed SSE line.
enum SseEvent {
    Chunk(Value),
    Done,
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

/// Fold one chat-completion chunk into the running state (pure — the wire seam).
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
    if let Some(text) = delta
        .and_then(|d| d.get("content"))
        .and_then(|c| c.as_str())
    {
        if !text.is_empty() {
            pending.push_back(Frame::LlmText(text.to_string()));
        }
    }
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

/// Assemble accumulated tool calls into one [`Frame::FunctionCallsStarted`] (pure).
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

/// Turn a reqwest byte stream of SSE into a [`Frame`] stream. Owns the body stream
/// so it does not borrow the service.
fn sse_to_frames<S, B, E>(byte_stream: S) -> BoxStream<'static, Frame>
where
    S: futures::Stream<Item = std::result::Result<B, E>> + Send + 'static,
    B: AsRef<[u8]> + Send + 'static,
    E: Send + 'static,
{
    let inner = Box::pin(byte_stream);
    let st = SseState {
        started: false,
        finished: false,
        pending: std::collections::VecDeque::new(),
        tool_acc: BTreeMap::new(),
    };
    stream::unfold(
        (inner, st, String::new()),
        |(mut inner, mut st, mut buf)| async move {
            loop {
                if let Some(f) = st.pending.pop_front() {
                    return Some((f, (inner, st, buf)));
                }
                if st.finished {
                    return None;
                }
                match inner.next().await {
                    Some(Ok(bytes)) => {
                        buf.push_str(&String::from_utf8_lossy(bytes.as_ref()));
                        while let Some(nl) = buf.find('\n') {
                            let line: String = buf.drain(..=nl).collect();
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
                            return Some((f, (inner, st, buf)));
                        }
                    }
                    Some(Err(_)) | None => {
                        st.finish();
                        if let Some(f) = st.pending.pop_front() {
                            return Some((f, (inner, st, buf)));
                        }
                        return None;
                    }
                }
            }
        },
    )
    .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn azure_url_uses_deployment_path_and_api_version() {
        let llm = AzureLlm::new("k", "https://my-resource.openai.azure.com/", "gpt-4o-mini");
        assert_eq!(
            llm.chat_url(),
            "https://my-resource.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-09-01-preview"
        );
        assert_eq!(llm.name(), "azure");
        assert_eq!(AZURE_DEFAULT_API_VERSION, "2024-09-01-preview");
        assert_eq!(AZURE_DEFAULT_DEPLOYMENT, "gpt-4o");
    }

    #[test]
    fn azure_uses_api_key_not_bearer_and_model_is_in_url() {
        // The auth seam: the client stores the key for the `api-key` header (set in
        // `run_llm`) and the body carries no `model` (it is in the URL). This is the
        // one cohort member that does NOT use bearer auth.
        let llm = AzureLlm::with_api_version(
            "secret",
            "https://r.openai.azure.com",
            "dep",
            "2025-01-01-preview",
        );
        assert_eq!(llm.api_key, "secret");
        assert_eq!(llm.api_version, "2025-01-01-preview");
        let ctx = LlmContext {
            messages: vec![json!({"role":"user","content":"hi"})],
            tools: vec![],
        };
        let body = llm.request_body(&ctx);
        assert_eq!(body["stream"], true);
        assert_eq!(body["messages"][0]["content"], "hi");
        assert!(
            body.get("model").is_none(),
            "model lives in the URL, not the body"
        );
    }

    #[test]
    fn parse_and_accumulate_match_the_openai_wire() {
        assert!(matches!(parse_sse_line("data: [DONE]"), SseEvent::Done));
        assert!(matches!(parse_sse_line(""), SseEvent::None));
        let mut acc = BTreeMap::new();
        let mut pending = std::collections::VecDeque::new();
        for token in ["He", "llo"] {
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
        assert_eq!(texts, vec!["He", "llo"]);
    }

    /// Live smoke (requires `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`,
    /// `AZURE_OPENAI_DEPLOYMENT`). Run with `-- --ignored azure_live`.
    #[tokio::test]
    #[ignore = "requires Azure OpenAI credentials"]
    async fn azure_live_streams_a_reply() {
        let key = std::env::var("AZURE_OPENAI_API_KEY").expect("AZURE_OPENAI_API_KEY");
        let endpoint = std::env::var("AZURE_OPENAI_ENDPOINT").expect("AZURE_OPENAI_ENDPOINT");
        let deployment = std::env::var("AZURE_OPENAI_DEPLOYMENT").expect("AZURE_OPENAI_DEPLOYMENT");
        let mut llm = AzureLlm::new(key, endpoint, deployment);
        let ctx = LlmContext {
            messages: vec![json!({"role":"user","content":"Say 'hi' and nothing else."})],
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
