// SPDX-License-Identifier: Apache-2.0
//
//! Remote "brain" (HTTP webhook) adapter.
//!
//! [`RemoteBrain`] is a reference [`AgentBrain`] that delegates the conversation
//! policy to an out-of-process HTTP service. The embedder authors the policy in
//! Python (or any language) and exposes two JSON endpoints; Flowcat consults them
//! at turn granularity. This lets a policy author drive Flowcat's conversation
//! state machine over the network without writing Rust or using in-process
//! bindings.
//!
//! ## Wire contract
//!
//! The adapter is configured with a `base_url` (and an optional bearer token) and
//! calls two JSON endpoints:
//!
//! **Session start** — `POST {base_url}/session`
//! - request: `{ "brain_config": <opaque JSON>, "provider": <string> }`
//! - response: `{ "system_prompt": <string>, "tools": [ {name,description,params} ],
//!   "node_id": <string>, "collected_vars": <object> }`
//!
//! **Tool call** — `POST {base_url}/tool-call`
//! - request: `{ "node_id": <string>, "tool": { "name": <string>, "args": <object> },
//!   "collected_vars": <object> }`
//! - response: `{ "action": "transition"|"stay"|"end", "system_prompt"?, "tools"?,
//!   "say"?, "disposition"?, "node_id", "collected_vars", "finished" }`
//!   (`system_prompt`/`tools` are required iff `action == "transition"`.)
//!
//! When `auth_token` is set, every request carries `Authorization: Bearer <token>`.
//!
//! ## Runtime requirement
//!
//! [`RemoteBrain`] requires a **multi-threaded** tokio runtime (the server default,
//! i.e. `#[tokio::main]` or `#[tokio::main(flavor = "multi_thread")]`). The
//! `AgentBrain` trait is synchronous, but each turn needs an async HTTP round-trip;
//! the sync→async bridge uses [`tokio::task::block_in_place`] +
//! [`tokio::runtime::Handle::block_on`], which need a multi-threaded runtime —
//! [`RemoteBrain::connect`] returns an error on a current-thread runtime instead
//! of panicking later. The `on_tool_call` block runs at turn granularity (not per
//! audio frame) and the media path runs on other tasks, so the brief in-place
//! block does not affect per-frame audio latency.
//!
//! ## Testability
//!
//! The serialization is factored into pure functions ([`build_session_request`],
//! [`parse_session_response`], [`build_tool_call_request`],
//! [`parse_tool_call_response`]) so the wire format is unit-tested without a
//! socket (see the `tests` module).

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use flowcat_core::error::FlowcatError;
use flowcat_core::{AgentBrain, BrainAction, ToolDecl};

/// Per-request timeout for remote-brain calls. A turn-level policy decision (often
/// an LLM round-trip) is given up to this long; past it the request errors and
/// `on_tool_call` falls back to [`BrainAction::Stay`] rather than stalling the
/// call. Generous because it sits between turns, not on the audio path.
const DEFAULT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

// ---------------------------------------------------------------------------
// Wire DTOs — tidy snake_case bodies. We deliberately do NOT serialize the core
// `BrainAction` enum on the wire (its externally-tagged form is ugly); instead a
// snake_case-tagged `action` field selects the shape, and we map DTO→BrainAction.
// ---------------------------------------------------------------------------

/// Request body for `POST {base_url}/session`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct SessionRequest {
    /// The embedder's opaque policy/graph config (passed through verbatim).
    brain_config: Value,
    /// The realtime/LLM provider name driving the call.
    provider: String,
}

/// Response body for `POST {base_url}/session`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct SessionResponse {
    /// System prompt for the initial conversation state.
    system_prompt: String,
    /// Tools available in the initial state.
    tools: Vec<ToolDecl>,
    /// The brain's initial node id.
    node_id: String,
    /// Variables collected so far (object).
    collected_vars: Value,
}

/// Request body for `POST {base_url}/tool-call`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct ToolCallRequest {
    /// The brain's current node id.
    node_id: String,
    /// The tool/function the model invoked.
    tool: ToolInvocation,
    /// Variables collected so far (object).
    collected_vars: Value,
}

/// The `{ name, args }` of a model tool/function call.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct ToolInvocation {
    /// Function name.
    name: String,
    /// Function arguments (object).
    args: Value,
}

/// The action verb in a `/tool-call` response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ActionKind {
    /// Move to a new state (re-prompt + swap tools).
    Transition,
    /// Keep the current prompt/tools.
    Stay,
    /// End the call.
    End,
}

/// Response body for `POST {base_url}/tool-call`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct ToolCallResponse {
    /// Which kind of action the policy decided.
    action: ActionKind,
    /// New system prompt — required iff `action == "transition"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    system_prompt: Option<String>,
    /// New tool set — required iff `action == "transition"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolDecl>>,
    /// Optional line for the bot to say on entering the state (transition only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    say: Option<String>,
    /// Optional disposition/outcome label (end only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    disposition: Option<String>,
    /// The brain's new current node id.
    node_id: String,
    /// Updated collected variables (object).
    collected_vars: Value,
    /// Whether the conversation has reached a terminal state.
    finished: bool,
}

// ---------------------------------------------------------------------------
// Pure encode/decode seam — no network, the unit-test surface.
// ---------------------------------------------------------------------------

/// The seeded conversation state parsed from a `/session` response.
#[derive(Debug)]
struct SeededState {
    system_prompt: String,
    tools: Vec<ToolDecl>,
    node_id: String,
    collected_vars: Value,
}

/// The decoded outcome of a `/tool-call` response: the mapped action plus the
/// updated cached fields the adapter folds back in on success.
#[derive(Debug)]
struct DecodedToolCall {
    action: BrainAction,
    node_id: String,
    collected_vars: Value,
    finished: bool,
}

/// Build the `/session` request body (pure).
fn build_session_request(brain_config: &Value, provider: &str) -> Value {
    let req = SessionRequest {
        brain_config: brain_config.clone(),
        provider: provider.to_string(),
    };
    // `SessionRequest` is composed of always-serializable fields, so this never
    // fails; fall back to an empty object rather than panic.
    serde_json::to_value(&req).unwrap_or_else(|_| json!({}))
}

/// Parse a `/session` response into the seeded conversation state (pure).
fn parse_session_response(body: &[u8]) -> Result<SeededState, FlowcatError> {
    let resp: SessionResponse = serde_json::from_slice(body)
        .map_err(|e| FlowcatError::Protocol(format!("remote brain /session decode: {e}")))?;
    Ok(SeededState {
        system_prompt: resp.system_prompt,
        tools: resp.tools,
        node_id: resp.node_id,
        collected_vars: resp.collected_vars,
    })
}

/// Build the `/tool-call` request body (pure).
fn build_tool_call_request(
    node_id: &str,
    name: &str,
    args: &Value,
    collected_vars: &Value,
) -> Value {
    let req = ToolCallRequest {
        node_id: node_id.to_string(),
        tool: ToolInvocation {
            name: name.to_string(),
            args: args.clone(),
        },
        collected_vars: collected_vars.clone(),
    };
    serde_json::to_value(&req).unwrap_or_else(|_| json!({}))
}

/// Parse a `/tool-call` response into the mapped [`BrainAction`] plus the updated
/// cached fields (pure). A `transition` missing `system_prompt`/`tools` is a
/// protocol error.
fn parse_tool_call_response(body: &[u8]) -> Result<DecodedToolCall, FlowcatError> {
    let resp: ToolCallResponse = serde_json::from_slice(body)
        .map_err(|e| FlowcatError::Protocol(format!("remote brain /tool-call decode: {e}")))?;

    let action = match resp.action {
        ActionKind::Transition => {
            let system_prompt = resp.system_prompt.ok_or_else(|| {
                FlowcatError::Protocol(
                    "remote brain /tool-call: transition missing system_prompt".into(),
                )
            })?;
            let tools = resp.tools.ok_or_else(|| {
                FlowcatError::Protocol("remote brain /tool-call: transition missing tools".into())
            })?;
            BrainAction::Transition {
                system_prompt,
                tools,
                say: resp.say,
            }
        }
        ActionKind::Stay => BrainAction::Stay,
        ActionKind::End => BrainAction::End {
            disposition: resp.disposition,
        },
    };

    Ok(DecodedToolCall {
        action,
        node_id: resp.node_id,
        collected_vars: resp.collected_vars,
        finished: resp.finished,
    })
}

// ---------------------------------------------------------------------------
// RemoteBrain
// ---------------------------------------------------------------------------

/// A reference [`AgentBrain`] that drives the conversation policy from a remote
/// HTTP service.
///
/// It caches the current conversation state (system prompt, tools, node id,
/// collected variables, finished flag) and consults the remote service on each
/// tool call to decide the next [`BrainAction`].
///
/// # Runtime requirement
///
/// `RemoteBrain` **requires a multi-threaded tokio runtime** (the server default —
/// `#[tokio::main]` / `#[tokio::main(flavor = "multi_thread")]`). The synchronous
/// `on_tool_call` bridges to the async HTTP call via
/// [`tokio::task::block_in_place`], which needs a multi-threaded runtime;
/// [`RemoteBrain::connect`] rejects a current-thread runtime with an error rather
/// than panicking on the first turn.
///
/// # Fail-safe
///
/// On any HTTP/parse error during `on_tool_call`, the adapter logs a warning and
/// returns [`BrainAction::Stay`] **without** mutating cached state — a transient
/// brain error must never crash a live call. Requests also time out (see
/// `DEFAULT_TIMEOUT`), so a hung policy service falls back to `Stay` rather than
/// stalling the call.
#[derive(Debug)]
pub struct RemoteBrain {
    /// Shared HTTP client (connection-pooled across turns).
    http: reqwest::Client,
    /// `{base_url}` (trailing slashes trimmed).
    base_url: String,
    /// Optional bearer token sent on every request.
    auth_token: Option<String>,

    // ---- cached conversation state ----
    system_prompt: String,
    tools: Vec<ToolDecl>,
    node_id: String,
    collected_vars: Value,
    finished: bool,
}

impl RemoteBrain {
    /// Connect to the remote brain service: POST `/session`, parse the response,
    /// and seed the cached conversation state (`finished = false`).
    ///
    /// `provider` is the realtime/LLM provider name driving the call; `auth_token`,
    /// when set, is sent as `Authorization: Bearer <token>` on every request.
    pub async fn connect(
        base_url: impl Into<String>,
        brain_config: Value,
        provider: impl Into<String>,
        auth_token: Option<String>,
    ) -> Result<Self, FlowcatError> {
        // The sync→async bridge in `on_tool_call` uses `block_in_place`, which
        // requires a multi-threaded runtime. Reject a current-thread runtime here
        // with a clear setup error rather than panicking on the first turn.
        if matches!(
            tokio::runtime::Handle::current().runtime_flavor(),
            tokio::runtime::RuntimeFlavor::CurrentThread
        ) {
            return Err(FlowcatError::Other(
                "RemoteBrain requires a multi-threaded tokio runtime (use \
                 #[tokio::main] or #[tokio::main(flavor = \"multi_thread\")]); the \
                 current-thread runtime cannot drive the on_tool_call bridge"
                    .into(),
            ));
        }

        let base_url = base_url.into().trim_end_matches('/').to_string();
        let provider = provider.into();
        let http = reqwest::Client::builder()
            .timeout(DEFAULT_TIMEOUT)
            .build()
            .map_err(|e| FlowcatError::Network(format!("remote brain http client: {e}")))?;

        let body = build_session_request(&brain_config, &provider);
        let url = format!("{base_url}/session");
        let mut req = http.post(&url).json(&body);
        if let Some(token) = auth_token.as_deref() {
            req = req.bearer_auth(token);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| FlowcatError::Network(format!("remote brain /session send: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(FlowcatError::Network(format!(
                "remote brain /session {status}: {text}"
            )));
        }
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| FlowcatError::Network(format!("remote brain /session body: {e}")))?;
        let seeded = parse_session_response(&bytes)?;

        Ok(Self {
            http,
            base_url,
            auth_token,
            system_prompt: seeded.system_prompt,
            tools: seeded.tools,
            node_id: seeded.node_id,
            collected_vars: seeded.collected_vars,
            finished: false,
        })
    }

    /// The async `/tool-call` round-trip. Returns the decoded outcome on success.
    async fn post_tool_call(
        &self,
        name: &str,
        args: &Value,
    ) -> Result<DecodedToolCall, FlowcatError> {
        let body = build_tool_call_request(&self.node_id, name, args, &self.collected_vars);
        let url = format!("{}/tool-call", self.base_url);
        let mut req = self.http.post(&url).json(&body);
        if let Some(token) = self.auth_token.as_deref() {
            req = req.bearer_auth(token);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| FlowcatError::Network(format!("remote brain /tool-call send: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(FlowcatError::Network(format!(
                "remote brain /tool-call {status}: {text}"
            )));
        }
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| FlowcatError::Network(format!("remote brain /tool-call body: {e}")))?;
        parse_tool_call_response(&bytes)
    }
}

impl AgentBrain for RemoteBrain {
    fn system_prompt(&self) -> String {
        self.system_prompt.clone()
    }

    fn tools(&self) -> Vec<ToolDecl> {
        self.tools.clone()
    }

    fn current_node_id(&self) -> String {
        self.node_id.clone()
    }

    fn on_tool_call(&mut self, name: &str, args: &Value) -> BrainAction {
        // Bridge the sync trait method to the async HTTP call. `block_in_place`
        // moves the current task off the runtime worker so `block_on` can drive
        // the future without deadlocking the runtime — this REQUIRES a
        // multi-threaded runtime (see the type-level docs).
        let outcome = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.post_tool_call(name, args))
        });

        match outcome {
            Ok(decoded) => {
                // Update cached state from the response. For a transition the
                // response also carries the destination prompt + tools.
                if let BrainAction::Transition {
                    system_prompt,
                    tools,
                    ..
                } = &decoded.action
                {
                    self.system_prompt = system_prompt.clone();
                    self.tools = tools.clone();
                }
                self.node_id = decoded.node_id;
                self.collected_vars = decoded.collected_vars;
                self.finished = decoded.finished;
                decoded.action
            }
            // Fail-safe: a transient brain error must not crash a live call. Log
            // and Stay, leaving cached state untouched.
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    node_id = %self.node_id,
                    tool = name,
                    "remote brain /tool-call failed; staying in current state"
                );
                BrainAction::Stay
            }
        }
    }

    fn is_finished(&self) -> bool {
        self.finished
    }

    fn collected_vars(&self) -> Value {
        self.collected_vars.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- request body shapes (the documented wire contract) ----

    #[test]
    fn session_request_body_matches_documented_shape() {
        let body = build_session_request(&json!({"graph": "demo"}), "gemini");
        assert_eq!(
            body,
            json!({ "brain_config": { "graph": "demo" }, "provider": "gemini" })
        );
    }

    #[test]
    fn tool_call_request_body_matches_documented_shape() {
        let body = build_tool_call_request(
            "greeting",
            "collect_name",
            &json!({ "name": "Ada" }),
            &json!({ "intent": "support" }),
        );
        assert_eq!(
            body,
            json!({
                "node_id": "greeting",
                "tool": { "name": "collect_name", "args": { "name": "Ada" } },
                "collected_vars": { "intent": "support" }
            })
        );
    }

    // ---- session response → seeded state ----

    #[test]
    fn parse_session_response_seeds_state() {
        let body = br#"{
            "system_prompt": "You are a helpful agent.",
            "tools": [ { "name": "end_call", "description": "End the call.", "params": { "type": "object" } } ],
            "node_id": "start",
            "collected_vars": { "intent": "support" }
        }"#;
        let seeded = parse_session_response(body).expect("decode");
        assert_eq!(seeded.system_prompt, "You are a helpful agent.");
        assert_eq!(seeded.tools.len(), 1);
        assert_eq!(seeded.tools[0].name, "end_call");
        assert_eq!(seeded.node_id, "start");
        assert_eq!(seeded.collected_vars, json!({ "intent": "support" }));
    }

    // ---- tool-call response → (BrainAction, node_id, collected_vars, finished) ----

    #[test]
    fn transition_response_maps_to_transition() {
        let body = br#"{
            "action": "transition",
            "system_prompt": "Now collect the email.",
            "tools": [ { "name": "collect_email", "description": "Collect the email.", "params": { "type": "object" } } ],
            "say": "Great, what is your email?",
            "node_id": "collect_email",
            "collected_vars": { "name": "Ada" },
            "finished": false
        }"#;
        let decoded = parse_tool_call_response(body).expect("decode");
        match decoded.action {
            BrainAction::Transition {
                system_prompt,
                tools,
                say,
            } => {
                assert_eq!(system_prompt, "Now collect the email.");
                assert_eq!(tools.len(), 1);
                assert_eq!(tools[0].name, "collect_email");
                assert_eq!(say.as_deref(), Some("Great, what is your email?"));
            }
            other => panic!("expected Transition, got {other:?}"),
        }
        assert_eq!(decoded.node_id, "collect_email");
        assert_eq!(decoded.collected_vars, json!({ "name": "Ada" }));
        assert!(!decoded.finished);
    }

    #[test]
    fn stay_response_maps_to_stay() {
        let body = br#"{
            "action": "stay",
            "node_id": "greeting",
            "collected_vars": { "name": "Ada" },
            "finished": false
        }"#;
        let decoded = parse_tool_call_response(body).expect("decode");
        assert!(matches!(decoded.action, BrainAction::Stay));
        assert_eq!(decoded.node_id, "greeting");
        assert!(!decoded.finished);
    }

    #[test]
    fn end_response_maps_to_end_with_disposition() {
        let body = br#"{
            "action": "end",
            "disposition": "completed",
            "node_id": "wrapup",
            "collected_vars": { "name": "Ada" },
            "finished": true
        }"#;
        let decoded = parse_tool_call_response(body).expect("decode");
        match decoded.action {
            BrainAction::End { disposition } => {
                assert_eq!(disposition.as_deref(), Some("completed"));
            }
            other => panic!("expected End, got {other:?}"),
        }
        assert_eq!(decoded.node_id, "wrapup");
        assert!(decoded.finished);
    }

    #[test]
    fn transition_missing_prompt_is_protocol_error() {
        // `system_prompt`/`tools` are REQUIRED for a transition.
        let body = br#"{
            "action": "transition",
            "tools": [],
            "node_id": "x",
            "collected_vars": {},
            "finished": false
        }"#;
        let err = parse_tool_call_response(body).expect_err("must reject");
        assert!(matches!(err, FlowcatError::Protocol(_)));
    }

    #[test]
    fn connect_rejects_current_thread_runtime() {
        // The sync→async bridge needs a multi-threaded runtime; connect() must fail
        // fast (not panic later) on a current-thread runtime. The flavor check runs
        // before any network, so the unreachable URL is never contacted.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let err = rt
            .block_on(RemoteBrain::connect(
                "http://127.0.0.1:0",
                json!({}),
                "test",
                None,
            ))
            .expect_err("current-thread runtime must be rejected");
        assert!(matches!(err, FlowcatError::Other(_)));
    }
}
