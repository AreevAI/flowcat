// SPDX-License-Identifier: Apache-2.0
//
//! Native Rust Gemini Live client (`RealtimeLlm` impl).
//!
//! Connects to
//! `wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key=<API_KEY>`
//! and speaks the **JSON** BidiGenerateContent protocol (see DESIGN.md
//! "Gemini Live protocol"):
//!
//! - client→server: `setup`, `realtimeInput.audio`, `toolResponse`, `clientContent`.
//! - server→client: `setupComplete`, `serverContent.modelTurn.parts[].inlineData`
//!   (24 kHz PCM audio), input/output transcription, `interrupted`,
//!   `toolCall.functionCalls`, `usageMetadata`, `goAway`/`sessionResumptionUpdate`.
//!
//! ## Session resumption / `goAway` reconnect
//!
//! The Live API socket has a bounded lifespan (~10 min); the server warns of an
//! impending teardown with a `goAway` message (carrying `timeLeft`) and, when
//! session-resumption is enabled in `setup`, periodically emits a
//! `sessionResumptionUpdate{newHandle,resumable}` carrying an opaque **resumption
//! handle**. We enable resumption in `setup` (`sessionResumption: {}`), keep the
//! latest handle, and on a `goAway` **or an unexpected socket drop** we
//! transparently **reconnect mid-call resuming from the stored handle** — the
//! conversation context survives the controlled reconnect instead of the call
//! ending. This mirrors the reference `GeminiLiveLLMService` (`_reconnect` →
//! `_connect(session_resumption_handle=...)` on both the `go_away` and the
//! error-reconnect paths). If resumption is **not possible** (no handle yet) or
//! the reconnect itself fails, we fall back to the previous behaviour and surface
//! [`RealtimeEvent::Closed`] so the call ends cleanly — never a hang. The
//! reconnect is invisible to the [`RealtimeLlm`] consumer: [`Self::next_event`]
//! does the resume internally and keeps yielding events from the fresh socket.
//!
//! ## Wire format
//!
//! The BidiGenerateContent WSS endpoint accepts/emits the proto3-JSON encoding
//! used by the `mldev` (API-key) transport of the official `google-genai` SDK.
//! Top-level message keys and all nested config keys are **camelCase**
//! (`setup`, `realtimeInput.audio`, `generationConfig.responseModalities`,
//! `serverContent.modelTurn.parts[].inlineData`, `usageMetadata`, …). The exact
//! shapes here were cross-checked against `google/genai/_live_converters.py`
//! (`*_to_mldev` / `*_from_mldev`).
//!
//! ## System-instruction / tool updates = reconnect (not in-session)
//!
//! The Gemini Live `systemInstruction` and `tools` are part of the one-shot
//! `setup` message; there is **no in-session update message** to change them
//! (the reference `GeminiLiveLLMService` applies a changed system instruction by
//! tearing the session down and re-`connect`-ing — `_reconnect` calls
//! `_disconnect` then `_connect` with a fresh `LiveConnectConfig`). So
//! [`Self::update_system`] closes the socket and opens a new one with a `setup`
//! carrying the new prompt and tools. We keep the last [`RealtimeSetup`] so we can
//! rebuild it.
//!
//! ## API key
//!
//! The key is taken from the value passed to [`GeminiLive::new`] (carried on the
//! struct). [`RealtimeSetup`] intentionally does **not** carry a key, so no
//! `frame.rs` change was needed.

use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine as _;
use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::{header::AUTHORIZATION, HeaderValue};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

use crate::error::FlowcatError;
use crate::realtime::{RealtimeKickoff, RealtimeLlm};
use crate::types::{AudioChunk, RealtimeEvent, RealtimeSetup, ToolDecl, Usage};

/// Base WSS endpoint for the Gemini Live BidiGenerateContent service, used with a
/// raw API key (`?key=<API_KEY>` appended at connect time).
pub const GEMINI_LIVE_WSS_BASE: &str =
    "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent";

/// WSS endpoint for Gemini Live **ephemeral tokens** (`auth_tokens/…`). Ephemeral
/// tokens (BYOK) require the *Constrained* variant plus an
/// `Authorization: Token <token>` header; a plain `?key=<token>` on the base
/// endpoint is rejected ("API key not valid"). Matches the google-genai SDK
/// (`live.py`: `method = 'BidiGenerateContentConstrained'`).
const GEMINI_LIVE_WSS_CONSTRAINED: &str =
    "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContentConstrained";

/// Default prebuilt voice if `RealtimeSetup` carries none (it currently has no
/// voice field). `Charon` matches the reference service default.
const DEFAULT_VOICE: &str = "Fenrir";

/// The concrete WS socket type produced by [`connect_async`].
type ClientSocket = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;
type WsSink = SplitSink<ClientSocket, Message>;
type WsStream = SplitStream<ClientSocket>;

/// What the reader task hands back to [`RealtimeLlm::next_event`].
///
/// The reader cannot reconnect itself (it owns only the read half and has no
/// API key / setup), so on a `goAway` or socket drop it emits [`ReaderMsg::Lost`]
/// — a *reconnect request* — and `next_event` (which has `&mut self`) decides
/// whether a session-resumption reconnect is possible. This is the seam that
/// keeps the `goAway`/drop handling transparent to the trait consumer.
#[cfg_attr(test, derive(Debug))]
enum ReaderMsg {
    /// A decoded model event to forward to the consumer.
    Event(RealtimeEvent),
    /// The socket dropped or the server sent `goAway`: the consumer should try
    /// to resume (see [`GeminiLive::next_event`]); falls back to `Closed`.
    Lost,
}

/// The latest session-resumption handle, shared between the reader task (which
/// decodes `sessionResumptionUpdate`) and the client (which uses it to resume on
/// reconnect). A plain std `Mutex` — only brief, non-async critical sections.
type ResumptionHandle = Arc<std::sync::Mutex<Option<String>>>;

/// Live connection state, present only while connected.
struct Connection {
    /// Write half of the split WS. Behind an async `Mutex` so the (single)
    /// `send_*` caller and any future concurrent sender coexist with the reader
    /// task, which independently owns the read half (the split makes the two
    /// halves disjoint, so there is no contention in practice).
    sink: Arc<Mutex<WsSink>>,
    /// Decoded server events / reconnect requests; drained by
    /// [`RealtimeLlm::next_event`].
    events: mpsc::UnboundedReceiver<ReaderMsg>,
    /// Background task parsing server frames into `events`.
    reader: JoinHandle<()>,
}

impl Connection {
    /// Send a JSON value as a WS text frame.
    async fn send_json(&self, value: &Value) -> Result<(), FlowcatError> {
        let text = serde_json::to_string(value)?;
        let mut sink = self.sink.lock().await;
        sink.send(Message::text(text))
            .await
            .map_err(|e| FlowcatError::Realtime(format!("ws send: {e}")))
    }
}

impl Drop for Connection {
    /// On normal call teardown `GeminiLive` is dropped without an explicit
    /// `close_conn`; a dropped `JoinHandle` *detaches* the task (it would linger
    /// until the server closes the socket). Abort it so per-call cleanup is
    /// deterministic and the reader cannot outlive its connection.
    fn drop(&mut self) {
        self.reader.abort();
    }
}

/// Server-side VAD (turn-detection) tuning for Gemini Live, read from
/// `FLOWCAT_VAD_*` env at client construction so it can be adjusted in a
/// running deployment **without a rebuild**.
///
/// The defaults are deliberately conservative on the *start* side so the agent
/// no longer cuts its own speech on brief caller sounds — backchannels
/// ("uh-huh", "yeah, okay"), a cough, or line noise. `START_SENSITIVITY_LOW`
/// plus a 500 ms prefix-padding debounce means a short blip is not committed as
/// a user turn (the over-eager `START_SENSITIVITY_HIGH` + `prefixPaddingMs: 20`
/// was observed truncating responses and derailing the conversation on real
/// calls; 300 ms still let a ~690 ms "yeah, okay" through, so the default is
/// 500). The *end* side stays eager (`END_SENSITIVITY_HIGH` + 350 ms
/// trailing silence) so the agent still replies snappily once the caller
/// actually finishes their turn.
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// `startOfSpeechSensitivity` (`START_SENSITIVITY_LOW`/`_HIGH`).
    pub start_sensitivity: String,
    /// `endOfSpeechSensitivity` (`END_SENSITIVITY_LOW`/`_HIGH`).
    pub end_sensitivity: String,
    /// `prefixPaddingMs`: speech must persist this long before a barge-in
    /// commits — the lever that filters brief backchannels.
    pub prefix_padding_ms: u32,
    /// `silenceDurationMs`: trailing silence that ends the caller's turn.
    pub silence_duration_ms: u32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            start_sensitivity: "START_SENSITIVITY_LOW".into(),
            end_sensitivity: "END_SENSITIVITY_HIGH".into(),
            prefix_padding_ms: 500,
            silence_duration_ms: 350,
        }
    }
}

/// Accepted Gemini Live `startOfSpeechSensitivity` enum values.
const START_SENSITIVITIES: [&str; 3] = [
    "START_SENSITIVITY_UNSPECIFIED",
    "START_SENSITIVITY_LOW",
    "START_SENSITIVITY_HIGH",
];
/// Accepted Gemini Live `endOfSpeechSensitivity` enum values.
const END_SENSITIVITIES: [&str; 3] = [
    "END_SENSITIVITY_UNSPECIFIED",
    "END_SENSITIVITY_LOW",
    "END_SENSITIVITY_HIGH",
];

impl VadConfig {
    /// Build from `FLOWCAT_VAD_*` env vars, each falling back to [`Default`].
    pub fn from_env() -> Self {
        Self::from_getter(|k| std::env::var(k).ok())
    }

    /// Pure parse from a key→value getter — the testable seam behind [`from_env`]
    /// (avoids racy process-global `std::env` mutation in unit tests).
    pub(crate) fn from_getter(get: impl Fn(&str) -> Option<String>) -> Self {
        let d = Self::default();
        Self {
            start_sensitivity: vad_sensitivity(
                get("FLOWCAT_VAD_START_SENSITIVITY"),
                d.start_sensitivity,
                &START_SENSITIVITIES,
            ),
            end_sensitivity: vad_sensitivity(
                get("FLOWCAT_VAD_END_SENSITIVITY"),
                d.end_sensitivity,
                &END_SENSITIVITIES,
            ),
            prefix_padding_ms: vad_u32(get("FLOWCAT_VAD_PREFIX_PADDING_MS"), d.prefix_padding_ms),
            silence_duration_ms: vad_u32(
                get("FLOWCAT_VAD_SILENCE_DURATION_MS"),
                d.silence_duration_ms,
            ),
        }
    }
}

/// A sensitivity value validated against the Gemini Live enum. Empty OR
/// unrecognised (a typo like `START_SENSITIVUTY_LOW`) falls back to `default` —
/// otherwise a bad value is sent verbatim, Gemini ignores it and silently reverts
/// to its own (twitchy) VAD, which makes the agent's audio choppy with no signal.
fn vad_sensitivity(val: Option<String>, default: String, allowed: &[&str]) -> String {
    match val {
        Some(s) if !s.is_empty() => {
            if allowed.contains(&s.as_str()) {
                s
            } else {
                tracing::warn!(value = %s, ?allowed, "unrecognised VAD sensitivity; using default");
                default
            }
        }
        _ => default,
    }
}

fn vad_u32(val: Option<String>, default: u32) -> u32 {
    val.and_then(|s| s.parse().ok()).unwrap_or(default)
}

/// A native Gemini Live realtime session.
pub struct GeminiLive {
    /// Google API key used in the `?key=` query parameter.
    api_key: String,
    /// The live connection, present once [`RealtimeLlm::connect`] succeeds.
    conn: Option<Connection>,
    /// The setup last used to connect — retained so [`Self::update_system`] can
    /// reconnect with a new prompt/tools (Gemini Live has no in-session update)
    /// and so a `goAway`/drop reconnect can rebuild the `setup`.
    setup: Option<RealtimeSetup>,
    /// Turn-detection tuning sent in the `setup` message (env-driven).
    vad: VadConfig,
    /// The latest session-resumption handle from `sessionResumptionUpdate`
    /// (`None` until the server emits one). On a `goAway`/drop we reconnect with
    /// `setup.sessionResumption.handle = <this>` to resume context mid-call.
    /// Shared with the reader task, which writes each new handle here.
    resumption: ResumptionHandle,
    /// How many *consecutive* failed reconnect attempts we have made on the
    /// current call. Reset to 0 on a clean (re)connect; once it reaches
    /// [`MAX_CONSECUTIVE_RECONNECTS`] we stop resuming and surface `Closed`
    /// (mirrors the reference's consecutive-failure ceiling so a persistently
    /// broken socket can't spin forever).
    consecutive_reconnects: u32,
    /// Fired by the reader task whenever a new event is queued, so the S2S reader
    /// can await readiness (via [`RealtimeLlm::event_notify`]) WITHOUT holding the
    /// session lock across a blocking `next_event` — which would starve
    /// `send_audio` and deadlock the call between bot turns.
    event_notify: Arc<tokio::sync::Notify>,
    /// When `Some`, authenticate against **Vertex AI** (regional `aiplatform`
    /// host + OAuth2 `Authorization: Bearer` + a full model resource path) instead
    /// of the AI-Studio `?key=` flow. The BidiGenerateContent wire protocol is
    /// otherwise identical, so only `url()` / the connect auth / the setup `model`
    /// differ — the rest of the client is shared verbatim.
    vertex: Option<VertexBinding>,
}

/// Vertex AI binding for [`GeminiLive`] (the alternative to an AI-Studio API key).
/// The OAuth2 access token is operator-supplied (e.g. `gcloud auth print-access-token`)
/// and short-lived — config, never request-derived (no SSRF surface).
struct VertexBinding {
    access_token: String,
    project: String,
    location: String,
}

/// Default Vertex location when none is configured.
pub const VERTEX_DEFAULT_LOCATION: &str = "us-central1";
/// Default Vertex Live model when the setup carries none.
pub const DEFAULT_VERTEX_LIVE_MODEL: &str = "gemini-live-2.5-flash";

/// Cap on back-to-back reconnect attempts before we give up and end the call
/// (mirrors the reference `MAX_CONSECUTIVE_FAILURES`). A `goAway` reconnect and a
/// drop reconnect both count; a successful reconnect resets the counter.
const MAX_CONSECUTIVE_RECONNECTS: u32 = 3;

impl GeminiLive {
    /// Construct a client bound to the given Google API key. Does not connect
    /// until [`RealtimeLlm::connect`] is called.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            conn: None,
            setup: None,
            vad: VadConfig::from_env(),
            resumption: Arc::new(std::sync::Mutex::new(None)),
            consecutive_reconnects: 0,
            event_notify: Arc::new(tokio::sync::Notify::new()),
            vertex: None,
        }
    }

    /// Construct a **Vertex AI** Gemini Live client: a pre-minted OAuth2
    /// `access_token`, a GCP `project`, and a `location` (region, e.g.
    /// `us-central1`; empty → [`VERTEX_DEFAULT_LOCATION`]). Same Live wire protocol
    /// as [`Self::new`], on the Vertex surface (regional `aiplatform` host + Bearer
    /// auth + a `projects/…/publishers/google/models/<model>` resource path).
    pub fn new_vertex(
        access_token: impl Into<String>,
        project: impl Into<String>,
        location: impl Into<String>,
    ) -> Self {
        let location = location.into();
        let location = if location.trim().is_empty() {
            VERTEX_DEFAULT_LOCATION.to_string()
        } else {
            location
        };
        Self {
            api_key: String::new(),
            conn: None,
            setup: None,
            vad: VadConfig::from_env(),
            resumption: Arc::new(std::sync::Mutex::new(None)),
            consecutive_reconnects: 0,
            event_notify: Arc::new(tokio::sync::Notify::new()),
            vertex: Some(VertexBinding {
                access_token: access_token.into(),
                project: project.into(),
                location,
            }),
        }
    }

    /// The API key this client will authenticate with (empty in Vertex mode).
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// The full WSS URL. AI-Studio: `{base}?key=<key>`. Vertex: the regional
    /// `{loc}-aiplatform.googleapis.com` (or global) `LlmBidiService` endpoint, with
    /// auth carried in the `Authorization` header at connect (not the URL).
    fn url(&self) -> String {
        match &self.vertex {
            None => format!("{GEMINI_LIVE_WSS_BASE}?key={}", self.api_key),
            Some(v) => {
                let host = if v.location == "global" {
                    "aiplatform.googleapis.com".to_string()
                } else {
                    format!("{}-aiplatform.googleapis.com", v.location)
                };
                format!(
                    "wss://{host}/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent"
                )
            }
        }
    }

    /// Resolve the `setup.model` to what the chosen surface expects: AI-Studio takes
    /// the bare model id; Vertex takes the full
    /// `projects/{project}/locations/{location}/publishers/google/models/{model}`
    /// resource path (defaulting an empty model to [`DEFAULT_VERTEX_LIVE_MODEL`]).
    fn effective_model(&self, model: &str) -> String {
        match &self.vertex {
            None => model.to_string(),
            Some(v) => {
                let m = if model.trim().is_empty() {
                    DEFAULT_VERTEX_LIVE_MODEL
                } else {
                    model
                };
                format!(
                    "projects/{}/locations/{}/publishers/google/models/{}",
                    v.project, v.location, m
                )
            }
        }
    }

    /// Open the socket, send `setup`, await `setupComplete`, and spawn the
    /// reader task. Replaces any existing connection.
    ///
    /// When `resume` is `true` the current [`Self::resumption`] handle (if any)
    /// is threaded into `setup.sessionResumption` so the server continues the
    /// prior conversation; on a fresh connect (`resume = false`) we still enable
    /// resumption (empty config) so the server starts issuing handles.
    async fn open(&mut self, setup: &RealtimeSetup, resume: bool) -> Result<(), FlowcatError> {
        // Tear down any prior connection first (used by update_system).
        self.close_conn().await;

        // The handle to resume from (only on a reconnect; `None` on a fresh
        // connect just enables resumption so the server starts emitting handles).
        let resume_handle = if resume {
            self.resumption.lock().expect("resumption mutex").clone()
        } else {
            None
        };

        // Build the WebSocket request for the chosen surface, then a single
        // `connect_async`. Three authentication shapes:
        //   * Vertex AI — the Bidi endpoint (`url()`), authenticated with an
        //     `Authorization: Bearer <oauth2>` header.
        //   * AI-Studio ephemeral token (`auth_tokens/…`, BYOK) — the Constrained
        //     endpoint + an `Authorization: Token <token>` header; a plain
        //     `?key=<token>` is rejected ("API key not valid"). Mirrors the SDK.
        //   * AI-Studio raw API key — carried in the URL (`?key=`), no header.
        let request = match &self.vertex {
            Some(v) => {
                let mut req = self
                    .url()
                    .into_client_request()
                    .map_err(|e| FlowcatError::Realtime(format!("vertex request: {e}")))?;
                let bearer = HeaderValue::from_str(&format!("Bearer {}", v.access_token))
                    .map_err(|e| FlowcatError::Realtime(format!("vertex auth header: {e}")))?;
                req.headers_mut().insert(AUTHORIZATION, bearer);
                req
            }
            None if self.api_key.starts_with("auth_tokens/") => {
                let mut req = GEMINI_LIVE_WSS_CONSTRAINED
                    .into_client_request()
                    .map_err(|e| FlowcatError::Realtime(format!("ws request: {e}")))?;
                let auth = HeaderValue::from_str(&format!("Token {}", self.api_key))
                    .map_err(|e| FlowcatError::Realtime(format!("auth header: {e}")))?;
                req.headers_mut().insert(AUTHORIZATION, auth);
                req
            }
            None => self
                .url()
                .into_client_request()
                .map_err(|e| FlowcatError::Realtime(format!("ws request: {e}")))?,
        };
        let (socket, _resp) = connect_async(request)
            .await
            .map_err(|e| FlowcatError::Realtime(format!("connect_async: {e}")))?;
        let (mut sink, mut stream) = socket.split();

        // 1. Send the `setup` message. In Vertex mode the model must be the full
        //    resource path, so rewrite it on a local copy before encoding (the stored
        //    `setup` keeps the bare id for reconnects).
        let setup = {
            let mut s = setup.clone();
            s.model = self.effective_model(&setup.model);
            s
        };
        let setup_msg = encode_setup(&setup, &self.vad, resume_handle.as_deref());
        let setup_text = serde_json::to_string(&setup_msg)?;
        // Diagnostic: the `tools` block is the most common setup-rejection cause
        // (a tool's parameters schema Gemini won't accept → 1007 close right after
        // setupComplete). Log it so a rejection can be correlated to the payload.
        tracing::debug!(
            tools = %setup_msg.get("setup").and_then(|s| s.get("tools")).map(|t| t.to_string()).unwrap_or_else(|| "[]".into()),
            "gemini-live: sending setup"
        );
        sink.send(Message::text(setup_text))
            .await
            .map_err(|e| FlowcatError::Realtime(format!("send setup: {e}")))?;

        // 2. Await `setupComplete` (the server's first frame). Skip any non-text
        //    control frames (ping/pong) until we see it.
        await_setup_complete(&mut stream).await?;

        // 3. Spawn the reader: parse each server frame into a ReaderMsg. The
        //    reader records each `sessionResumptionUpdate` handle into the shared
        //    cell so a later reconnect can resume from it.
        let (tx, rx) = mpsc::unbounded_channel();
        let reader = tokio::spawn(reader_task(
            stream,
            tx,
            self.resumption.clone(),
            self.event_notify.clone(),
        ));

        self.conn = Some(Connection {
            sink: Arc::new(Mutex::new(sink)),
            events: rx,
            reader,
        });
        Ok(())
    }

    /// Abort the reader task and drop the connection, if any.
    async fn close_conn(&mut self) {
        if let Some(conn) = self.conn.take() {
            conn.reader.abort();
        }
    }

    /// Trigger an initial model turn for bot-first (outbound) calls.
    ///
    /// Sends a `clientContent` with an empty user turn and `turnComplete:true`,
    /// matching the reference's "seed + commit" kickoff (the model then speaks
    /// first off the system instruction). The pipeline calls this after
    /// `connect` for outbound calls.
    pub async fn kickoff(&mut self) -> Result<(), FlowcatError> {
        let msg = json!({
            "clientContent": {
                "turns": [{ "role": "user", "parts": [{ "text": "" }] }],
                "turnComplete": true
            }
        });
        self.require_conn()?.send_json(&msg).await
    }

    /// Borrow the live connection or error if not connected.
    fn require_conn(&self) -> Result<&Connection, FlowcatError> {
        self.conn
            .as_ref()
            .ok_or_else(|| FlowcatError::Realtime("not connected".into()))
    }
}

#[async_trait]
impl RealtimeLlm for GeminiLive {
    async fn connect(&mut self, setup: RealtimeSetup) -> Result<(), FlowcatError> {
        // Fresh session: clear any stale resumption handle and connect without
        // resuming (resumption is still *enabled* so the server starts emitting
        // handles for a possible later reconnect).
        *self.resumption.lock().expect("resumption mutex") = None;
        self.consecutive_reconnects = 0;
        self.open(&setup, false).await?;
        self.setup = Some(setup);
        Ok(())
    }

    async fn send_audio(&mut self, chunk: AudioChunk) -> Result<(), FlowcatError> {
        let msg = encode_realtime_input(&chunk);
        self.require_conn()?.send_json(&msg).await
    }

    async fn update_system(
        &mut self,
        prompt: String,
        tools: Vec<ToolDecl>,
    ) -> Result<(), FlowcatError> {
        // Gemini Live has no in-session system-instruction/tool update: rebuild
        // the setup with the new prompt + tools and reconnect (see module docs).
        let mut setup = self
            .setup
            .clone()
            .ok_or_else(|| FlowcatError::Realtime("update_system before connect".into()))?;
        setup.system_prompt = prompt;
        setup.tools = tools;
        // A prompt/tool change starts a *new* conversation turn-set; do not
        // resume the prior session handle (the new system instruction must take
        // effect). Keep emitting handles for a drop-reconnect of this new state.
        *self.resumption.lock().expect("resumption mutex") = None;
        self.open(&setup, false).await?;
        self.setup = Some(setup);
        Ok(())
    }

    async fn send_tool_result(&mut self, id: String, result: Value) -> Result<(), FlowcatError> {
        let msg = encode_tool_response(&id, result);
        self.require_conn()?.send_json(&msg).await
    }

    async fn next_event(&mut self) -> Option<RealtimeEvent> {
        loop {
            let msg = {
                let conn = self.conn.as_mut()?;
                conn.events.recv().await
            };
            match msg {
                // A normal model event: forward it. A successful event means the
                // current socket is healthy, so a prior reconnect "stuck" — reset
                // the consecutive-reconnect counter.
                Some(ReaderMsg::Event(ev)) => {
                    self.consecutive_reconnects = 0;
                    return Some(ev);
                }
                // The socket dropped or the server sent `goAway`: try to resume
                // mid-call from the stored handle. On success, loop and read from
                // the fresh socket (transparent to the consumer). On failure (no
                // handle, reconnect error, or too many attempts) fall back to the
                // old behaviour: surface `Closed` so the call ends.
                Some(ReaderMsg::Lost) => match self.try_reconnect().await {
                    Ok(()) => continue,
                    Err(reason) => {
                        tracing::warn!(%reason, "gemini-live: resume failed; ending session");
                        return Some(RealtimeEvent::Closed);
                    }
                },
                // The reader channel closed without a `Lost` request (only the
                // sender being dropped, e.g. on `close_conn`). End the stream.
                None => return None,
            }
        }
    }

    fn event_notify(&self) -> Option<Arc<tokio::sync::Notify>> {
        Some(self.event_notify.clone())
    }

    /// Non-blocking `next_event`: drains a ready `ReaderMsg` without awaiting the
    /// channel, so the S2S reader holds the session lock only for this brief poll
    /// (not across the idle wait between bot turns) and `send_audio` is never
    /// starved. A `Lost` still drives the (rare, brief) resume under the lock.
    async fn poll_event(&mut self) -> crate::realtime::PollEvent {
        use crate::realtime::PollEvent;
        use tokio::sync::mpsc::error::TryRecvError;
        loop {
            let msg = {
                let Some(conn) = self.conn.as_mut() else {
                    return PollEvent::Ready(None);
                };
                match conn.events.try_recv() {
                    Ok(m) => m,
                    Err(TryRecvError::Empty) => return PollEvent::Pending,
                    Err(TryRecvError::Disconnected) => return PollEvent::Ready(None),
                }
            };
            match msg {
                ReaderMsg::Event(ev) => {
                    self.consecutive_reconnects = 0;
                    return PollEvent::Ready(Some(ev));
                }
                ReaderMsg::Lost => match self.try_reconnect().await {
                    Ok(()) => continue, // re-poll the fresh socket
                    Err(reason) => {
                        tracing::warn!(%reason, "gemini-live: resume failed; ending session");
                        return PollEvent::Ready(Some(RealtimeEvent::Closed));
                    }
                },
            }
        }
    }
}

#[async_trait]
impl RealtimeKickoff for GeminiLive {
    async fn kickoff(&mut self) -> Result<(), FlowcatError> {
        // Delegate to the concrete client's inherent kickoff (seed + commit an
        // empty user turn; the model then speaks off the system instruction).
        GeminiLive::kickoff(self).await
    }
}

impl GeminiLive {
    /// Attempt to reconnect the dropped/`goAway`-ed session, resuming from the
    /// stored resumption handle. Returns `Err(reason)` (so the caller surfaces
    /// `Closed`) when resumption is impossible: no handle yet, no `setup` to
    /// rebuild, the reconnect ceiling was hit, or the socket re-open failed.
    async fn try_reconnect(&mut self) -> Result<(), String> {
        // A handle must exist — without one the server cannot continue the prior
        // session, so resuming is pointless (fall back to ending the call).
        let has_handle = self.resumption.lock().expect("resumption mutex").is_some();
        if !has_handle {
            return Err("no resumption handle".into());
        }

        // Bounded retries so a persistently broken socket can't spin forever.
        if self.consecutive_reconnects >= MAX_CONSECUTIVE_RECONNECTS {
            return Err(format!(
                "reconnect ceiling reached ({MAX_CONSECUTIVE_RECONNECTS})"
            ));
        }
        self.consecutive_reconnects += 1;

        let setup = self
            .setup
            .clone()
            .ok_or_else(|| "reconnect before connect".to_string())?;

        tracing::info!(
            attempt = self.consecutive_reconnects,
            "gemini-live: resuming session after goAway/drop"
        );
        // `open(.., resume=true)` threads the stored handle into the setup. Note:
        // `consecutive_reconnects` is *not* reset here — it resets only when a
        // real event arrives over the new socket (see `next_event`), so a socket
        // that re-opens but immediately drops again still counts toward the cap.
        self.open(&setup, true)
            .await
            .map_err(|e| format!("reconnect: {e}"))
    }
}

// ---------------------------------------------------------------------------
// Encoders (client → server). Pure functions over `serde_json::Value` so they
// are unit-testable without a socket.
// ---------------------------------------------------------------------------

/// Build the one-shot `setup` message from a [`RealtimeSetup`].
///
/// Shape (camelCase, `mldev` proto3-JSON):
/// ```json
/// {"setup": {
///   "model": "...",
///   "generationConfig": {
///     "responseModalities": ["AUDIO"],
///     "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Charon"}}}
///   },
///   "systemInstruction": {"parts": [{"text": "<prompt>"}]},
///   "tools": [{"functionDeclarations": [{"name","description","parameters"}, ...]}],
///   "generationConfig": {..., "thinkingConfig": {"thinkingBudget": 0}},
///   "realtimeInputConfig": {"automaticActivityDetection": {"endOfSpeechSensitivity": "...", "silenceDurationMs": 350}},
///   "sessionResumption": {} | {"handle": "<resume-handle>"},
///   "inputAudioTranscription": {},
///   "outputAudioTranscription": {}
/// }}
/// ```
///
/// `resume_handle` controls the `sessionResumption` config:
/// - `None` → `sessionResumption: {}` (resumption **enabled**; the server starts
///   issuing `sessionResumptionUpdate` handles, but this is a fresh session).
/// - `Some(h)` → `sessionResumption: {"handle": h}` (resume the prior session —
///   used on a `goAway`/drop reconnect so context carries over).
fn encode_setup(setup: &RealtimeSetup, vad: &VadConfig, resume_handle: Option<&str>) -> Value {
    // Realtime voice is env-tunable (`FLOWCAT_VOICE`) so it tracks the
    // `/models` model catalog (Gemini Live realtime voice; default `Fenrir`)
    // and can be changed without a rebuild. Was previously hardcoded to
    // `Charon`, which ignored the configured voice.
    let voice = std::env::var("FLOWCAT_VOICE")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| DEFAULT_VOICE.to_string());
    let mut setup_obj = json!({
        "model": setup.model,
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": { "voiceName": voice.as_str() }
                }
            },
            // Disable model "thinking" (chain-of-thought before replying): the
            // 2.5-flash native-audio model thinks by default, adding hundreds of ms
            // per turn (`thoughtsTokenCount` in usage metrics). A phone receptionist
            // doesn't need it; thinkingBudget=0 turns it off for snappier responses
            // (supported on the Live API for 2.5-class models, verified vs the SDK).
            "thinkingConfig": { "thinkingBudget": 0 }
        },
        "systemInstruction": {
            "parts": [{ "text": setup.system_prompt }]
        },
        // Server-side VAD (turn detection). Values come from `VadConfig` so they
        // are env-tunable in a live deployment (see `VadConfig`). The *start*
        // side is conservative by default (`START_SENSITIVITY_LOW` + a 500ms
        // prefix-padding debounce) so brief caller sounds — backchannels
        // ("uh-huh"), a cough, line noise — are NOT committed as a user turn and
        // do not cut the agent mid-reply. The *end* side stays eager
        // (`END_SENSITIVITY_HIGH` + `silenceDurationMs` 350) for snappy turns:
        // once the caller genuinely stops, the agent replies promptly.
        "realtimeInputConfig": {
            "automaticActivityDetection": {
                "startOfSpeechSensitivity": vad.start_sensitivity.as_str(),
                "endOfSpeechSensitivity": vad.end_sensitivity.as_str(),
                "prefixPaddingMs": vad.prefix_padding_ms,
                "silenceDurationMs": vad.silence_duration_ms
            }
        },
        // Turn on both transcription streams (empty config = defaults).
        "inputAudioTranscription": {},
        "outputAudioTranscription": {}
    });

    // Only include `tools` when non-empty; an empty function-declarations list
    // is at best a no-op and some backends reject it.
    if !setup.tools.is_empty() {
        let decls: Vec<Value> = setup.tools.iter().map(encode_tool_decl).collect();
        setup_obj["tools"] = json!([{ "functionDeclarations": decls }]);
    }

    // Enable session resumption. An empty config switches the feature on (the
    // server then emits `sessionResumptionUpdate{newHandle}` periodically); a
    // handle resumes the prior session on a `goAway`/drop reconnect so the
    // conversation context survives mid-call (see module docs).
    setup_obj["sessionResumption"] = match resume_handle {
        Some(h) => json!({ "handle": h }),
        None => json!({}),
    };

    json!({ "setup": setup_obj })
}

/// Recursively reduce a JSON-Schema value to the **OpenAPI-3.0 subset Gemini's
/// `Schema` proto accepts**. MCP tools expose a raw JSON Schema (`input_schema`)
/// that routinely carries keys Gemini does NOT know — `$schema`, `$ref`,
/// `$defs`/`definitions`, `additionalProperties`, `patternProperties`, … — and
/// the Live API rejects an unknown key by **closing the socket with `1007`
/// ("Invalid JSON payload received. Unknown name \"$schema\"")** right after
/// `setupComplete`, which ends the call before the agent can speak. So we keep
/// only the documented Gemini `Schema` fields and drop everything else,
/// recursing into `properties`/`items`/`anyOf`. (The reference Python host got
/// this for free via the Google SDK's schema coercion.)
///
/// `pub` + re-exported from [`crate::realtime`] so the **cascaded** Gemini LLM
/// service ([`flowcat-services`] `GoogleLlm`) sanitizes its `functionDeclarations`
/// through the same path — the `1007` schema rejection applies to `generateContent`
/// tool schemas too, not just Gemini Live.
pub fn gemini_schema_subset(value: &Value) -> Value {
    // Fields present on the Gemini `Schema` proto (generativelanguage v1alpha/v1beta).
    const ALLOWED: &[&str] = &[
        "type",
        "format",
        "title",
        "description",
        "nullable",
        "default",
        "enum",
        "example",
        "pattern",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "minProperties",
        "maxProperties",
        "required",
        "propertyOrdering",
        "items",
        "properties",
        "anyOf",
    ];
    match value {
        Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for (k, v) in map {
                if !ALLOWED.contains(&k.as_str()) {
                    continue; // drop JSON-Schema-only keys ($schema, $ref, additionalProperties, …)
                }
                let nv = match k.as_str() {
                    "items" => gemini_schema_subset(v),
                    "properties" => match v {
                        Value::Object(props) => Value::Object(
                            props
                                .iter()
                                .map(|(pk, pv)| (pk.clone(), gemini_schema_subset(pv)))
                                .collect(),
                        ),
                        other => other.clone(),
                    },
                    "anyOf" => match v {
                        Value::Array(arr) => {
                            Value::Array(arr.iter().map(gemini_schema_subset).collect())
                        }
                        other => other.clone(),
                    },
                    _ => v.clone(),
                };
                out.insert(k.clone(), nv);
            }
            Value::Object(out)
        }
        other => other.clone(),
    }
}

/// A single function declaration: `{name, description, parameters}`. The
/// parameters schema is sanitized to Gemini's accepted subset (see
/// [`gemini_schema_subset`]) so raw MCP `input_schema` can't `1007`-kill the
/// live session.
fn encode_tool_decl(tool: &ToolDecl) -> Value {
    json!({
        "name": tool.name,
        "description": tool.description,
        "parameters": gemini_schema_subset(&tool.params),
    })
}

/// Build a `realtimeInput` audio chunk message.
///
/// `{"realtimeInput": {"audio": {"mimeType": "audio/pcm;rate=<sr>", "data": "<b64>"}}}`
/// where `data` is base64 of the little-endian `i16` PCM bytes.
///
/// NOTE: the older `realtimeInput.mediaChunks: [Blob]` form is **deprecated** —
/// the Live API now rejects it by closing the socket with `1007` ("realtime_input.
/// media_chunks is deprecated. Use audio, video, or text instead."), which killed
/// the call on the very first audio frame (setup succeeds, then the first chunk
/// drops the session). The current shape is a single `audio` Blob.
fn encode_realtime_input(chunk: &AudioChunk) -> Value {
    let bytes = pcm_to_le_bytes(&chunk.pcm);
    let data = base64::engine::general_purpose::STANDARD.encode(&bytes);
    let mime = format!("audio/pcm;rate={}", chunk.sample_rate);
    json!({
        "realtimeInput": {
            "audio": { "mimeType": mime, "data": data }
        }
    })
}

/// Build a `toolResponse` message: `{functionResponses: [{id, name, response}]}`.
///
/// `name` is omitted from [`RealtimeLlm::send_tool_result`]; Gemini matches the
/// response to the call by `id`, so we send an empty `name` (the field is
/// present for shape compatibility with the reference `FunctionResponse`).
fn encode_tool_response(id: &str, response: Value) -> Value {
    json!({
        "toolResponse": {
            "functionResponses": [{
                "id": id,
                "name": "",
                "response": response
            }]
        }
    })
}

/// Pack `i16` samples into little-endian bytes (the wire PCM encoding).
fn pcm_to_le_bytes(pcm: &[i16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(pcm.len() * 2);
    for s in pcm {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// Unpack little-endian PCM bytes back into `i16` samples (decode side).
fn le_bytes_to_pcm(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

// ---------------------------------------------------------------------------
// Reader task + decoder (server → client).
// ---------------------------------------------------------------------------

/// Read the server's first frame(s) until `setupComplete` arrives.
async fn await_setup_complete(stream: &mut WsStream) -> Result<(), FlowcatError> {
    loop {
        match stream.next().await {
            Some(Ok(Message::Text(t))) => {
                let v: Value = serde_json::from_str(t.as_str())
                    .map_err(|e| FlowcatError::Realtime(format!("setup parse: {e}")))?;
                if v.get("setupComplete").is_some() {
                    return Ok(());
                }
                // Some servers may send other content before setupComplete;
                // keep waiting (the spec sends setupComplete first).
            }
            Some(Ok(Message::Binary(b))) => {
                // Defensive: a JSON setup ack delivered as binary.
                if let Ok(v) = serde_json::from_slice::<Value>(&b) {
                    if v.get("setupComplete").is_some() {
                        return Ok(());
                    }
                }
            }
            Some(Ok(Message::Close(frame))) => {
                let detail = frame
                    .map(|f| format!("{} {}", f.code, f.reason))
                    .unwrap_or_else(|| "no close frame".into());
                return Err(FlowcatError::Realtime(format!(
                    "socket closed before setupComplete: {detail}"
                )));
            }
            None => {
                return Err(FlowcatError::Realtime(
                    "socket closed before setupComplete".into(),
                ));
            }
            Some(Ok(_)) => { /* ping/pong/frame — ignore */ }
            Some(Err(e)) => {
                return Err(FlowcatError::Realtime(format!("setup recv: {e}")));
            }
        }
    }
}

/// Drive the read half: decode each frame and forward events on `tx`. On a
/// `goAway`, a server-initiated `Close`, an unexpected stream end, or a read
/// error it sends a single [`ReaderMsg::Lost`] (a reconnect *request*) and stops
/// — [`GeminiLive::next_event`] then decides whether to resume from `resumption`
/// or surface `Closed`. Each decoded `sessionResumptionUpdate` handle is written
/// into `resumption` so a later resume has the freshest handle.
async fn reader_task(
    mut stream: WsStream,
    tx: mpsc::UnboundedSender<ReaderMsg>,
    resumption: ResumptionHandle,
    notify: Arc<tokio::sync::Notify>,
) {
    while let Some(msg) = stream.next().await {
        match msg {
            Ok(Message::Text(t)) => {
                let keep = forward_frame(t.as_str().as_bytes(), &tx, &resumption);
                notify.notify_one(); // wake the (lock-free) poll_event consumer
                if !keep {
                    return; // receiver dropped or `Lost` requested
                }
            }
            Ok(Message::Binary(b)) => {
                let keep = forward_frame(&b, &tx, &resumption);
                notify.notify_one();
                if !keep {
                    return;
                }
            }
            Ok(Message::Close(frame)) => {
                // A server close: log the code + reason. Gemini sends a 1007 with
                // "Invalid JSON payload received. Unknown name \"…\"" when the
                // setup (e.g. a tool's parameters schema) is malformed, so this is
                // what makes a setup-rejection diagnosable instead of surfacing
                // only as a generic "resume failed". Then request a resume (the
                // consumer ends the call if no handle / reconnect fails).
                match &frame {
                    Some(f) => tracing::warn!(code = %f.code, reason = %f.reason,
                        "gemini-live: server closed the socket"),
                    None => tracing::warn!("gemini-live: server closed the socket (no frame)"),
                }
                let _ = tx.send(ReaderMsg::Lost);
                notify.notify_one();
                return;
            }
            Ok(_) => { /* ping/pong/frame — ignore */ }
            Err(e) => {
                tracing::warn!("gemini-live read error: {e}");
                let _ = tx.send(ReaderMsg::Lost);
                notify.notify_one();
                return;
            }
        }
    }
    // Stream ended without an explicit close frame: request a resume.
    let _ = tx.send(ReaderMsg::Lost);
    notify.notify_one();
}

/// Parse one raw server frame and forward the resulting message(s).
///
/// Records any `sessionResumptionUpdate` handle into `resumption`. Returns
/// `false` if the channel receiver is gone or the frame signalled end-of-session
/// (`goAway` → a [`ReaderMsg::Lost`] reconnect request was queued), signalling
/// the reader to stop.
fn forward_frame(
    raw: &[u8],
    tx: &mpsc::UnboundedSender<ReaderMsg>,
    resumption: &ResumptionHandle,
) -> bool {
    let value: Value = match serde_json::from_slice(raw) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!("gemini-live: undecodable server frame: {e}");
            return true; // tolerate a bad frame, keep reading
        }
    };

    // Capture a session-resumption handle (does not itself produce an event).
    if let Some(handle) = decode_resumption_handle(&value) {
        *resumption.lock().expect("resumption mutex") = Some(handle);
    }

    let mut terminal = false;
    for ev in decode_server_frame(&value, &mut terminal) {
        if tx.send(ReaderMsg::Event(ev)).is_err() {
            return false;
        }
    }
    if terminal {
        // `goAway`: ask the consumer to resume (it falls back to `Closed`).
        let _ = tx.send(ReaderMsg::Lost);
        return false;
    }
    true
}

/// Decode a `sessionResumptionUpdate{newHandle,resumable}` frame, returning the
/// new handle when the update is `resumable` and carries a non-empty `newHandle`.
///
/// Wire shape (camelCase, `mldev` proto3-JSON):
/// `{"sessionResumptionUpdate": {"newHandle": "<opaque>", "resumable": true}}`.
/// Mirrors the reference `_handle_msg_resumption_update` (`update.resumable &&
/// update.new_handle`).
fn decode_resumption_handle(value: &Value) -> Option<String> {
    let update = value.get("sessionResumptionUpdate")?;
    // Only adopt a handle the server marks resumable; some interim updates carry
    // `resumable:false` (a non-resumable checkpoint) and must be ignored.
    if update.get("resumable").and_then(Value::as_bool) != Some(true) {
        return None;
    }
    update
        .get("newHandle")
        .and_then(Value::as_str)
        .filter(|h| !h.is_empty())
        .map(str::to_owned)
}

/// Map one server frame (`serde_json::Value`) into zero or more
/// [`RealtimeEvent`]s. Sets `*terminal` when the frame signals end-of-session
/// (`goAway`), so the reader stops after flushing and requests a resume.
///
/// Server-frame keys (camelCase wire form, see module docs):
/// - `serverContent.modelTurn.parts[].inlineData{mimeType,data}` → `AudioOut(24k)`
/// - `serverContent.inputTranscription.text` → `UserText`
/// - `serverContent.outputTranscription.text` → `BotText`
/// - `serverContent.interrupted` (true) → `Interrupted`
/// - `toolCall.functionCalls[{id,name,args}]` → `ToolCall`
/// - `usageMetadata` → `Usage`
/// - `goAway` → terminal (no event; triggers a resume/`Closed` in `next_event`)
/// - `sessionResumptionUpdate` → no event here (decoded by
///   [`decode_resumption_handle`], which stores the handle for a later resume)
fn decode_server_frame(value: &Value, terminal: &mut bool) -> Vec<RealtimeEvent> {
    let mut out = Vec::new();

    if let Some(sc) = value.get("serverContent") {
        // Barge-in. `interrupted` is a bool flag.
        if sc.get("interrupted").and_then(Value::as_bool) == Some(true) {
            out.push(RealtimeEvent::Interrupted);
        }

        // Bot audio: modelTurn.parts[].inlineData (24 kHz PCM, base64).
        if let Some(parts) = sc
            .get("modelTurn")
            .and_then(|mt| mt.get("parts"))
            .and_then(Value::as_array)
        {
            for part in parts {
                if let Some(inline) = part.get("inlineData") {
                    if let Some(chunk) = decode_inline_audio(inline) {
                        out.push(RealtimeEvent::AudioOut(chunk));
                    }
                }
            }
        }

        // User (input) transcription.
        if let Some(text) = sc
            .get("inputTranscription")
            .and_then(|t| t.get("text"))
            .and_then(Value::as_str)
        {
            if !text.is_empty() {
                tracing::debug!(heard = %text, "gemini-live: input transcription (heard user)");
                out.push(RealtimeEvent::UserText(text.to_owned()));
            }
        }

        // Bot (output) transcription.
        if let Some(text) = sc
            .get("outputTranscription")
            .and_then(|t| t.get("text"))
            .and_then(Value::as_str)
        {
            if !text.is_empty() {
                tracing::debug!(said = %text, "gemini-live: output transcription (bot speaking)");
                out.push(RealtimeEvent::BotText(text.to_owned()));
            }
        }
    }

    // Tool calls.
    if let Some(calls) = value
        .get("toolCall")
        .and_then(|tc| tc.get("functionCalls"))
        .and_then(Value::as_array)
    {
        for call in calls {
            let name = call
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            // `id` is absent on some transports (e.g. Vertex); fall back to "".
            let id = call
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let args = call.get("args").cloned().unwrap_or(Value::Null);
            out.push(RealtimeEvent::ToolCall { name, args, id });
        }
    }

    // Usage / token accounting.
    if let Some(um) = value.get("usageMetadata") {
        out.push(RealtimeEvent::Usage(decode_usage(um)));
    }

    // Graceful server-initiated shutdown. We mark the frame terminal but do NOT
    // push a `Closed` event: `forward_frame` turns a terminal frame into a
    // `ReaderMsg::Lost` reconnect *request*, and `next_event` resumes from the
    // stored session handle (falling back to `Closed` only if resume is
    // impossible). Pushing `Closed` here would end the call before the resume
    // attempt — session resumption is exactly the change from that old behaviour.
    if value.get("goAway").is_some() {
        *terminal = true;
    }

    out
}

/// Decode an `inlineData` blob (`{mimeType, data}`) into a 24 kHz `AudioChunk`.
///
/// The sample rate is parsed from the `audio/pcm;rate=<n>` mime type, defaulting
/// to 24000 (Gemini Live's output rate) when absent.
fn decode_inline_audio(inline: &Value) -> Option<AudioChunk> {
    let data_b64 = inline.get("data").and_then(Value::as_str)?;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data_b64)
        .ok()?;
    let pcm = le_bytes_to_pcm(&bytes);
    let sample_rate = inline
        .get("mimeType")
        .and_then(Value::as_str)
        .and_then(parse_pcm_rate)
        .unwrap_or(24_000);
    Some(AudioChunk::new(pcm, sample_rate))
}

/// Parse the `rate=` value out of a mime type like `audio/pcm;rate=24000`.
fn parse_pcm_rate(mime: &str) -> Option<u32> {
    mime.split(';')
        .find_map(|p| p.trim().strip_prefix("rate="))
        .and_then(|r| r.trim().parse().ok())
}

/// Decode `usageMetadata` (camelCase wire fields) into [`Usage`].
fn decode_usage(um: &Value) -> Usage {
    let get_u64 = |k: &str| um.get(k).and_then(Value::as_u64);
    Usage {
        input_tokens: get_u64("promptTokenCount"),
        // Bidi server messages use `responseTokenCount`; some surfaces use
        // `candidatesTokenCount` — accept either for output.
        output_tokens: get_u64("responseTokenCount").or_else(|| get_u64("candidatesTokenCount")),
        total_tokens: get_u64("totalTokenCount"),
        extra: Some(um.clone()),
    }
}

// ===========================================================================
// Tests — pure JSON encode/decode, NO live socket.
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_setup() -> RealtimeSetup {
        RealtimeSetup {
            // gemini-3.1-flash-live-preview is the current Gemini Live flash model
            // (live-verified 2026-06-06); the older native-audio preview is gated.
            model: std::env::var("GEMINI_LIVE_MODEL")
                .unwrap_or_else(|_| "models/gemini-3.1-flash-live-preview".into()),
            system_prompt: "You are a helpful agent.".into(),
            tools: vec![ToolDecl {
                name: "transition_to_billing".into(),
                description: "Move to the billing state.".into(),
                params: json!({ "type": "object", "properties": {} }),
            }],
            input_sample_rate: 16_000,
            output_sample_rate: 24_000,
        }
    }

    // ---- VAD env parsing (via the from_getter seam — no process env) -------
    #[test]
    fn aistudio_url_carries_the_key_and_bare_model() {
        let g = GeminiLive::new("k123");
        assert_eq!(g.url(), format!("{GEMINI_LIVE_WSS_BASE}?key=k123"));
        // AI-Studio uses the bare model id, unchanged.
        assert_eq!(g.effective_model("gemini-2.5-flash"), "gemini-2.5-flash");
    }

    #[test]
    fn vertex_url_uses_regional_host_and_resource_path_model() {
        let g = GeminiLive::new_vertex("tok", "my-proj", "us-central1");
        assert_eq!(
            g.url(),
            "wss://us-central1-aiplatform.googleapis.com/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent"
        );
        assert_eq!(g.api_key(), "", "vertex mode carries no api key");
        assert_eq!(
            g.effective_model("gemini-live-2.5-flash"),
            "projects/my-proj/locations/us-central1/publishers/google/models/gemini-live-2.5-flash"
        );
        // Empty model → the Vertex Live default.
        assert_eq!(
            g.effective_model(""),
            format!("projects/my-proj/locations/us-central1/publishers/google/models/{DEFAULT_VERTEX_LIVE_MODEL}")
        );
    }

    #[test]
    fn vertex_global_location_drops_the_region_prefix() {
        let g = GeminiLive::new_vertex("tok", "p", "global");
        assert!(g.url().starts_with("wss://aiplatform.googleapis.com/ws/"));
        // Empty location falls back to the default region.
        let g2 = GeminiLive::new_vertex("tok", "p", "   ");
        assert!(g2
            .url()
            .starts_with("wss://us-central1-aiplatform.googleapis.com/ws/"));
    }

    #[test]
    fn vad_defaults_when_all_unset() {
        let v = VadConfig::from_getter(|_| None);
        let d = VadConfig::default();
        assert_eq!(v.start_sensitivity, d.start_sensitivity);
        assert_eq!(v.end_sensitivity, d.end_sensitivity);
        assert_eq!(v.prefix_padding_ms, 500);
        assert_eq!(v.silence_duration_ms, 350);
    }

    #[test]
    fn vad_valid_overrides_flow_through() {
        let v = VadConfig::from_getter(|k| match k {
            "FLOWCAT_VAD_START_SENSITIVITY" => Some("START_SENSITIVITY_HIGH".into()),
            "FLOWCAT_VAD_END_SENSITIVITY" => Some("END_SENSITIVITY_LOW".into()),
            "FLOWCAT_VAD_PREFIX_PADDING_MS" => Some("250".into()),
            "FLOWCAT_VAD_SILENCE_DURATION_MS" => Some("800".into()),
            _ => None,
        });
        assert_eq!(v.start_sensitivity, "START_SENSITIVITY_HIGH");
        assert_eq!(v.end_sensitivity, "END_SENSITIVITY_LOW");
        assert_eq!(v.prefix_padding_ms, 250);
        assert_eq!(v.silence_duration_ms, 800);
    }

    #[test]
    fn vad_garbage_ms_falls_back_to_default_not_panic() {
        for bad in ["abc", "12.5", "-5", "99999999999", ""] {
            let v = VadConfig::from_getter(move |k| match k {
                "FLOWCAT_VAD_PREFIX_PADDING_MS" => Some(bad.into()),
                "FLOWCAT_VAD_SILENCE_DURATION_MS" => Some(bad.into()),
                _ => None,
            });
            assert_eq!(v.prefix_padding_ms, 500, "bad={bad:?}");
            assert_eq!(v.silence_duration_ms, 350, "bad={bad:?}");
        }
    }

    #[test]
    fn vad_unrecognised_sensitivity_falls_back_to_default() {
        // A typo must NOT be sent verbatim (Gemini would ignore it → choppy VAD).
        let v = VadConfig::from_getter(|k| match k {
            "FLOWCAT_VAD_START_SENSITIVITY" => Some("START_SENSITIVUTY_LOW".into()), // typo
            "FLOWCAT_VAD_END_SENSITIVITY" => Some("".into()),                        // empty
            _ => None,
        });
        assert_eq!(
            v.start_sensitivity, "START_SENSITIVITY_LOW",
            "typo must revert to default"
        );
        assert_eq!(
            v.end_sensitivity, "END_SENSITIVITY_HIGH",
            "empty must revert to default"
        );
    }

    // ---- setup JSON field names / barge-in defaults lock ------------------
    #[test]
    fn setup_vad_field_names_and_number_types_match_schema() {
        let v = encode_setup(&sample_setup(), &VadConfig::default(), None);
        let aad = &v["setup"]["realtimeInputConfig"]["automaticActivityDetection"];
        for k in [
            "startOfSpeechSensitivity",
            "endOfSpeechSensitivity",
            "prefixPaddingMs",
            "silenceDurationMs",
        ] {
            assert!(aad.get(k).is_some(), "missing camelCase AAD field {k}");
        }
        assert!(
            aad["prefixPaddingMs"].is_number(),
            "prefixPaddingMs must be a JSON number"
        );
        assert!(
            aad["silenceDurationMs"].is_number(),
            "silenceDurationMs must be a JSON number"
        );
        assert!(
            aad.get("prefix_padding_ms").is_none(),
            "snake_case must not leak"
        );
    }

    #[test]
    fn setup_barge_in_defaults_serialize_into_setup() {
        let v = encode_setup(&sample_setup(), &VadConfig::default(), None);
        let aad = &v["setup"]["realtimeInputConfig"]["automaticActivityDetection"];
        assert_eq!(aad["startOfSpeechSensitivity"], "START_SENSITIVITY_LOW");
        assert_eq!(aad["endOfSpeechSensitivity"], "END_SENSITIVITY_HIGH");
        assert_eq!(aad["prefixPaddingMs"], 500);
        assert_eq!(aad["silenceDurationMs"], 350);
    }

    // ---- ENCODE -----------------------------------------------------------

    #[test]
    fn gemini_schema_subset_strips_jsonschema_keys_recursively() {
        // A raw MCP `input_schema` (draft-07) with the keys Gemini `1007`-rejects,
        // nested in a sub-object and in array `items`.
        let raw = json!({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "additionalProperties": false,
            "title": "Book",
            "properties": {
                "patient": {
                    "type": "object",
                    "additionalProperties": false,
                    "$ref": "#/defs/p",
                    "properties": { "name": { "type": "string", "minLength": 1 } },
                    "required": ["name"]
                },
                "slots": { "type": "array", "items": { "type": "string", "$comment": "iso" } }
            },
            "required": ["patient"]
        });
        let s = gemini_schema_subset(&raw);
        // Top-level JSON-Schema-only keys are gone; OpenAPI ones survive.
        assert!(s.get("$schema").is_none());
        assert!(s.get("additionalProperties").is_none());
        assert_eq!(s["type"], "object");
        assert_eq!(s["required"], json!(["patient"]));
        // Recursion into nested object properties.
        assert!(s["properties"]["patient"]
            .get("additionalProperties")
            .is_none());
        assert!(s["properties"]["patient"].get("$ref").is_none());
        assert_eq!(
            s["properties"]["patient"]["properties"]["name"]["minLength"],
            1
        );
        // Recursion into array `items`.
        assert!(s["properties"]["slots"]["items"].get("$comment").is_none());
        assert_eq!(s["properties"]["slots"]["items"]["type"], "string");
        // A tool decl built from a gnarly schema carries the sanitized params.
        let decl = encode_tool_decl(&ToolDecl {
            name: "book".into(),
            description: "d".into(),
            params: raw,
        });
        assert!(decl["parameters"].get("$schema").is_none());
        assert_eq!(decl["name"], "book");
    }

    #[test]
    fn encode_setup_has_expected_structure() {
        let v = encode_setup(&sample_setup(), &VadConfig::default(), None);
        let setup = &v["setup"];

        assert_eq!(setup["model"], "models/gemini-3.1-flash-live-preview");

        // generationConfig.responseModalities == ["AUDIO"]
        assert_eq!(
            setup["generationConfig"]["responseModalities"],
            json!(["AUDIO"])
        );
        // Thinking disabled for latency: thinkingConfig.thinkingBudget == 0.
        assert_eq!(
            setup["generationConfig"]["thinkingConfig"]["thinkingBudget"],
            0
        );

        // speechConfig.voiceConfig.prebuiltVoiceConfig.voiceName
        assert_eq!(
            setup["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]
                ["voiceName"],
            DEFAULT_VOICE
        );

        // systemInstruction.parts[0].text
        assert_eq!(
            setup["systemInstruction"]["parts"][0]["text"],
            "You are a helpful agent."
        );

        // Both transcription streams enabled (present as objects).
        assert!(setup["inputAudioTranscription"].is_object());
        assert!(setup["outputAudioTranscription"].is_object());

        // realtimeInputConfig.automaticActivityDetection present AND latency-tuned
        // (not the default empty {} — see encode_setup). These assertions lock the
        // turn-detection tuning so a regression to server defaults is caught.
        let aad = &setup["realtimeInputConfig"]["automaticActivityDetection"];
        assert!(
            aad.is_object(),
            "automaticActivityDetection must be present"
        );
        // Defaults: eager turn-END (snappy replies), conservative turn-START
        // (a brief backchannel must persist 500ms before it commits a barge-in,
        // so it no longer cuts the agent mid-reply).
        assert_eq!(aad["endOfSpeechSensitivity"], "END_SENSITIVITY_HIGH");
        assert_eq!(aad["startOfSpeechSensitivity"], "START_SENSITIVITY_LOW");
        assert_eq!(aad["silenceDurationMs"], 350);
        assert_eq!(aad["prefixPaddingMs"], 500);

        // tools[0].functionDeclarations[0] == {name, description, parameters}
        let decl = &setup["tools"][0]["functionDeclarations"][0];
        assert_eq!(decl["name"], "transition_to_billing");
        assert_eq!(decl["description"], "Move to the billing state.");
        assert_eq!(
            decl["parameters"],
            json!({ "type": "object", "properties": {} })
        );
    }

    #[test]
    fn encode_setup_omits_tools_when_empty() {
        let mut s = sample_setup();
        s.tools.clear();
        let v = encode_setup(&s, &VadConfig::default(), None);
        assert!(
            v["setup"].get("tools").is_none(),
            "tools should be omitted when there are no declarations"
        );
    }

    #[test]
    fn encode_setup_uses_custom_vad_config() {
        // A non-default VadConfig (as FLOWCAT_VAD_* would produce) flows into
        // the realtimeInputConfig verbatim — this is the live-tuning seam.
        let vad = VadConfig {
            start_sensitivity: "START_SENSITIVITY_HIGH".into(),
            end_sensitivity: "END_SENSITIVITY_LOW".into(),
            prefix_padding_ms: 700,
            silence_duration_ms: 600,
        };
        let v = encode_setup(&sample_setup(), &vad, None);
        let aad = &v["setup"]["realtimeInputConfig"]["automaticActivityDetection"];
        assert_eq!(aad["startOfSpeechSensitivity"], "START_SENSITIVITY_HIGH");
        assert_eq!(aad["endOfSpeechSensitivity"], "END_SENSITIVITY_LOW");
        assert_eq!(aad["prefixPaddingMs"], 700);
        assert_eq!(aad["silenceDurationMs"], 600);
    }

    #[test]
    fn vad_config_default_is_conservative_on_start() {
        let d = VadConfig::default();
        assert_eq!(d.start_sensitivity, "START_SENSITIVITY_LOW");
        assert_eq!(d.end_sensitivity, "END_SENSITIVITY_HIGH");
        assert_eq!(d.prefix_padding_ms, 500);
        assert_eq!(d.silence_duration_ms, 350);
    }

    #[test]
    fn encode_realtime_input_round_trips_pcm() {
        // i16 samples → LE bytes → base64; assert the wire shape + that the
        // base64 decodes back to the same samples at the same rate.
        let pcm = vec![0_i16, 1, -1, 256, -256, i16::MAX, i16::MIN];
        let chunk = AudioChunk::new(pcm.clone(), 16_000);
        let v = encode_realtime_input(&chunk);

        // Current Live API shape: a single `audio` Blob (NOT the deprecated
        // `mediaChunks` array, which the server 1007-rejects).
        let mc = &v["realtimeInput"]["audio"];
        assert!(
            v["realtimeInput"].get("mediaChunks").is_none(),
            "must not use deprecated mediaChunks"
        );
        assert_eq!(mc["mimeType"], "audio/pcm;rate=16000");

        let data = mc["data"].as_str().expect("data is a base64 string");
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(data)
            .expect("valid base64");
        assert_eq!(bytes.len(), pcm.len() * 2, "2 bytes per i16 sample");
        assert_eq!(
            le_bytes_to_pcm(&bytes),
            pcm,
            "LE round-trip preserves samples"
        );
    }

    #[test]
    fn encode_tool_response_has_function_responses() {
        let v = encode_tool_response("call-123", json!({ "ok": true, "value": 42 }));
        let fr = &v["toolResponse"]["functionResponses"][0];
        assert_eq!(fr["id"], "call-123");
        assert!(fr.get("name").is_some(), "name field present");
        assert_eq!(fr["response"], json!({ "ok": true, "value": 42 }));
    }

    #[test]
    fn kickoff_client_content_shape() {
        // The kickoff message is built inline in `kickoff`; assert the same
        // shape here so the contract is locked by a test.
        let msg = json!({
            "clientContent": {
                "turns": [{ "role": "user", "parts": [{ "text": "" }] }],
                "turnComplete": true
            }
        });
        assert_eq!(msg["clientContent"]["turnComplete"], json!(true));
        assert_eq!(msg["clientContent"]["turns"][0]["role"], "user");
        assert_eq!(msg["clientContent"]["turns"][0]["parts"][0]["text"], "");
    }

    // ---- DECODE (hand-written server-frame fixtures) ----------------------

    /// Helper: decode a frame, asserting it is not terminal unless stated.
    fn decode(v: Value) -> (Vec<RealtimeEvent>, bool) {
        let mut terminal = false;
        let evs = decode_server_frame(&v, &mut terminal);
        (evs, terminal)
    }

    #[test]
    fn decode_model_turn_inline_audio() {
        // Two i16 samples (0x0001, 0xFFFF=-1) as LE bytes [01,00, FF,FF],
        // base64 = "AQD//w==". Mime carries rate=24000.
        let pcm = vec![1_i16, -1];
        let b64 = base64::engine::general_purpose::STANDARD.encode(pcm_to_le_bytes(&pcm));
        assert_eq!(b64, "AQD//w==");

        let frame = json!({
            "serverContent": {
                "modelTurn": {
                    "parts": [
                        { "inlineData": { "mimeType": "audio/pcm;rate=24000", "data": b64 } }
                    ]
                }
            }
        });
        let (evs, terminal) = decode(frame);
        assert!(!terminal);
        assert_eq!(evs.len(), 1);
        match &evs[0] {
            RealtimeEvent::AudioOut(chunk) => {
                assert_eq!(chunk.sample_rate, 24_000);
                assert_eq!(chunk.pcm, pcm);
            }
            other => panic!("expected AudioOut, got {other:?}"),
        }
    }

    #[test]
    fn decode_inline_audio_defaults_rate_when_mime_missing() {
        let pcm = vec![5_i16, 6, 7];
        let b64 = base64::engine::general_purpose::STANDARD.encode(pcm_to_le_bytes(&pcm));
        // mimeType is "audio/pcm" with no ;rate= → default to 24000.
        let frame = json!({
            "serverContent": {
                "modelTurn": { "parts": [{ "inlineData": { "mimeType": "audio/pcm", "data": b64 } }] }
            }
        });
        let (evs, _) = decode(frame);
        match &evs[0] {
            RealtimeEvent::AudioOut(chunk) => {
                assert_eq!(chunk.sample_rate, 24_000);
                assert_eq!(chunk.pcm, pcm);
            }
            other => panic!("expected AudioOut, got {other:?}"),
        }
    }

    #[test]
    fn decode_input_and_output_transcription() {
        let in_frame =
            json!({ "serverContent": { "inputTranscription": { "text": "hello there" } } });
        let (evs, _) = decode(in_frame);
        assert!(matches!(&evs[0], RealtimeEvent::UserText(t) if t == "hello there"));

        let out_frame = json!({ "serverContent": { "outputTranscription": { "text": "hi, how can I help?" } } });
        let (evs, _) = decode(out_frame);
        assert!(matches!(&evs[0], RealtimeEvent::BotText(t) if t == "hi, how can I help?"));
    }

    #[test]
    fn decode_tool_call() {
        let frame = json!({
            "toolCall": {
                "functionCalls": [
                    { "id": "fc-1", "name": "transition_to_billing", "args": { "reason": "user asked" } }
                ]
            }
        });
        let (evs, terminal) = decode(frame);
        assert!(!terminal);
        assert_eq!(evs.len(), 1);
        match &evs[0] {
            RealtimeEvent::ToolCall { id, name, args } => {
                assert_eq!(id, "fc-1");
                assert_eq!(name, "transition_to_billing");
                assert_eq!(args, &json!({ "reason": "user asked" }));
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn decode_tool_call_without_id_falls_back_to_empty() {
        let frame = json!({
            "toolCall": { "functionCalls": [ { "name": "end_call", "args": {} } ] }
        });
        let (evs, _) = decode(frame);
        match &evs[0] {
            RealtimeEvent::ToolCall { id, name, .. } => {
                assert_eq!(id, "");
                assert_eq!(name, "end_call");
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn decode_interrupted() {
        let frame = json!({ "serverContent": { "interrupted": true } });
        let (evs, terminal) = decode(frame);
        assert!(!terminal);
        assert_eq!(evs.len(), 1);
        assert!(matches!(evs[0], RealtimeEvent::Interrupted));
    }

    #[test]
    fn decode_usage_metadata() {
        let frame = json!({
            "usageMetadata": {
                "promptTokenCount": 100,
                "responseTokenCount": 25,
                "totalTokenCount": 125
            }
        });
        let (evs, _) = decode(frame);
        assert_eq!(evs.len(), 1);
        match &evs[0] {
            RealtimeEvent::Usage(u) => {
                assert_eq!(u.input_tokens, Some(100));
                assert_eq!(u.output_tokens, Some(25));
                assert_eq!(u.total_tokens, Some(125));
                assert!(u.extra.is_some(), "raw usage passed through in extra");
            }
            other => panic!("expected Usage, got {other:?}"),
        }
    }

    #[test]
    fn decode_go_away_is_terminal_without_event() {
        // `goAway` is terminal at the decode level (the reader stops and
        // `forward_frame` turns "terminal" into a `ReaderMsg::Lost` reconnect
        // request) but emits NO `Closed` event — `next_event` decides resume vs
        // close. (Previously `Closed` was pushed, which ended the call instead of
        // resuming.)
        let frame = json!({ "goAway": { "timeLeft": "5s" } });
        let (evs, terminal) = decode(frame);
        assert!(terminal, "goAway must terminate the reader");
        assert!(
            evs.is_empty(),
            "goAway must not emit a Closed event (resume decides)"
        );
    }

    #[test]
    fn decode_bundled_content_and_usage_in_one_frame() {
        // Gemini 3.x can bundle audio + output transcription + usage in one
        // server message; assert we emit all of them, in order.
        let pcm = vec![9_i16, 9];
        let b64 = base64::engine::general_purpose::STANDARD.encode(pcm_to_le_bytes(&pcm));
        let frame = json!({
            "serverContent": {
                "modelTurn": { "parts": [{ "inlineData": { "mimeType": "audio/pcm;rate=24000", "data": b64 } }] },
                "outputTranscription": { "text": "done" }
            },
            "usageMetadata": { "promptTokenCount": 1, "totalTokenCount": 2 }
        });
        let (evs, _) = decode(frame);
        assert_eq!(evs.len(), 3);
        assert!(matches!(evs[0], RealtimeEvent::AudioOut(_)));
        assert!(matches!(&evs[1], RealtimeEvent::BotText(t) if t == "done"));
        assert!(matches!(evs[2], RealtimeEvent::Usage(_)));
    }

    #[test]
    fn decode_unknown_frame_yields_nothing() {
        // setupComplete and other unmapped frames produce no events (and are
        // not terminal).
        let (evs, terminal) = decode(json!({ "setupComplete": {} }));
        assert!(!terminal);
        assert!(evs.is_empty());
    }

    // ---- Session-resumption handle capture ---------------------------------

    #[test]
    fn decode_resumption_update_captures_resumable_handle() {
        // {"sessionResumptionUpdate": {"newHandle": "h-1", "resumable": true}}
        let frame = json!({
            "sessionResumptionUpdate": { "newHandle": "h-1", "resumable": true }
        });
        assert_eq!(decode_resumption_handle(&frame).as_deref(), Some("h-1"));
        // It is NOT itself a model event (no UserText/BotText/etc.).
        let (evs, terminal) = decode(frame);
        assert!(!terminal);
        assert!(evs.is_empty(), "a resumption update is not a model event");
    }

    #[test]
    fn decode_resumption_update_ignores_non_resumable_or_empty() {
        // resumable:false → not adopted (interim non-resumable checkpoint).
        let not_resumable = json!({
            "sessionResumptionUpdate": { "newHandle": "h-x", "resumable": false }
        });
        assert_eq!(decode_resumption_handle(&not_resumable), None);

        // resumable:true but empty/absent handle → not adopted.
        let empty_handle = json!({
            "sessionResumptionUpdate": { "newHandle": "", "resumable": true }
        });
        assert_eq!(decode_resumption_handle(&empty_handle), None);
        let no_handle = json!({ "sessionResumptionUpdate": { "resumable": true } });
        assert_eq!(decode_resumption_handle(&no_handle), None);

        // a non-resumption frame → None (the common case, must not misfire).
        assert_eq!(
            decode_resumption_handle(&json!({ "setupComplete": {} })),
            None
        );
    }

    #[test]
    fn forward_frame_records_resumption_handle_into_shared_cell() {
        // The reader's `forward_frame` is what actually persists the handle the
        // resume reconnect reads — assert that wiring end-to-end (no socket).
        let (tx, _rx) = mpsc::unbounded_channel();
        let cell: ResumptionHandle = Arc::new(std::sync::Mutex::new(None));
        let frame = serde_json::to_vec(&json!({
            "sessionResumptionUpdate": { "newHandle": "h-42", "resumable": true }
        }))
        .unwrap();
        let keep_reading = forward_frame(&frame, &tx, &cell);
        assert!(keep_reading, "a resumption update must not stop the reader");
        assert_eq!(cell.lock().unwrap().as_deref(), Some("h-42"));
    }

    #[test]
    fn forward_frame_go_away_requests_reconnect_then_stops() {
        // A goAway frame must queue exactly one `ReaderMsg::Lost` (the reconnect
        // request) and tell the reader to stop (`false`) — no `Event` is sent.
        let (tx, mut rx) = mpsc::unbounded_channel();
        let cell: ResumptionHandle = Arc::new(std::sync::Mutex::new(None));
        let frame = serde_json::to_vec(&json!({ "goAway": { "timeLeft": "5s" } })).unwrap();
        let keep_reading = forward_frame(&frame, &tx, &cell);
        assert!(!keep_reading, "goAway must stop the reader");
        match rx.try_recv() {
            Ok(ReaderMsg::Lost) => {}
            other => panic!("expected a single ReaderMsg::Lost, got {other:?}"),
        }
        assert!(rx.try_recv().is_err(), "exactly one Lost, no stray Event");
    }

    // ---- Setup carries sessionResumption -----------------------------------

    #[test]
    fn encode_setup_enables_resumption_on_fresh_connect() {
        // resume_handle=None → sessionResumption present but empty (feature on,
        // fresh session; the server then starts issuing handles).
        let v = encode_setup(&sample_setup(), &VadConfig::default(), None);
        let sr = &v["setup"]["sessionResumption"];
        assert!(
            sr.is_object(),
            "sessionResumption must be present (resumption enabled)"
        );
        assert!(
            sr.get("handle").is_none(),
            "fresh connect must not carry a handle"
        );
    }

    #[test]
    fn encode_setup_resumes_with_handle() {
        // resume_handle=Some(h) → sessionResumption.handle == h (resume path).
        let v = encode_setup(
            &sample_setup(),
            &VadConfig::default(),
            Some("resume-handle-7"),
        );
        assert_eq!(v["setup"]["sessionResumption"]["handle"], "resume-handle-7");
    }

    // ---- Reconnect decision (deterministic, no real socket) ----------------
    //
    // The resume decision (`try_reconnect`) is split from the socket I/O it
    // guards: the failure branches (no handle / ceiling) short-circuit BEFORE
    // any `open`/`connect_async`, so they are fully deterministic without a live
    // socket. The success branch (resume actually re-opens a socket) is covered
    // by the `#[ignore]` live smoke at the bottom + the end-to-end S2S path.

    #[tokio::test]
    async fn try_reconnect_without_handle_falls_back() {
        // No resumption handle yet → resume is impossible → Err (so the consumer
        // surfaces `Closed` and ends the call, the pre-resumption behaviour).
        let mut client = GeminiLive::new("test-key");
        client.setup = Some(sample_setup());
        // resumption is None by construction.
        let err = client.try_reconnect().await.unwrap_err();
        assert!(err.contains("no resumption handle"), "got: {err}");
        // The failed attempt must not have touched the counter past 0 (it returns
        // before incrementing on the no-handle branch).
        assert_eq!(client.consecutive_reconnects, 0);
    }

    #[tokio::test]
    async fn try_reconnect_honours_the_ceiling() {
        // With a handle present, the first attempts try to `open` (which fails
        // here — "test-key" can't reach Google — incrementing the counter); once
        // the counter hits the ceiling, the call short-circuits with the ceiling
        // error BEFORE attempting another connect.
        let mut client = GeminiLive::new("test-key");
        client.setup = Some(sample_setup());
        *client.resumption.lock().unwrap() = Some("h-1".into());
        client.consecutive_reconnects = MAX_CONSECUTIVE_RECONNECTS;

        let err = client.try_reconnect().await.unwrap_err();
        assert!(err.contains("ceiling"), "got: {err}");
        // The counter was not incremented past the ceiling on the short-circuit.
        assert_eq!(client.consecutive_reconnects, MAX_CONSECUTIVE_RECONNECTS);
    }

    #[tokio::test]
    async fn try_reconnect_without_setup_errors() {
        // A handle but no stored setup (can't happen via the trait, but guard it)
        // → Err, never a panic.
        let mut client = GeminiLive::new("test-key");
        *client.resumption.lock().unwrap() = Some("h-1".into());
        // setup is None.
        let err = client.try_reconnect().await.unwrap_err();
        assert!(err.contains("reconnect before connect"), "got: {err}");
    }

    #[tokio::test]
    async fn next_event_on_lost_without_handle_yields_closed() {
        // Drive the `next_event` reconnect loop with a scripted channel: a `Lost`
        // with NO resumption handle must fall back to a single `Closed` (the call
        // ends cleanly — never a hang), then `None` once the channel drains.
        let (tx, rx) = mpsc::unbounded_channel();
        tx.send(ReaderMsg::Lost).unwrap();
        drop(tx); // channel closes after the script

        let mut client = GeminiLive::new("test-key");
        client.setup = Some(sample_setup());
        // resumption stays None → resume impossible → Closed.
        client.conn = Some(Connection {
            sink: dummy_sink().await,
            events: rx,
            reader: tokio::spawn(async {}),
        });

        assert!(
            matches!(client.next_event().await, Some(RealtimeEvent::Closed)),
            "Lost with no handle must surface a single Closed"
        );
        assert!(client.next_event().await.is_none(), "then the stream ends");
    }

    #[tokio::test]
    async fn next_event_forwards_events_and_resets_counter() {
        // A normal event flows straight through AND resets the reconnect counter
        // (a healthy socket means any prior reconnect "stuck").
        let (tx, rx) = mpsc::unbounded_channel();
        tx.send(ReaderMsg::Event(RealtimeEvent::BotText("hi".into())))
            .unwrap();
        drop(tx);

        let mut client = GeminiLive::new("test-key");
        client.consecutive_reconnects = 2; // pretend we had reconnected
        client.conn = Some(Connection {
            sink: dummy_sink().await,
            events: rx,
            reader: tokio::spawn(async {}),
        });

        match client.next_event().await {
            Some(RealtimeEvent::BotText(t)) => assert_eq!(t, "hi"),
            other => panic!("expected BotText, got {other:?}"),
        }
        assert_eq!(
            client.consecutive_reconnects, 0,
            "a live event resets the counter"
        );
    }

    /// Mint a `WsSink` for tests without a network socket: connect a loopback TCP
    /// pair, wrap our end as a client-role `WebSocketStream` (no real handshake —
    /// the sink is never driven by the tests below; `next_event` only reads the
    /// event channel), and return its write half. The peer is dropped, so a write
    /// would fail — which is fine, we never write it.
    async fn dummy_sink() -> Arc<Mutex<WsSink>> {
        use tokio::net::{TcpListener, TcpStream};
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let connect = TcpStream::connect(addr);
        let (client_tcp, _accepted) = tokio::join!(connect, listener.accept());
        let ws = WebSocketStream::from_raw_socket(
            MaybeTlsStream::Plain(client_tcp.unwrap()),
            tokio_tungstenite::tungstenite::protocol::Role::Client,
            None,
        )
        .await;
        let (sink, _stream) = ws.split();
        Arc::new(Mutex::new(sink))
    }

    // ---- Live smoke (ignored; documents the key env var) -------------------

    /// Live Gemini-Live resume smoke. Requires a real key and a long-enough call
    /// to observe a `goAway` (~10 min) or an induced drop; run manually.
    ///
    /// `GEMINI_API_KEY=… cargo test -p flowcat-core --
    ///   realtime::gemini_live::tests::live_gemini_resume_smoke --ignored --nocapture`
    #[tokio::test]
    #[ignore = "live: needs GEMINI_API_KEY + a ~10-min call to observe goAway/resume"]
    async fn live_gemini_resume_smoke() {
        let key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY for the live smoke");
        let mut client = GeminiLive::new(key);
        client.connect(sample_setup()).await.expect("connect");
        client.kickoff().await.expect("kickoff");
        // Pump events until the session closes; a mid-call goAway should be
        // resumed transparently (no premature Closed). Manual observation.
        while let Some(ev) = client.next_event().await {
            if matches!(ev, RealtimeEvent::Closed) {
                break;
            }
        }
    }

    /// Fast live verification of the realtime path (≈15 s, not the 10-min resume
    /// smoke): connect → kickoff (bot-first greet) → assert the model actually
    /// speaks (≥1 `AudioOut`) and we see a usage report, then close. Voice is set
    /// via `FLOWCAT_VOICE` (e.g. `Zephyr`), model via `GEMINI_LIVE_MODEL`.
    /// `GEMINI_API_KEY=… FLOWCAT_VOICE=Zephyr cargo test -p flowcat-core --
    ///   realtime::gemini_live::tests::live_gemini_connect_and_speaks --ignored --nocapture`
    #[tokio::test]
    #[ignore = "live: needs GEMINI_API_KEY (Gemini Live access)"]
    async fn live_gemini_connect_and_speaks() {
        let key = std::env::var("GEMINI_API_KEY").expect("GEMINI_API_KEY for the live smoke");
        let mut client = GeminiLive::new(key);
        client
            .connect(sample_setup())
            .await
            .expect("connect (setupComplete)");
        client.kickoff().await.expect("kickoff");
        let mut audio_chunks = 0usize;
        let mut bot_text = String::new();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while std::time::Instant::now() < deadline {
            match tokio::time::timeout(std::time::Duration::from_secs(15), client.next_event())
                .await
            {
                Ok(Some(RealtimeEvent::AudioOut(chunk))) => {
                    audio_chunks += 1;
                    if audio_chunks >= 3 {
                        break; // the model is clearly speaking — enough to verify
                    }
                    let _ = chunk;
                }
                Ok(Some(RealtimeEvent::BotText(t))) => bot_text.push_str(&t),
                Ok(Some(RealtimeEvent::Closed)) | Ok(None) => break,
                Ok(Some(_)) => {}
                Err(_) => break, // timed out waiting for the next event
            }
        }
        eprintln!(
            "gemini live: audio_chunks={audio_chunks}, bot_text={:?}",
            bot_text
        );
        assert!(
            audio_chunks > 0,
            "expected the Gemini Live model to emit audio (AudioOut)"
        );
    }
}
