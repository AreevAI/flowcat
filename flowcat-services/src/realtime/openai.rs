// SPDX-License-Identifier: Apache-2.0
//
//! **OpenAI Realtime** (speech-to-speech) client — the priority second
//! realtime provider after Gemini Live.
//!
//! Impls [`RealtimeLlmService`](flowcat_core::service::RealtimeLlmService) over
//! the OpenAI Realtime **WebSocket** API (GA `gpt-realtime` surface). Behind the
//! `realtime-openai` feature (pulls `tokio-tungstenite`, rustls).
//!
//! ## Wire protocol (server-to-server WebSocket)
//!
//! Connects to `wss://api.openai.com/v1/realtime?model=<model>` with an
//! `Authorization: Bearer <key>` header and speaks the JSON event protocol
//! (<https://platform.openai.com/docs/api-reference/realtime>):
//!
//! - **client→server:** `session.update` (instructions + tools + audio formats +
//!   server VAD + transcription), `input_audio_buffer.append` (base64 PCM in),
//!   `conversation.item.create` (a `function_call_output` item) + `response.create`
//!   (return a tool result), `response.create` (kick off a bot-first turn).
//! - **server→client:** `response.output_audio.delta` (base64 PCM out),
//!   `response.output_audio_transcript.delta` (bot transcript),
//!   `conversation.item.input_audio_transcription.{delta,completed}` (user
//!   transcript), `response.function_call_arguments.done` (a complete tool call),
//!   `input_audio_buffer.speech_started` (barge-in), `response.done` (usage),
//!   `error`, plus a server `close`.
//!
//! The exact event names + field shapes were cross-checked against the vendored
//! reference `pipecat/src/pipecat/services/openai/realtime/{events,llm}.py`.
//!
//! ## Audio rates
//!
//! OpenAI Realtime uses **24 kHz** mono PCM in *and* out (`audio/pcm`, rate
//! 24000). The pipeline resamples the carrier↔24k at the edges; this client
//! tags appended audio and decodes returned audio at the rate from
//! [`RealtimeServiceSetup`] (defaulting to 24000), keeping the seam honest if a
//! future transport negotiates μ-law (`audio/pcmu`).
//!
//! ## Keys / auth (security note)
//!
//! The API key is held on the struct (passed to [`OpenAiRealtime::new`]) and used
//! **only** in the `Authorization` header at connect — it is never logged and is
//! not carried in [`RealtimeServiceSetup`]. The `?model=` query carries no secret.

use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine as _;
use futures::stream::{SplitSink, SplitStream};
use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::sync::{mpsc, Mutex, Notify};
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::{HeaderName, HeaderValue};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

use flowcat_core::error::FlowcatError;
use flowcat_core::processor::frame::AudioFrame;
use flowcat_core::realtime::PollEvent;
use flowcat_core::service::{RealtimeLlmService, RealtimeServiceSetup, Tool};
use flowcat_core::types::{AudioChunk, RealtimeEvent, ToolDecl, Usage};

/// Base WSS endpoint for the OpenAI Realtime service. `?model=<model>` is
/// appended at connect time.
pub const OPENAI_REALTIME_WSS_BASE: &str = "wss://api.openai.com/v1/realtime";

/// Default model if [`RealtimeServiceSetup::model`] is empty.
const DEFAULT_MODEL: &str = "gpt-realtime";

/// Default output voice (overridable via `FLOWCAT_VOICE`).
const DEFAULT_VOICE: &str = "alloy";

/// OpenAI Realtime PCM rate (mono 16-bit), used in *both* directions.
const OPENAI_SAMPLE_RATE: u32 = 24_000;

type ClientSocket = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;
type WsSink = SplitSink<ClientSocket, Message>;
type WsStream = SplitStream<ClientSocket>;

/// Live connection state, present only while connected.
struct Connection {
    sink: Arc<Mutex<WsSink>>,
    events: mpsc::UnboundedReceiver<RealtimeEvent>,
    reader: JoinHandle<()>,
}

impl Connection {
    async fn send_json(&self, value: &Value) -> Result<(), FlowcatError> {
        let text = serde_json::to_string(value)?;
        let mut sink = self.sink.lock().await;
        sink.send(Message::text(text))
            .await
            .map_err(|e| FlowcatError::Realtime(format!("ws send: {e}")))
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        self.reader.abort();
    }
}

/// A native OpenAI Realtime speech-to-speech session.
pub struct OpenAiRealtime {
    api_key: String,
    /// Connect endpoint base (overridden for Azure/Grok, whose endpoints are
    /// OpenAI-Realtime-compatible); defaults to [`OPENAI_REALTIME_WSS_BASE`].
    base_url: String,
    /// Extra connect headers (Azure uses `api-key` instead of `Authorization`).
    /// Values are secret-bearing and never logged.
    extra_headers: Vec<(String, String)>,
    /// Optional ISO-639-1 language hint for the INPUT transcription (e.g. `"en"`).
    /// `None` → the transcription model auto-detects (any language).
    language: Option<String>,
    conn: Option<Connection>,
    setup: Option<RealtimeServiceSetup>,
    /// Readiness notify for the lock-free event path, created **once** and reused
    /// across every [`open`](Self::open) (incl. a [`rebase_session`] reconnect). The
    /// s2s reader captures this handle a single time at startup, so it must stay
    /// stable across reconnects — a per-connection notify would strand the reader on
    /// a dead handle after a re-base (mirrors [`GeminiLive`]'s persistent
    /// `event_notify`).
    event_notify: Arc<Notify>,
    /// The session's `turn_detection` config. Defaults to `{"type":"semantic_vad"}`
    /// (a model decides when the caller is *semantically* done — best for live human
    /// calls). Overridable to `server_vad` (endpoint purely on a silence duration),
    /// which is more robust for non-interactive / synthetic / file-streamed audio.
    turn_detection: Value,
}

impl OpenAiRealtime {
    /// Construct a client bound to the given OpenAI API key (Bearer auth).
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: OPENAI_REALTIME_WSS_BASE.to_string(),
            extra_headers: Vec::new(),
            language: None,
            conn: None,
            setup: None,
            event_notify: Arc::new(Notify::new()),
            turn_detection: json!({ "type": "semantic_vad" }),
        }
    }

    /// Set the INPUT-transcription language hint (ISO-639-1, e.g. `Some("en")`).
    /// `None` leaves the transcription model to auto-detect — so a caller may
    /// speak any language. A hint helps the model lock a known language and avoids
    /// the occasional mis-detection (e.g. English heard as another script).
    ///
    /// NOTE: `gpt-4o-transcribe` (the default input-transcription model below) may
    /// not always honor this hint; if reliable locking is needed, switch that model
    /// to `whisper-1` (which respects `language`) — a one-line follow-up.
    pub fn with_input_language(mut self, language: Option<String>) -> Self {
        self.language = language.filter(|s| !s.is_empty());
        self
    }

    /// Override the session `turn_detection` JSON (the default is
    /// `{"type":"semantic_vad"}`). Pass a full object, e.g.
    /// `json!({"type":"server_vad","silence_duration_ms":500})`.
    pub fn with_turn_detection(mut self, turn_detection: Value) -> Self {
        self.turn_detection = turn_detection;
        self
    }

    /// Use **server VAD** (silence-duration endpointing) instead of the default
    /// semantic VAD. More robust for non-interactive callers — synthetic speech,
    /// file-streamed audio, or any source whose end-of-turn is a plain pause rather
    /// than a semantic cue (semantic VAD can fail to endpoint such audio). Auto-
    /// creates the bot response on turn end, matching the semantic-VAD default.
    pub fn with_server_vad(self) -> Self {
        self.with_turn_detection(json!({
            "type": "server_vad",
            "silence_duration_ms": 500,
            "create_response": true,
        }))
    }

    /// Override the WSS base URL (used by the Azure/Grok wrappers). The
    /// `?model=` query is appended only when the URL carries no query.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Replace the connect headers (Azure auth uses `api-key: <key>` and no
    /// Bearer). Each entry is `(header-name, header-value)`. Secret-bearing.
    pub fn with_headers(mut self, headers: Vec<(String, String)>) -> Self {
        self.extra_headers = headers;
        self
    }

    /// The API key this client authenticates with (never logged).
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Build the connect URL (`<base>?model=<model>`), preserving an existing
    /// query (e.g. the Azure `api-version` + `deployment`).
    fn url(&self, model: &str) -> String {
        if self.base_url.contains('?') || self.base_url.contains("model=") {
            self.base_url.clone()
        } else {
            format!("{}?model={}", self.base_url, model)
        }
    }

    /// Open the socket, send `session.update`, and spawn the reader task.
    async fn open(&mut self, setup: &RealtimeServiceSetup) -> Result<(), FlowcatError> {
        self.close_conn().await;

        let model = if setup.model.is_empty() {
            DEFAULT_MODEL
        } else {
            setup.model.as_str()
        };
        let url = self.url(model);

        let mut req = url
            .into_client_request()
            .map_err(|e| FlowcatError::Realtime(format!("bad realtime url: {e}")))?;
        // Auth header(s). Default = Bearer; Azure replaces this via with_headers.
        let headers = req.headers_mut();
        if self.extra_headers.is_empty() {
            let bearer = format!("Bearer {}", self.api_key);
            headers.insert(
                "Authorization",
                HeaderValue::from_str(&bearer)
                    .map_err(|e| FlowcatError::Realtime(format!("bad auth header: {e}")))?,
            );
        } else {
            for (name, value) in &self.extra_headers {
                let hn: HeaderName = name
                    .parse()
                    .map_err(|e| FlowcatError::Realtime(format!("bad header name {name}: {e}")))?;
                headers.insert(
                    hn,
                    HeaderValue::from_str(value)
                        .map_err(|e| FlowcatError::Realtime(format!("bad header value: {e}")))?,
                );
            }
        }

        let (socket, _resp) = connect_async(req)
            .await
            .map_err(|e| FlowcatError::Realtime(format!("connect_async: {e}")))?;
        let (mut sink, stream) = socket.split();

        // Send the initial session.update (instructions + tools + audio config).
        let session = encode_session_update(setup, self.language.as_deref(), &self.turn_detection);
        sink.send(Message::text(serde_json::to_string(&session)?))
            .await
            .map_err(|e| FlowcatError::Realtime(format!("send session.update: {e}")))?;

        let (tx, rx) = mpsc::unbounded_channel();
        let out_rate = out_or_default(setup.output_sample_rate);
        // Reuse the persistent notify (stable across reconnects — see the field doc).
        let reader = tokio::spawn(reader_task(stream, tx, out_rate, self.event_notify.clone()));

        self.conn = Some(Connection {
            sink: Arc::new(Mutex::new(sink)),
            events: rx,
            reader,
        });
        Ok(())
    }

    async fn close_conn(&mut self) {
        if let Some(conn) = self.conn.take() {
            conn.reader.abort();
        }
    }

    fn require_conn(&self) -> Result<&Connection, FlowcatError> {
        self.conn
            .as_ref()
            .ok_or_else(|| FlowcatError::Realtime("not connected".into()))
    }
}

#[async_trait]
impl RealtimeLlmService for OpenAiRealtime {
    async fn connect(&mut self, setup: RealtimeServiceSetup) -> Result<(), FlowcatError> {
        self.open(&setup).await?;
        self.setup = Some(setup);
        Ok(())
    }

    async fn send_audio(&mut self, chunk: Arc<AudioFrame>) -> Result<(), FlowcatError> {
        let msg = encode_audio_append(&chunk.pcm);
        self.require_conn()?.send_json(&msg).await
    }

    async fn update_system(
        &mut self,
        prompt: String,
        tools: Vec<Tool>,
    ) -> Result<(), FlowcatError> {
        // OpenAI Realtime supports in-session updates: a `session.update` with
        // the new instructions + tools (no reconnect needed, unlike Gemini).
        let mut setup = self
            .setup
            .clone()
            .ok_or_else(|| FlowcatError::Realtime("update_system before connect".into()))?;
        setup.system_prompt = prompt;
        setup.tools = tools;
        let msg = encode_session_update(&setup, self.language.as_deref(), &self.turn_detection);
        self.require_conn()?.send_json(&msg).await?;
        self.setup = Some(setup);
        Ok(())
    }

    async fn rebase_session(
        &mut self,
        prompt: String,
        tools: Vec<Tool>,
    ) -> Result<(), FlowcatError> {
        // A ContextRelay re-base must DROP the accumulated audio history — but
        // `update_system` above is an in-session `session.update` that KEEPS every
        // prior audio item in the conversation, so they keep being re-billed each
        // turn. OpenAI Realtime has no reliable in-session "evict all audio" that
        // shrinks the billed context window, so — exactly like Gemini Live's
        // reconnecting `update_system` — we reopen the socket with the new
        // instructions (the caller's text digest) as the fresh session prompt. The
        // new session carries no prior audio; the cheap text digest stands in for it.
        //
        // No re-kickoff: a re-base fires at an idle bot-turn boundary, so the next
        // caller utterance drives the first response on the fresh session (mirrors
        // Gemini's silent reconnect — the bot doesn't re-greet). `open()` reuses the
        // persistent `event_notify`, so the s2s reader keeps waking after the reopen.
        let mut setup = self
            .setup
            .clone()
            .ok_or_else(|| FlowcatError::Realtime("rebase_session before connect".into()))?;
        setup.system_prompt = prompt;
        setup.tools = tools;
        self.open(&setup).await?;
        self.setup = Some(setup);
        Ok(())
    }

    async fn send_tool_result(&mut self, id: String, result: Value) -> Result<(), FlowcatError> {
        // A function_call_output item, then a response.create so the model
        // continues. Mirrors the reference's _handle_function_call return path.
        let conn = self.require_conn()?;
        conn.send_json(&encode_tool_result(&id, &result)).await?;
        conn.send_json(&json!({ "type": "response.create" })).await
    }

    async fn next_event(&mut self) -> Option<RealtimeEvent> {
        self.conn.as_mut()?.events.recv().await
    }

    /// The reader fires this on each queued event, so the pipeline can await
    /// readiness WITHOUT holding the session lock — the lock-free path that keeps
    /// `send_audio` (caller audio) from starving between bot turns.
    fn event_notify(&self) -> Option<Arc<Notify>> {
        // The persistent handle (not the per-connection one) so it stays valid across
        // a `rebase_session` reconnect — the s2s reader captures this exactly once.
        Some(self.event_notify.clone())
    }

    /// Non-blocking poll: `try_recv` the event channel (brief lock, no idle wait).
    /// `Empty` → `Pending` (caller awaits `event_notify`); `Disconnected` → closed.
    async fn poll_event(&mut self) -> PollEvent {
        match self.conn.as_mut() {
            None => PollEvent::Ready(None),
            Some(c) => match c.events.try_recv() {
                Ok(ev) => PollEvent::Ready(Some(ev)),
                Err(mpsc::error::TryRecvError::Empty) => PollEvent::Pending,
                Err(mpsc::error::TryRecvError::Disconnected) => PollEvent::Ready(None),
            },
        }
    }

    /// OpenAI Realtime requires PCM input at ≥ 24 kHz (it 400s on 16 kHz). The
    /// pipeline reads this to resample caller audio + set the session config.
    fn input_sample_rate(&self) -> u32 {
        OPENAI_SAMPLE_RATE
    }

    /// Trigger an initial bot-first turn: a `response.create` (so the agent greets
    /// before the caller speaks).
    async fn kickoff(&mut self) -> Result<(), FlowcatError> {
        self.require_conn()?
            .send_json(&json!({ "type": "response.create" }))
            .await
    }
}

/// Output rate, defaulting to [`OPENAI_SAMPLE_RATE`] when the setup leaves it 0.
fn out_or_default(rate: u32) -> u32 {
    if rate == 0 {
        OPENAI_SAMPLE_RATE
    } else {
        rate
    }
}

// ---------------------------------------------------------------------------
// Encoders (client → server) — pure functions over `serde_json::Value`.
// ---------------------------------------------------------------------------

/// Build the `session.update` event (GA `audio.input/output` shape).
///
/// ```json
/// {"type":"session.update","session":{
///   "type":"realtime",
///   "instructions":"<prompt>",
///   "audio":{
///     "input":{"format":{"type":"audio/pcm","rate":24000},
///              "transcription":{"model":"gpt-4o-transcribe"},
///              "turn_detection":{"type":"server_vad"}},
///     "output":{"format":{"type":"audio/pcm","rate":24000},"voice":"alloy"}
///   },
///   "tools":[{"type":"function","name","description","parameters"}],
///   "tool_choice":"auto"
/// }}
/// ```
fn encode_session_update(
    setup: &RealtimeServiceSetup,
    language: Option<&str>,
    turn_detection: &Value,
) -> Value {
    let voice = std::env::var("FLOWCAT_VOICE")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| DEFAULT_VOICE.to_string());

    let in_rate = out_or_default(setup.input_sample_rate);
    let out_rate = out_or_default(setup.output_sample_rate);

    // Input transcription model + optional language hint.
    //  - No language → `gpt-4o-transcribe` (best-quality auto-detect; any language).
    //  - Language set → `whisper-1`, which RELIABLY honors the `language` param.
    //    `gpt-4o-transcribe` is documented to (and in practice does) ignore the
    //    hint — it leaked Cyrillic for English audio — so we switch models to lock
    //    the configured language hard.
    let transcription = match language {
        Some(lang) => json!({ "model": "whisper-1", "language": lang }),
        None => json!({ "model": "gpt-4o-transcribe" }),
    };

    let mut session = json!({
        "type": "realtime",
        "output_modalities": ["audio"],
        "instructions": setup.system_prompt,
        "audio": {
            "input": {
                "format": { "type": "audio/pcm", "rate": in_rate },
                "transcription": transcription,
                // Turn detection (default `semantic_vad`): a model ends the turn when
                // the user is SEMANTICALLY done, not on raw silence. `server_vad`
                // chunks on every silence gap, so a natural mid-sentence pause splits
                // one utterance into several `input_audio_transcription.completed`
                // events → several transcript bubbles ("Got" / "It ." / …). Semantic
                // VAD keeps an utterance as one segment → one bubble, and is less
                // likely to cut the user off. Barge-in still works (speech_started).
                // Overridable via `with_turn_detection` / `with_server_vad` (server
                // VAD is more robust for non-interactive / synthetic audio).
                "turn_detection": turn_detection.clone()
            },
            "output": {
                "format": { "type": "audio/pcm", "rate": out_rate },
                "voice": voice
            }
        },
        "tool_choice": "auto"
    });

    if !setup.tools.is_empty() {
        let tools: Vec<Value> = setup.tools.iter().map(encode_tool_decl).collect();
        session["tools"] = Value::Array(tools);
    }

    json!({ "type": "session.update", "session": session })
}

/// A single tool declaration as an OpenAI Realtime function tool.
fn encode_tool_decl(tool: &ToolDecl) -> Value {
    json!({
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.params,
    })
}

/// `input_audio_buffer.append` with base64 LE PCM.
fn encode_audio_append(pcm: &[i16]) -> Value {
    let bytes = pcm_to_le_bytes(pcm);
    let data = base64::engine::general_purpose::STANDARD.encode(&bytes);
    json!({ "type": "input_audio_buffer.append", "audio": data })
}

/// `conversation.item.create` carrying a `function_call_output` item.
fn encode_tool_result(call_id: &str, result: &Value) -> Value {
    // `output` must be a string per the API; serialize JSON results.
    let output = match result {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    json!({
        "type": "conversation.item.create",
        "item": {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output
        }
    })
}

fn pcm_to_le_bytes(pcm: &[i16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(pcm.len() * 2);
    for s in pcm {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

fn le_bytes_to_pcm(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

// ---------------------------------------------------------------------------
// Reader task + decoder (server → client).
// ---------------------------------------------------------------------------

async fn reader_task(
    mut stream: WsStream,
    tx: mpsc::UnboundedSender<RealtimeEvent>,
    out_rate: u32,
    notify: Arc<Notify>,
) {
    while let Some(msg) = stream.next().await {
        match msg {
            Ok(Message::Text(t)) => {
                if !forward_frame(t.as_str().as_bytes(), &tx, out_rate, &notify) {
                    return;
                }
            }
            Ok(Message::Binary(b)) => {
                if !forward_frame(&b, &tx, out_rate, &notify) {
                    return;
                }
            }
            Ok(Message::Close(_)) => {
                let _ = tx.send(RealtimeEvent::Closed);
                notify.notify_one();
                return;
            }
            Ok(_) => { /* ping/pong — ignore */ }
            Err(e) => {
                tracing::warn!("openai-realtime read error: {e}");
                let _ = tx.send(RealtimeEvent::Closed);
                notify.notify_one();
                return;
            }
        }
    }
    let _ = tx.send(RealtimeEvent::Closed);
    notify.notify_one();
}

fn forward_frame(
    raw: &[u8],
    tx: &mpsc::UnboundedSender<RealtimeEvent>,
    out_rate: u32,
    notify: &Notify,
) -> bool {
    let value: Value = match serde_json::from_slice(raw) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!("openai-realtime: undecodable frame: {e}");
            return true;
        }
    };
    let mut terminal = false;
    let mut sent_any = false;
    for ev in decode_server_event(&value, out_rate, &mut terminal) {
        if tx.send(ev).is_err() {
            return false;
        }
        sent_any = true;
    }
    // Wake the lock-free poller so it `try_recv`s the just-queued event(s) without
    // the pipeline ever holding the session lock across the idle wait.
    if sent_any {
        notify.notify_one();
    }
    !terminal
}

/// Map one OpenAI Realtime server event into zero or more [`RealtimeEvent`]s.
/// Sets `*terminal` on a fatal `error` (the call cannot continue).
///
/// Recognised `type`s:
/// - `response.output_audio.delta` → `AudioOut` (base64 PCM @ `out_rate`)
/// - `response.output_audio_transcript.delta` / `response.output_text.delta` → `BotText`
/// - `conversation.item.input_audio_transcription.delta` → `UserInterimText` (partial)
/// - `conversation.item.input_audio_transcription.completed` → `UserText` (finalized)
/// - `response.function_call_arguments.done` → `ToolCall`
/// - `input_audio_buffer.speech_started` → `Interrupted` (barge-in)
/// - `response.done` → `Usage`
/// - `error` → `Closed` (terminal)
pub(crate) fn decode_server_event(
    value: &Value,
    out_rate: u32,
    terminal: &mut bool,
) -> Vec<RealtimeEvent> {
    let mut out = Vec::new();
    let kind = value
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default();

    match kind {
        "response.output_audio.delta" => {
            if let Some(b64) = value.get("delta").and_then(Value::as_str) {
                if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(b64) {
                    out.push(RealtimeEvent::AudioOut(AudioChunk::new(
                        le_bytes_to_pcm(&bytes),
                        out_rate,
                    )));
                }
            }
        }
        // Bot transcript (incremental delta).
        "response.output_audio_transcript.delta" | "response.output_text.delta" => {
            push_nonempty(&mut out, value, "delta", RealtimeEvent::BotText);
        }
        // User (input) transcription. Streaming `.delta` events are partials (one
        // per word) → emit as interim so the UI accumulates them into one bubble;
        // `.completed` carries the full utterance → the finalized line.
        "conversation.item.input_audio_transcription.delta" => {
            push_nonempty(&mut out, value, "delta", RealtimeEvent::UserInterimText);
        }
        "conversation.item.input_audio_transcription.completed" => {
            if let Some(t) = value.get("transcript").and_then(Value::as_str) {
                tracing::debug!(segment = %t, "openai-realtime: user transcript segment completed");
            }
            push_nonempty(&mut out, value, "transcript", RealtimeEvent::UserText);
        }
        // A completed function call → ToolCall.
        "response.function_call_arguments.done" => {
            let name = value
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let id = value
                .get("call_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            // `arguments` is a JSON *string*; parse to a Value (Null if blank).
            let args = value
                .get("arguments")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .and_then(|s| serde_json::from_str::<Value>(s).ok())
                .unwrap_or(Value::Null);
            out.push(RealtimeEvent::ToolCall { name, args, id });
        }
        // Barge-in: the server detected the user started speaking.
        "input_audio_buffer.speech_started" => {
            out.push(RealtimeEvent::Interrupted);
        }
        // End of a response carries token usage.
        "response.done" => {
            if let Some(usage) = value.get("response").and_then(|r| r.get("usage")) {
                out.push(RealtimeEvent::Usage(decode_usage(usage)));
            }
        }
        // A server error ends the session.
        "error" => {
            let msg = value
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            tracing::warn!("openai-realtime server error: {msg}");
            out.push(RealtimeEvent::Closed);
            *terminal = true;
        }
        _ => { /* session.created/updated, response.created, etc. — ignored */ }
    }

    out
}

/// Push a `RealtimeEvent` built from a non-empty string field, if present.
fn push_nonempty(
    out: &mut Vec<RealtimeEvent>,
    value: &Value,
    field: &str,
    make: impl Fn(String) -> RealtimeEvent,
) {
    if let Some(s) = value.get(field).and_then(Value::as_str) {
        if !s.is_empty() {
            out.push(make(s.to_owned()));
        }
    }
}

/// Decode `response.usage` (`{input_tokens,output_tokens,total_tokens}`).
fn decode_usage(usage: &Value) -> Usage {
    let get = |k: &str| usage.get(k).and_then(Value::as_u64);
    Usage {
        input_tokens: get("input_tokens"),
        output_tokens: get("output_tokens"),
        total_tokens: get("total_tokens"),
        extra: Some(usage.clone()),
    }
}

// ===========================================================================
// Tests — pure JSON encode/decode against hand-written fixtures, NO live socket.
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_setup() -> RealtimeServiceSetup {
        RealtimeServiceSetup {
            model: "gpt-realtime".into(),
            system_prompt: "You are a helpful agent.".into(),
            tools: vec![ToolDecl {
                name: "transition_to_billing".into(),
                description: "Move to the billing state.".into(),
                params: json!({ "type": "object", "properties": {} }),
            }],
            input_sample_rate: 24_000,
            output_sample_rate: 24_000,
        }
    }

    fn decode(v: Value) -> (Vec<RealtimeEvent>, bool) {
        let mut terminal = false;
        let evs = decode_server_event(&v, OPENAI_SAMPLE_RATE, &mut terminal);
        (evs, terminal)
    }

    // ---- ENCODE -----------------------------------------------------------

    /// The default turn-detection JSON, for the encode fixtures below.
    fn semantic() -> Value {
        json!({ "type": "semantic_vad" })
    }

    #[test]
    fn session_update_has_instructions_audio_and_tools() {
        let v = encode_session_update(&sample_setup(), None, &semantic());
        assert_eq!(v["type"], "session.update");
        let s = &v["session"];
        assert_eq!(s["type"], "realtime");
        assert_eq!(s["instructions"], "You are a helpful agent.");
        assert_eq!(s["audio"]["input"]["format"]["type"], "audio/pcm");
        assert_eq!(s["audio"]["input"]["format"]["rate"], 24_000);
        assert_eq!(s["audio"]["output"]["format"]["rate"], 24_000);
        assert_eq!(
            s["audio"]["input"]["turn_detection"]["type"],
            "semantic_vad"
        );
        assert!(s["audio"]["input"]["transcription"].is_object());
        // No language hint → transcription auto-detects (no `language` key).
        assert!(s["audio"]["input"]["transcription"]
            .get("language")
            .is_none());
        assert_eq!(s["tool_choice"], "auto");
        let tool = &s["tools"][0];
        assert_eq!(tool["type"], "function");
        assert_eq!(tool["name"], "transition_to_billing");
        assert_eq!(
            tool["parameters"],
            json!({ "type": "object", "properties": {} })
        );
    }

    #[test]
    fn session_update_omits_tools_when_empty() {
        let mut s = sample_setup();
        s.tools.clear();
        let v = encode_session_update(&s, None, &semantic());
        assert!(v["session"].get("tools").is_none());
    }

    #[test]
    fn session_update_uses_the_given_turn_detection() {
        // The default semantic_vad rides through (asserted above); a server_vad
        // override is emitted verbatim so callers can pick endpointing behaviour.
        let sv = json!({ "type": "server_vad", "silence_duration_ms": 500 });
        let v = encode_session_update(&sample_setup(), None, &sv);
        assert_eq!(v["session"]["audio"]["input"]["turn_detection"], sv);
    }

    #[test]
    fn session_update_includes_language_hint_when_set() {
        let v = encode_session_update(&sample_setup(), Some("en"), &semantic());
        let t = &v["session"]["audio"]["input"]["transcription"];
        assert_eq!(t["language"], "en");
        // A configured language switches to whisper-1 (it honors the hint).
        assert_eq!(t["model"], "whisper-1");
    }

    #[test]
    fn session_update_auto_detects_without_language() {
        let v = encode_session_update(&sample_setup(), None, &semantic());
        let t = &v["session"]["audio"]["input"]["transcription"];
        assert!(t.get("language").is_none());
        assert_eq!(t["model"], "gpt-4o-transcribe");
    }

    #[test]
    fn audio_append_round_trips_pcm() {
        let pcm = vec![0_i16, 1, -1, 256, -256, i16::MAX, i16::MIN];
        let v = encode_audio_append(&pcm);
        assert_eq!(v["type"], "input_audio_buffer.append");
        let data = v["audio"].as_str().expect("base64 string");
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(data)
            .unwrap();
        assert_eq!(le_bytes_to_pcm(&bytes), pcm);
    }

    #[test]
    fn tool_result_creates_function_call_output_item() {
        let v = encode_tool_result("call-7", &json!({ "ok": true }));
        assert_eq!(v["type"], "conversation.item.create");
        let item = &v["item"];
        assert_eq!(item["type"], "function_call_output");
        assert_eq!(item["call_id"], "call-7");
        // `output` is a JSON *string*.
        assert_eq!(item["output"], json!({ "ok": true }).to_string());
    }

    #[test]
    fn tool_result_passes_string_output_through() {
        let v = encode_tool_result("c1", &json!("done"));
        assert_eq!(v["item"]["output"], "done");
    }

    #[test]
    fn url_appends_model_query() {
        let c = OpenAiRealtime::new("sk-test");
        assert_eq!(
            c.url("gpt-realtime"),
            "wss://api.openai.com/v1/realtime?model=gpt-realtime"
        );
        // A base that already carries a query is left intact (Azure case).
        let azure = OpenAiRealtime::new("k").with_base_url(
            "wss://x.openai.azure.com/openai/realtime?api-version=2025-04-01&deployment=d",
        );
        assert!(azure.url("ignored").contains("deployment=d"));
    }

    #[test]
    fn event_notify_is_persistent_across_calls() {
        // The s2s reader captures `event_notify()` exactly once, so it must return a
        // STABLE handle even before connect and across reconnects — a per-connection
        // notify would strand the reader after a `rebase_session` reopen.
        use flowcat_core::service::RealtimeLlmService;
        let c = OpenAiRealtime::new("sk-test");
        let a = c.event_notify().expect("a notify even before connect");
        let b = c.event_notify().expect("a notify");
        assert!(Arc::ptr_eq(&a, &b), "event_notify must be the same Arc");
    }

    #[tokio::test]
    async fn rebase_session_before_connect_is_a_clean_error() {
        // A re-base needs the stored setup to reopen with; before connect there is
        // none, so it must be a clean error, not a panic or a cryptic socket failure.
        use flowcat_core::service::RealtimeLlmService;
        let mut c = OpenAiRealtime::new("sk-test");
        let err = match c.rebase_session("digest prompt".into(), vec![]).await {
            Ok(()) => panic!("rebase_session before connect should error"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("before connect"), "got: {err}");
    }

    // ---- DECODE (hand-written server-event fixtures) ----------------------

    #[test]
    fn decode_audio_delta() {
        let pcm = vec![1_i16, -1];
        let b64 = base64::engine::general_purpose::STANDARD.encode(pcm_to_le_bytes(&pcm));
        let frame = json!({ "type": "response.output_audio.delta", "delta": b64 });
        let (evs, terminal) = decode(frame);
        assert!(!terminal);
        match &evs[0] {
            RealtimeEvent::AudioOut(c) => {
                assert_eq!(c.sample_rate, 24_000);
                assert_eq!(c.pcm, pcm);
            }
            other => panic!("expected AudioOut, got {other:?}"),
        }
    }

    #[test]
    fn decode_bot_and_user_transcripts() {
        let bot = json!({ "type": "response.output_audio_transcript.delta", "delta": "hi there" });
        assert!(matches!(&decode(bot).0[0], RealtimeEvent::BotText(t) if t == "hi there"));

        let user_delta = json!({
            "type": "conversation.item.input_audio_transcription.delta", "delta": "I want"
        });
        assert!(
            matches!(&decode(user_delta).0[0], RealtimeEvent::UserInterimText(t) if t == "I want")
        );

        let user_done = json!({
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "I want to end now"
        });
        assert!(
            matches!(&decode(user_done).0[0], RealtimeEvent::UserText(t) if t == "I want to end now")
        );
    }

    #[test]
    fn decode_function_call() {
        let frame = json!({
            "type": "response.function_call_arguments.done",
            "call_id": "fc-1",
            "name": "transition_to_billing",
            "arguments": "{\"reason\":\"user asked\"}"
        });
        let (evs, terminal) = decode(frame);
        assert!(!terminal);
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
    fn decode_function_call_blank_args_is_null() {
        let frame = json!({
            "type": "response.function_call_arguments.done",
            "call_id": "c", "name": "end_call", "arguments": ""
        });
        match &decode(frame).0[0] {
            RealtimeEvent::ToolCall { args, .. } => assert_eq!(args, &Value::Null),
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn decode_speech_started_is_interrupt() {
        let frame = json!({ "type": "input_audio_buffer.speech_started", "audio_start_ms": 12, "item_id": "x" });
        assert!(matches!(decode(frame).0[0], RealtimeEvent::Interrupted));
    }

    #[test]
    fn decode_response_done_usage() {
        let frame = json!({
            "type": "response.done",
            "response": { "usage": { "input_tokens": 100, "output_tokens": 25, "total_tokens": 125 } }
        });
        match &decode(frame).0[0] {
            RealtimeEvent::Usage(u) => {
                assert_eq!(u.input_tokens, Some(100));
                assert_eq!(u.output_tokens, Some(25));
                assert_eq!(u.total_tokens, Some(125));
                assert!(u.extra.is_some());
            }
            other => panic!("expected Usage, got {other:?}"),
        }
    }

    #[test]
    fn decode_error_is_terminal_closed() {
        let frame = json!({ "type": "error", "error": { "type": "invalid_request_error", "message": "bad" } });
        let (evs, terminal) = decode(frame);
        assert!(terminal, "error must terminate the reader");
        assert!(matches!(evs[0], RealtimeEvent::Closed));
    }

    #[test]
    fn decode_unknown_event_yields_nothing() {
        for t in [
            "session.created",
            "session.updated",
            "response.created",
            "rate_limits.updated",
        ] {
            let (evs, terminal) = decode(json!({ "type": t }));
            assert!(!terminal);
            assert!(evs.is_empty(), "{t} should map to no event");
        }
    }

    // ---- live smoke (ignored; documents the key env var) ------------------

    /// `OPENAI_API_KEY=sk-… cargo test -p flowcat-services --features realtime-openai --
    ///   realtime::openai::tests::live_openai_realtime_smoke --ignored --nocapture`
    #[tokio::test]
    #[ignore = "live: needs OPENAI_API_KEY + network (OpenAI Realtime)"]
    async fn live_openai_realtime_smoke() {
        let key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY for the live smoke");
        let mut c = OpenAiRealtime::new(key);
        c.connect(sample_setup()).await.expect("connect");
        c.kickoff().await.expect("kickoff");
        while let Some(ev) = c.next_event().await {
            if matches!(ev, RealtimeEvent::Closed) {
                break;
            }
        }
    }
}
