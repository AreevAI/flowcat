// SPDX-License-Identifier: Apache-2.0
//
//! **Ultravox Realtime** (speech-to-speech) client — fixture-skeleton.
//!
//! Ultravox uses a **distinct** wire protocol from OpenAI Realtime: the call is
//! created via the Ultravox REST API (`POST /api/agents/{id}/calls` or
//! `/api/calls`), which returns a `joinUrl`; the client then connects a plain
//! WebSocket to that join URL and exchanges:
//!
//! - **client→server:** raw **binary PCM** frames (caller audio, at the
//!   negotiated rate — pipecat uses 48 kHz), plus JSON control messages
//!   `{"type":"user_text_message","text":…}`, `{"type":"client_tool_result",
//!   "invocationId":…,"result":…}`, `{"type":"set_output_medium","medium":"voice"}`.
//! - **server→client:** raw **binary PCM** frames (bot audio), and JSON
//!   `{"type":"state","state":"speaking"|…}` (a transition away from `speaking`
//!   ends the bot turn), `{"type":"client_tool_invocation","toolName":…,
//!   "invocationId":…,"parameters":…}` (a tool call), `{"type":"transcript",
//!   "role":"user"|"agent","text":…,"delta":…,"final":…}`.
//!
//! Cross-checked against `pipecat/src/pipecat/services/ultravox/llm.py`.
//!
//! **Fixture-skeleton scope:** the wire **encode/decode** of every key message is
//! implemented + tested. The **join-URL acquisition** (the REST call that mints a
//! session) and the live WS transport are a follow-up (they need an Ultravox key
//! + network) — [`UltravoxRealtime::connect`] expects the join URL to be supplied
//! out-of-band via the setup (`RealtimeServiceSetup::model` carries the `joinUrl`)
//! and the WS plumbing is wired but only exercised by the `#[ignore]` live smoke.
//!
//! ## Keys / auth (security note)
//!
//! The Ultravox **API key** authenticates the REST call that issues the join URL;
//! the join URL itself is a **capability URL** (a bearer secret in the path) — it
//! must be treated as a credential (never logged). This skeleton holds the join
//! URL on the struct and uses it only at connect.

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{SplitSink, SplitStream};
use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

use flowcat_core::error::FlowcatError;
use flowcat_core::processor::frame::AudioFrame;
use flowcat_core::service::{RealtimeLlmService, RealtimeServiceSetup, Tool};
use flowcat_core::types::{AudioChunk, RealtimeEvent};

/// Ultravox Realtime default PCM sample rate (mono 16-bit), both directions.
const ULTRAVOX_SAMPLE_RATE: u32 = 48_000;

type ClientSocket = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;
type WsSink = SplitSink<ClientSocket, Message>;
type WsStream = SplitStream<ClientSocket>;

struct Connection {
    sink: Arc<Mutex<WsSink>>,
    events: mpsc::UnboundedReceiver<RealtimeEvent>,
    reader: JoinHandle<()>,
}

impl Connection {
    async fn send_json(&self, value: &Value) -> Result<(), FlowcatError> {
        let mut sink = self.sink.lock().await;
        sink.send(Message::text(serde_json::to_string(value)?))
            .await
            .map_err(|e| FlowcatError::Realtime(format!("ws send: {e}")))
    }

    async fn send_binary(&self, bytes: Vec<u8>) -> Result<(), FlowcatError> {
        let mut sink = self.sink.lock().await;
        sink.send(Message::binary(bytes))
            .await
            .map_err(|e| FlowcatError::Realtime(format!("ws send audio: {e}")))
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        self.reader.abort();
    }
}

/// An Ultravox Realtime session.
pub struct UltravoxRealtime {
    /// The capability join URL (a bearer secret — never logged). Empty until set.
    join_url: String,
    /// Output medium for the agent (`voice` for a phone call).
    output_medium: String,
    sample_rate: u32,
    conn: Option<Connection>,
}

impl UltravoxRealtime {
    /// Construct a client from a pre-issued Ultravox **join URL** (minted by the
    /// Ultravox REST API). The join URL is a capability secret; keep it out of
    /// logs. Defaults to `voice` output @ 48 kHz.
    pub fn new(join_url: impl Into<String>) -> Self {
        Self {
            join_url: join_url.into(),
            output_medium: "voice".to_string(),
            sample_rate: ULTRAVOX_SAMPLE_RATE,
            conn: None,
        }
    }

    /// Override the PCM sample rate negotiated with Ultravox (default 48 kHz).
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Inject a typed user turn (`user_text_message`) — useful for text-mode or
    /// a seeded first turn. Not part of the [`RealtimeLlmService`] trait (which
    /// is audio-driven), so it is an inherent method.
    pub async fn send_user_text(&mut self, text: &str) -> Result<(), FlowcatError> {
        self.require_conn()?
            .send_json(&encode_user_text(text))
            .await
    }

    fn require_conn(&self) -> Result<&Connection, FlowcatError> {
        self.conn
            .as_ref()
            .ok_or_else(|| FlowcatError::Realtime("not connected".into()))
    }
}

#[async_trait]
impl RealtimeLlmService for UltravoxRealtime {
    async fn connect(&mut self, setup: RealtimeServiceSetup) -> Result<(), FlowcatError> {
        // The join URL is supplied at construction; for parity with the other
        // providers, `setup.model` may also carry it (it is not a model name for
        // Ultravox). Prefer an explicit non-empty `model` join URL if present.
        if !setup.model.is_empty() && setup.model.starts_with("ws") {
            self.join_url = setup.model.clone();
        }
        if self.join_url.is_empty() {
            return Err(FlowcatError::Realtime(
                "ultravox: no join URL (mint one via the Ultravox REST API first)".into(),
            ));
        }

        let (socket, _resp) = connect_async(&self.join_url)
            .await
            .map_err(|e| FlowcatError::Realtime(format!("ultravox connect: {e}")))?;
        let (mut sink, stream) = socket.split();

        // Set the output medium up-front (voice for a phone call).
        sink.send(Message::text(serde_json::to_string(
            &encode_set_output_medium(&self.output_medium),
        )?))
        .await
        .map_err(|e| FlowcatError::Realtime(format!("ultravox set medium: {e}")))?;

        let (tx, rx) = mpsc::unbounded_channel();
        let reader = tokio::spawn(reader_task(stream, tx, self.sample_rate));
        self.conn = Some(Connection {
            sink: Arc::new(Mutex::new(sink)),
            events: rx,
            reader,
        });
        Ok(())
    }

    async fn send_audio(&mut self, chunk: Arc<AudioFrame>) -> Result<(), FlowcatError> {
        // Ultravox takes raw binary PCM frames (no JSON envelope).
        let bytes = pcm_to_le_bytes(&chunk.pcm);
        self.require_conn()?.send_binary(bytes).await
    }

    async fn update_system(
        &mut self,
        _prompt: String,
        _tools: Vec<Tool>,
    ) -> Result<(), FlowcatError> {
        // Ultravox binds the system prompt + tools to the *agent/call* at REST
        // creation time; there is no in-session system update over the WS. A
        // brain transition is expressed by minting a new call (REST) — out of
        // scope for this skeleton. No-op so a transition does not error the call.
        tracing::debug!("ultravox: update_system is a no-op (prompt/tools bound at call creation)");
        Ok(())
    }

    async fn send_tool_result(&mut self, id: String, result: Value) -> Result<(), FlowcatError> {
        self.require_conn()?
            .send_json(&encode_tool_result(&id, &result))
            .await
    }

    async fn next_event(&mut self) -> Option<RealtimeEvent> {
        self.conn.as_mut()?.events.recv().await
    }

    /// Ultravox negotiates PCM at 48 kHz by default (override via `with_sample_rate`).
    fn input_sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

// ---------------------------------------------------------------------------
// Encoders (client → server).
// ---------------------------------------------------------------------------

/// `{"type":"set_output_medium","medium":"voice"}`.
fn encode_set_output_medium(medium: &str) -> Value {
    json!({ "type": "set_output_medium", "medium": medium })
}

/// `{"type":"user_text_message","text":…}` — inject a typed user turn.
pub(crate) fn encode_user_text(text: &str) -> Value {
    json!({ "type": "user_text_message", "text": text })
}

/// `{"type":"client_tool_result","invocationId":id,"result":<stringified>}`.
fn encode_tool_result(invocation_id: &str, result: &Value) -> Value {
    let result_str = match result {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    json!({
        "type": "client_tool_result",
        "invocationId": invocation_id,
        "result": result_str
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

async fn reader_task(mut stream: WsStream, tx: mpsc::UnboundedSender<RealtimeEvent>, rate: u32) {
    // Track whether the bot is mid-turn, so a `state != speaking` transition is
    // only surfaced once (mirrors pipecat's `_bot_responding` gate).
    let mut bot_speaking = false;
    while let Some(msg) = stream.next().await {
        match msg {
            // Binary = bot PCM audio.
            Ok(Message::Binary(b)) => {
                if !b.is_empty() {
                    bot_speaking = true;
                    if tx
                        .send(RealtimeEvent::AudioOut(AudioChunk::new(
                            le_bytes_to_pcm(&b),
                            rate,
                        )))
                        .is_err()
                    {
                        return;
                    }
                }
            }
            Ok(Message::Text(t)) => {
                let value: Value = match serde_json::from_str(t.as_str()) {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::warn!("ultravox: undecodable frame: {e}");
                        continue;
                    }
                };
                for ev in decode_server_message(&value, &mut bot_speaking) {
                    if tx.send(ev).is_err() {
                        return;
                    }
                }
            }
            Ok(Message::Close(_)) => {
                let _ = tx.send(RealtimeEvent::Closed);
                return;
            }
            Ok(_) => {}
            Err(e) => {
                tracing::warn!("ultravox read error: {e}");
                let _ = tx.send(RealtimeEvent::Closed);
                return;
            }
        }
    }
    let _ = tx.send(RealtimeEvent::Closed);
}

/// Map one Ultravox JSON control message into zero or more [`RealtimeEvent`]s.
///
/// - `state` (transition away from `speaking` while a bot turn was active) →
///   nothing surfaced as a discrete event today (the turn end is implicit); we
///   only flip the `bot_speaking` gate. A barge-in is not a distinct Ultravox
///   message — the model just stops emitting audio.
/// - `client_tool_invocation{toolName,invocationId,parameters}` → `ToolCall`
/// - `transcript{role:"user",text,final}` → `UserText` (final only)
/// - `transcript{role:"agent",text|delta}` → `BotText`
pub(crate) fn decode_server_message(value: &Value, bot_speaking: &mut bool) -> Vec<RealtimeEvent> {
    let mut out = Vec::new();
    match value
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default()
    {
        "state" => {
            let speaking = value.get("state").and_then(Value::as_str) == Some("speaking");
            if *bot_speaking && !speaking {
                // Bot finished its turn. No discrete `RealtimeEvent` for this in
                // the frozen enum; the gate flips so the next audio re-arms it.
                *bot_speaking = false;
            } else if speaking {
                *bot_speaking = true;
            }
        }
        "client_tool_invocation" => {
            let name = value
                .get("toolName")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let id = value
                .get("invocationId")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let args = value.get("parameters").cloned().unwrap_or(Value::Null);
            out.push(RealtimeEvent::ToolCall { name, args, id });
        }
        "transcript" => match value.get("role").and_then(Value::as_str) {
            Some("user") => {
                // Only surface *final* user transcripts (interim are noisy).
                if value.get("final").and_then(Value::as_bool) == Some(true) {
                    if let Some(text) = nonempty(value, "text") {
                        out.push(RealtimeEvent::UserText(text));
                    }
                }
            }
            Some("agent") => {
                // Prefer the incremental `delta`; fall back to the full `text`.
                if let Some(text) = nonempty(value, "delta").or_else(|| nonempty(value, "text")) {
                    out.push(RealtimeEvent::BotText(text));
                }
            }
            _ => {}
        },
        _ => {}
    }
    out
}

/// A non-empty string field, owned.
fn nonempty(value: &Value, field: &str) -> Option<String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
}

// ===========================================================================
// Tests — pure encode/decode against hand-written fixtures, NO live socket.
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(v: Value) -> Vec<RealtimeEvent> {
        let mut speaking = false;
        decode_server_message(&v, &mut speaking)
    }

    // ---- ENCODE -----------------------------------------------------------

    #[test]
    fn set_output_medium_message() {
        assert_eq!(
            encode_set_output_medium("voice"),
            json!({ "type": "set_output_medium", "medium": "voice" })
        );
    }

    #[test]
    fn user_text_message() {
        assert_eq!(
            encode_user_text("hello"),
            json!({ "type": "user_text_message", "text": "hello" })
        );
    }

    #[test]
    fn tool_result_message_stringifies_json() {
        let v = encode_tool_result("inv-1", &json!({ "ok": true }));
        assert_eq!(v["type"], "client_tool_result");
        assert_eq!(v["invocationId"], "inv-1");
        assert_eq!(v["result"], json!({ "ok": true }).to_string());
        // A plain string result passes through unquoted.
        assert_eq!(encode_tool_result("i", &json!("done"))["result"], "done");
    }

    #[test]
    fn audio_pcm_round_trips_through_le_bytes() {
        let pcm = vec![0_i16, 7, -7, i16::MAX, i16::MIN];
        assert_eq!(le_bytes_to_pcm(&pcm_to_le_bytes(&pcm)), pcm);
    }

    // ---- DECODE -----------------------------------------------------------

    #[test]
    fn decode_tool_invocation() {
        let frame = json!({
            "type": "client_tool_invocation",
            "toolName": "transition_to_billing",
            "invocationId": "inv-9",
            "parameters": { "reason": "user asked" }
        });
        match &decode(frame)[0] {
            RealtimeEvent::ToolCall { id, name, args } => {
                assert_eq!(id, "inv-9");
                assert_eq!(name, "transition_to_billing");
                assert_eq!(args, &json!({ "reason": "user asked" }));
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn decode_user_transcript_final_only() {
        let interim =
            json!({ "type": "transcript", "role": "user", "text": "I wa", "final": false });
        assert!(
            decode(interim).is_empty(),
            "interim user transcript is dropped"
        );

        let fin =
            json!({ "type": "transcript", "role": "user", "text": "I want to end", "final": true });
        assert!(matches!(&decode(fin)[0], RealtimeEvent::UserText(t) if t == "I want to end"));
    }

    #[test]
    fn decode_agent_transcript_prefers_delta() {
        let delta =
            json!({ "type": "transcript", "role": "agent", "delta": "hel", "text": "hello" });
        assert!(matches!(&decode(delta)[0], RealtimeEvent::BotText(t) if t == "hel"));

        let full = json!({ "type": "transcript", "role": "agent", "text": "hello" });
        assert!(matches!(&decode(full)[0], RealtimeEvent::BotText(t) if t == "hello"));
    }

    #[test]
    fn decode_state_toggles_speaking_gate_without_event() {
        let mut speaking = true;
        let evs =
            decode_server_message(&json!({ "type": "state", "state": "idle" }), &mut speaking);
        assert!(evs.is_empty(), "a state change is not a discrete event");
        assert!(!speaking, "leaving 'speaking' clears the gate");

        let mut speaking = false;
        decode_server_message(
            &json!({ "type": "state", "state": "speaking" }),
            &mut speaking,
        );
        assert!(speaking, "entering 'speaking' sets the gate");
    }

    #[test]
    fn decode_unknown_message_yields_nothing() {
        assert!(decode(json!({ "type": "debug", "x": 1 })).is_empty());
    }

    // ---- live smoke (ignored; documents the env var) ----------------------

    /// `ULTRAVOX_JOIN_URL=wss://… cargo test -p flowcat-services \
    ///   --features realtime-ultravox -- \
    ///   realtime::ultravox::tests::live_ultravox_realtime_smoke --ignored --nocapture`
    ///
    /// The join URL is minted by the Ultravox REST API
    /// (`POST https://api.ultravox.ai/api/calls` with your `X-API-Key`).
    #[tokio::test]
    #[ignore = "live: needs ULTRAVOX_JOIN_URL (mint via the Ultravox REST API)"]
    async fn live_ultravox_realtime_smoke() {
        let join_url = std::env::var("ULTRAVOX_JOIN_URL").expect("ULTRAVOX_JOIN_URL");
        let mut c = UltravoxRealtime::new(join_url);
        c.connect(RealtimeServiceSetup {
            model: String::new(),
            system_prompt: String::new(), // bound at call creation
            tools: vec![],
            input_sample_rate: ULTRAVOX_SAMPLE_RATE,
            output_sample_rate: ULTRAVOX_SAMPLE_RATE,
        })
        .await
        .expect("connect");
        while let Some(ev) = c.next_event().await {
            if matches!(ev, RealtimeEvent::Closed) {
                break;
            }
        }
    }
}
