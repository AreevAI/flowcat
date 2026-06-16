// SPDX-License-Identifier: Apache-2.0
//
//! **AsyncAI** streaming TTS (Async `/text_to_speech/websocket/ws`).
//!
//! A **(D)istinct** streaming-WebSocket client (matched to pipecat
//! `services/asyncai/tts.py`). Connect to the fixed
//! `wss://api.async.com/text_to_speech/websocket/ws?api_key=<key>&version=v1`,
//! send a JSON **init** message once at connect (model + voice + output format),
//! then per utterance a **single** `transcript` message with `force: true` — that
//! forced message carries the text AsyncAI synthesizes. It is **not** a
//! buffer-then-whitespace-flush provider like Cartesia: a forced empty/whitespace
//! transcript yields empty audio.
//!
//! ```json
//! // init (sent once at connect):
//! { "model_id": "async_flash_v1.0", "voice": { "mode": "id", "id": "<voice>" },
//!   "output_format": { "container": "raw", "encoding": "pcm_s16le", "sample_rate": 24000 },
//!   "language": null }
//! // per utterance (one message: the full text, force = true):
//! { "transcript": "hello there", "context_id": "ctx-1", "force": true }
//! ```
//!
//! Server messages are `{ "audio": "<base64 pcm>", "context_id": "…" }` chunks
//! followed by a terminal `{ "final": true, "context_id": "…" }`. Base64 PCM
//! → [`Frame::TtsAudio`]; `final` ends the run. AsyncAI WS does not emit word
//! timestamps. The API key is a **required query param** for this provider's
//! handshake (it has no header-auth form) — the host is still fixed, so there is
//! no request-derived URL. AsyncAI binds one synthesis context to the socket and
//! closes it after `final`, so this connector **reconnects per utterance**.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message;

use flowcat_core::error::{FlowcatError, Result};
use flowcat_core::processor::frame::{AudioFrame, Frame, StartParams};
use flowcat_core::service::TtsService;

#[allow(clippy::duplicate_mod)] // each WS provider owns its own copy (feature-independent)
#[path = "ws_tts_common.rs"]
pub mod ws_tts;

use ws_tts::Decoded;

/// The live AsyncAI WebSocket (TLS).
type ClientSocket =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// Idle gap after the last audio chunk that marks "synthesis complete". Must
/// exceed AsyncAI's largest mid-stream inter-chunk gap (~1s observed) — AsyncAI
/// never emits `final` on its own, and a `close_context` sent before synthesis
/// finishes truncates the audio, so we wait out this gap before closing.
const ASYNCAI_IDLE_DONE: Duration = Duration::from_millis(1500);
/// Wait budget for the one-time `init_ack` handshake / first audio chunk.
const ASYNCAI_ACK_TIMEOUT: Duration = Duration::from_secs(10);
/// Overall safety cap on one utterance's synthesis.
const ASYNCAI_MAX_WAIT: Duration = Duration::from_secs(30);

/// AsyncAI's TTS WebSocket host. The `api_key` + `version` query is appended at
/// connect time (the provider requires the key as a query param).
pub const ASYNCAI_WSS_BASE: &str = "wss://api.async.com/text_to_speech/websocket/ws";
/// The Async API version this client speaks.
pub const ASYNCAI_VERSION: &str = "v1";

/// AsyncAI streaming-TTS session.
pub struct AsyncAiTts {
    api_key: String,
    voice_id: String,
    model: String,
    sample_rate: u32,
    /// A pre-warmed socket (init already sent) for the next utterance. AsyncAI
    /// binds one synthesis context per socket and closes it after `final`, so it
    /// is consumed and re-established each turn.
    session: Option<ClientSocket>,
}

impl AsyncAiTts {
    /// Construct bound to `api_key` + `voice_id` (default `async_flash_v1.0`
    /// model, 24 kHz raw PCM output).
    pub fn new(api_key: impl Into<String>, voice_id: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            voice_id: voice_id.into(),
            model: "async_flash_v1.0".to_string(),
            sample_rate: 24_000,
            session: None,
        }
    }

    /// Override the model (default `async_flash_v1.0`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the output sample rate (default 24 kHz).
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    fn url(&self) -> String {
        format!(
            "{ASYNCAI_WSS_BASE}?api_key={}&version={ASYNCAI_VERSION}",
            self.api_key
        )
    }

    /// The init message sent once at connect (pure — the wire-fixture seam).
    fn init_message(&self) -> Value {
        json!({
            "model_id": self.model,
            "voice": { "mode": "id", "id": self.voice_id },
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": self.sample_rate,
            },
            "language": null,
        })
    }

    /// Open the WS and send the init handshake (no read yet).
    async fn connect_and_init(&self) -> Result<ClientSocket> {
        let request = self
            .url()
            .into_client_request()
            .map_err(|e| FlowcatError::Network(format!("asyncai url: {e}")))?;
        let (mut socket, _resp) = tokio_tungstenite::connect_async(request)
            .await
            .map_err(|e| FlowcatError::Network(format!("asyncai connect: {e}")))?;
        socket
            .send(Message::text(self.init_message().to_string()))
            .await
            .map_err(|e| FlowcatError::Network(format!("asyncai init send: {e}")))?;
        Ok(socket)
    }

    /// Synthesize one utterance over `socket` (init already sent): read `init_ack`
    /// for the server-minted context id, send the forced transcript on it, collect
    /// audio until it quiesces, then `close_context` to draw the terminal `final`.
    /// AsyncAI does NOT emit `final` on its own after synthesis, and a client-chosen
    /// context id is ignored (→ empty audio), so both steps are mandatory.
    async fn synthesize_one(&self, socket: &mut ClientSocket, text: &str) -> Result<Vec<Frame>> {
        let rate = self.sample_rate;

        // 1. The server mints the synthesis context at init and reports it here.
        let server_ctx = read_init_ack(socket).await?;
        let ctx: Arc<str> = Arc::from(server_ctx.as_str());

        // 2. One forced message carrying the full text, on the server's context.
        socket
            .send(Message::text(speak_message(text, &server_ctx).to_string()))
            .await
            .map_err(|e| FlowcatError::Network(format!("asyncai speak send: {e}")))?;

        // 3. Collect audio until it quiesces, then close the context for `final`.
        let mut out = vec![Frame::TtsStarted {
            context_id: Some(ctx.clone()),
        }];
        let mut got_audio = false;
        let mut closed_sent = false;
        let deadline = tokio::time::Instant::now() + ASYNCAI_MAX_WAIT;

        loop {
            if tokio::time::Instant::now() >= deadline {
                break;
            }
            // Long wait for the first chunk; once audio flows, a short idle gap
            // means synthesis is done and it is safe to close the context.
            let idle = if got_audio {
                ASYNCAI_IDLE_DONE
            } else {
                ASYNCAI_ACK_TIMEOUT
            };
            match tokio::time::timeout(idle, socket.next()).await {
                Err(_) => {
                    if got_audio && !closed_sent {
                        socket
                            .send(Message::text(close_message(&server_ctx).to_string()))
                            .await
                            .map_err(|e| {
                                FlowcatError::Network(format!("asyncai close send: {e}"))
                            })?;
                        closed_sent = true;
                        continue;
                    }
                    break;
                }
                Ok(None) => break, // socket closed
                Ok(Some(Ok(Message::Text(t)))) => {
                    let Ok(v) = serde_json::from_str::<Value>(&t) else {
                        continue;
                    };
                    match decode_message(Some(&v), None) {
                        Decoded::Audio(pcm) => {
                            if !pcm.is_empty() {
                                got_audio = true;
                                out.push(Frame::TtsAudio {
                                    audio: Arc::new(AudioFrame::mono(pcm, rate)),
                                    context_id: Some(ctx.clone()),
                                });
                            }
                        }
                        Decoded::Done => break,
                        _ => {}
                    }
                }
                Ok(Some(Ok(Message::Close(_)))) => break,
                Ok(Some(Err(_))) => break,
                Ok(Some(Ok(_))) => {} // ping / pong / binary — ignore
            }
        }

        out.push(Frame::TtsStopped {
            context_id: Some(ctx),
        });
        Ok(out)
    }
}

#[async_trait]
impl TtsService for AsyncAiTts {
    fn name(&self) -> &str {
        "asyncai"
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        // Pre-warm a socket (with init sent) for the first utterance.
        self.session = Some(self.connect_and_init().await?);
        Ok(())
    }

    async fn run_tts(&mut self, text: &str) -> Result<Vec<Frame>> {
        // Use the pre-warmed socket if present, else connect fresh. AsyncAI binds
        // one context per socket and closes it after `final`, so each utterance
        // takes (and then drops) its own connection.
        let mut socket = match self.session.take() {
            Some(s) => s,
            None => self.connect_and_init().await?,
        };
        let frames = self.synthesize_one(&mut socket, text).await;
        drop(socket); // server closes after `final`; the next turn reconnects
        frames
    }
}

/// Read server messages until the one-time `init_ack`, returning its `context_id`
/// — the id AsyncAI binds the synthesis context to (transcripts must use it).
async fn read_init_ack(socket: &mut ClientSocket) -> Result<String> {
    let read = async {
        while let Some(msg) = socket.next().await {
            let Ok(Message::Text(t)) = msg else { continue };
            let Ok(v) = serde_json::from_str::<Value>(&t) else {
                continue;
            };
            if v.get("event").and_then(|e| e.as_str()) == Some("init_ack") {
                if let Some(c) = v.get("context_id").and_then(|c| c.as_str()) {
                    return Ok(c.to_string());
                }
            }
        }
        Err(FlowcatError::Network(
            "asyncai: socket closed before init_ack".to_string(),
        ))
    };
    tokio::time::timeout(ASYNCAI_ACK_TIMEOUT, read)
        .await
        .map_err(|_| FlowcatError::Network("asyncai: init_ack timeout".to_string()))?
}

/// The single forced-synthesis message for one utterance (pure — the wire-fixture
/// seam). AsyncAI synthesizes the transcript of the `force: true` message, so the
/// full text goes here in one shot (no buffer/whitespace-flush split).
fn speak_message(text: &str, context_id: &str) -> Value {
    json!({ "transcript": text, "context_id": context_id, "force": true })
}

/// The context-close message that draws the terminal `final` (pure). Sent only
/// after audio quiesces — sent early, it truncates or empties the synthesis.
fn close_message(context_id: &str) -> Value {
    json!({ "context_id": context_id, "close_context": true, "transcript": "" })
}

/// Decode one AsyncAI server message (pure — the wire-fixture seam). An `audio`
/// field → PCM; `final: true` ends the run; binary / anything else is ignored.
pub(crate) fn decode_message(json: Option<&Value>, _binary: Option<&[u8]>) -> Decoded {
    let Some(value) = json else {
        return Decoded::Ignore;
    };
    if value.get("final").and_then(|f| f.as_bool()) == Some(true) {
        return Decoded::Done;
    }
    if value.get("audio").and_then(|a| a.as_str()).is_some() {
        return Decoded::Audio(ws_tts::pcm_from_b64_field(value, "audio"));
    }
    Decoded::Ignore
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine as _;

    #[test]
    fn url_uses_the_fixed_host_with_version() {
        let t = AsyncAiTts::new("k", "voice-x");
        let url = t.url();
        assert!(url.starts_with("wss://api.async.com/text_to_speech/websocket/ws?"));
        assert!(url.contains("version=v1"));
        // This provider requires the key as a query param.
        assert!(url.contains("api_key=k"));
    }

    #[test]
    fn init_carries_voice_and_output_format() {
        let init = AsyncAiTts::new("k", "voice-x")
            .sample_rate(16_000)
            .init_message();
        assert_eq!(init["model_id"], "async_flash_v1.0");
        assert_eq!(init["voice"]["mode"], "id");
        assert_eq!(init["voice"]["id"], "voice-x");
        assert_eq!(init["output_format"]["container"], "raw");
        assert_eq!(init["output_format"]["encoding"], "pcm_s16le");
        assert_eq!(init["output_format"]["sample_rate"], 16_000);
        assert!(init["language"].is_null());
    }

    #[test]
    fn speak_message_is_single_forced_transcript() {
        // One forced message carrying the full text on the server's context id —
        // NOT a buffer + whitespace flush, and never a client-chosen context id.
        let m = speak_message("hello there", "srv-ctx");
        assert_eq!(m["transcript"], "hello there");
        assert_eq!(m["context_id"], "srv-ctx");
        assert_eq!(m["force"], true);
    }

    #[test]
    fn close_message_closes_the_named_context() {
        let m = close_message("srv-ctx");
        assert_eq!(m["context_id"], "srv-ctx");
        assert_eq!(m["close_context"], true);
        assert_eq!(m["transcript"], "");
    }

    #[test]
    fn decode_audio_into_pcm() {
        let b64 = base64::engine::general_purpose::STANDARD.encode([1u8, 0, 255, 255]);
        let msg = json!({ "audio": b64, "context_id": "ctx-1" });
        match decode_message(Some(&msg), None) {
            Decoded::Audio(pcm) => assert_eq!(pcm, vec![1, -1]),
            _ => panic!("expected Audio"),
        }
    }

    #[test]
    fn decode_final_and_ignore() {
        assert!(matches!(
            decode_message(Some(&json!({ "final": true, "context_id": "ctx-1" })), None),
            Decoded::Done
        ));
        // No audio / final → ignore (no panic).
        assert!(matches!(
            decode_message(Some(&json!({ "status": "ok" })), None),
            Decoded::Ignore
        ));
        assert!(matches!(
            decode_message(Some(&json!("nope")), None),
            Decoded::Ignore
        ));
    }

    /// Live smoke (requires `ASYNCAI_API_KEY` + `ASYNCAI_VOICE_ID`). Run:
    /// `ASYNCAI_API_KEY=… ASYNCAI_VOICE_ID=… cargo test -p flowcat-services --features tts-asyncai -- --ignored asyncai_live`
    #[tokio::test]
    #[ignore = "requires ASYNCAI_API_KEY + ASYNCAI_VOICE_ID"]
    async fn asyncai_live_synthesizes_audio() {
        // rustls 0.23 needs a CryptoProvider selected before the TLS WS handshake
        // (the vaais-media binary installs this at startup; the smoke test must too).
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let key = std::env::var("ASYNCAI_API_KEY").expect("ASYNCAI_API_KEY");
        let voice = std::env::var("ASYNCAI_VOICE_ID").expect("ASYNCAI_VOICE_ID");
        let mut tts = AsyncAiTts::new(key, voice);
        tts.start(&StartParams::default()).await.expect("connect");
        let frames = tts
            .run_tts("Hello there, this is a test of the AsyncAI voice connector.")
            .await
            .expect("run_tts");
        // Sum the actual PCM produced — guards against the empty-audio regression
        // (wrong model / client context id / early close_context all yield 0).
        let samples: usize = frames
            .iter()
            .filter_map(|f| match f {
                Frame::TtsAudio { audio, .. } => Some(audio.pcm.len()),
                _ => None,
            })
            .sum();
        // ~3.5s of 24 kHz mono speech → tens of thousands of samples; require a
        // clearly-non-empty floor (0.5s) so a truncated/empty synthesis fails.
        assert!(
            samples > 12_000,
            "expected real audio (>0.5s), got {samples} samples"
        );
        // And a second utterance must work too (per-utterance reconnect path).
        let frames2 = tts.run_tts("Second turn.").await.expect("run_tts 2");
        assert!(
            frames2.iter().any(|f| matches!(f, Frame::TtsAudio { .. })),
            "expected audio on the second utterance"
        );
    }
}
