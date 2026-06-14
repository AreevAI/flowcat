// SPDX-License-Identifier: Apache-2.0
//
//! Cartesia streaming TTS over the `/tts/websocket` endpoint.
//!
//! Wire protocol (cross-checked against pipecat `services/cartesia/tts.py`):
//! connect to
//! `wss://api.cartesia.ai/tts/websocket?api_key=<key>&cartesia_version=2026-03-01`,
//! send a JSON synthesis request per utterance:
//!
//! ```json
//! { "transcript": "hello", "model_id": "sonic-2", "context_id": "…",
//!   "voice": { "mode": "id", "id": "<voice-id>" },
//!   "output_format": { "container": "raw", "encoding": "pcm_s16le", "sample_rate": 24000 },
//!   "continue": false }
//! ```
//!
//! and read JSON responses: `{ "type": "chunk", "data": "<base64 pcm>", "context_id": … }`
//! frames followed by `{ "type": "done", "context_id": … }`. Base64 PCM is
//! decoded into [`Frame::TtsAudio`]; the run is framed by [`Frame::TtsStarted`] /
//! [`Frame::TtsStopped`]. The request encode + audio decode are **pure functions**
//! ([`build_request`], [`decode_message`]) so the wire format is unit-tested
//! without a socket.

use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine as _;
use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio_tungstenite::tungstenite::Message;

use flowcat_core::error::{FlowcatError, Result};
use flowcat_core::processor::frame::{AudioFrame, Frame, StartParams};
use flowcat_core::service::TtsService;

/// Cartesia's TTS WebSocket base. The `api_key` + `cartesia_version` query is
/// appended at connect time. The **host is fixed** — no request-derived URL.
pub const CARTESIA_WSS_BASE: &str = "wss://api.cartesia.ai/tts/websocket";
/// The Cartesia API version this client speaks.
pub const CARTESIA_VERSION: &str = "2026-03-01";

/// Builder for [`CartesiaTts`].
#[derive(Debug, Clone)]
pub struct CartesiaTtsBuilder {
    api_key: String,
    model: String,
    voice_id: String,
    sample_rate: u32,
}

impl CartesiaTtsBuilder {
    /// Start a builder bound to `api_key` + `voice_id` (default model sonic-2,
    /// 24 kHz raw PCM).
    pub fn new(api_key: impl Into<String>, voice_id: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: "sonic-2".to_string(),
            voice_id: voice_id.into(),
            sample_rate: 24_000,
        }
    }

    /// Override the model (default `sonic-2`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the output sample rate (default 24 kHz).
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Build the (not-yet-connected) client.
    pub fn build(self) -> CartesiaTts {
        CartesiaTts {
            api_key: self.api_key,
            model: self.model,
            voice_id: self.voice_id,
            sample_rate: self.sample_rate,
            conn: None,
            ctx_counter: 0,
        }
    }
}

type ClientSocket =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// A native Cartesia streaming-TTS session.
pub struct CartesiaTts {
    api_key: String,
    model: String,
    voice_id: String,
    sample_rate: u32,
    conn: Option<ClientSocket>,
    ctx_counter: u64,
}

impl CartesiaTts {
    /// Construct with defaults. Use [`CartesiaTtsBuilder`] for non-defaults.
    pub fn new(api_key: impl Into<String>, voice_id: impl Into<String>) -> Self {
        CartesiaTtsBuilder::new(api_key, voice_id).build()
    }

    fn url(&self) -> String {
        format!(
            "{CARTESIA_WSS_BASE}?api_key={}&cartesia_version={CARTESIA_VERSION}",
            self.api_key
        )
    }

    /// Build the synthesis request body for `text` + `context_id` (pure).
    fn build_request(&self, text: &str, context_id: &str) -> Value {
        build_request(
            text,
            context_id,
            &self.model,
            &self.voice_id,
            self.sample_rate,
        )
    }

    async fn open(&mut self) -> Result<()> {
        let (socket, _resp) = tokio_tungstenite::connect_async(self.url())
            .await
            .map_err(|e| FlowcatError::Network(format!("cartesia connect: {e}")))?;
        self.conn = Some(socket);
        Ok(())
    }
}

#[async_trait]
impl TtsService for CartesiaTts {
    fn name(&self) -> &str {
        "cartesia"
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        // Lazy connect: do NOT open the WebSocket here. `start()` runs inside the
        // pipeline Start handshake (every processor's `start()` must complete
        // before any audio — or the greeting `ClientConnected` — is pumped), so a
        // slow/hung/failed Cartesia connect would stall the whole call. Open on
        // the first `run_tts` instead, where a connect error is a visible run
        // error rather than a silent handshake hang.
        Ok(())
    }

    async fn run_tts(&mut self, text: &str) -> Result<Vec<Frame>> {
        self.ctx_counter += 1;
        let context_id: Arc<str> = Arc::from(format!("ctx-{}", self.ctx_counter));
        let rate = self.sample_rate;
        let request = self.build_request(text, &context_id);
        // Lazy connect (see `start`): open the WS on the first synthesis.
        if self.conn.is_none() {
            self.open().await?;
        }
        let socket = self
            .conn
            .as_mut()
            .ok_or_else(|| FlowcatError::Network("cartesia: not connected".into()))?;

        // Send the synthesis request.
        socket
            .send(Message::text(
                serde_json::to_string(&request).map_err(FlowcatError::from)?,
            ))
            .await
            .map_err(|e| FlowcatError::Network(format!("cartesia send: {e}")))?;

        // Frame the run; read chunks until `done` for this context.
        let mut out = vec![Frame::TtsStarted {
            context_id: Some(context_id.clone()),
        }];
        while let Some(msg) = socket.next().await {
            let text = match msg {
                Ok(Message::Text(t)) => t.to_string(),
                Ok(Message::Binary(_)) | Ok(Message::Ping(_)) | Ok(Message::Pong(_)) => continue,
                Ok(Message::Close(_)) | Err(_) => break,
                Ok(_) => continue,
            };
            let Ok(value) = serde_json::from_str::<Value>(&text) else {
                continue;
            };
            match decode_message(&value, rate, &context_id) {
                Decoded::Audio(frame) => out.push(frame),
                Decoded::Done => break,
                Decoded::Error(e) => return Err(FlowcatError::Network(format!("cartesia: {e}"))),
                Decoded::Ignore => {}
            }
        }
        out.push(Frame::TtsStopped {
            context_id: Some(context_id),
        });
        Ok(out)
    }
}

/// Build the Cartesia synthesis request body (pure — the wire-fixture seam).
fn build_request(
    text: &str,
    context_id: &str,
    model: &str,
    voice_id: &str,
    sample_rate: u32,
) -> Value {
    json!({
        "transcript": text,
        "model_id": model,
        "context_id": context_id,
        "continue": false,
        "voice": { "mode": "id", "id": voice_id },
        "output_format": {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": sample_rate,
        },
    })
}

/// The decode of one Cartesia server message.
enum Decoded {
    /// A decoded audio chunk → `TtsAudio`.
    Audio(Frame),
    /// The terminal `done` message for this context.
    Done,
    /// A provider error message.
    Error(String),
    /// Anything else (timestamps, flush acks, mismatched context).
    Ignore,
}

/// Decode one Cartesia server JSON message at `rate` for `context_id` (pure —
/// the wire-fixture seam). A `chunk` with base64 `data` becomes an
/// [`Frame::TtsAudio`]; `done` ends the run; `error` surfaces the message.
fn decode_message(value: &Value, rate: u32, context_id: &Arc<str>) -> Decoded {
    match value.get("type").and_then(|t| t.as_str()) {
        Some("chunk") => {
            let Some(b64) = value.get("data").and_then(|d| d.as_str()) else {
                return Decoded::Ignore;
            };
            let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(b64) else {
                return Decoded::Ignore;
            };
            let pcm = pcm_from_le_bytes(&bytes);
            let audio = Arc::new(AudioFrame::mono(pcm, rate));
            Decoded::Audio(Frame::TtsAudio {
                audio,
                context_id: Some(context_id.clone()),
            })
        }
        Some("done") => Decoded::Done,
        Some("error") => Decoded::Error(
            value
                .get("error")
                .and_then(|e| e.as_str())
                .unwrap_or("unknown")
                .to_string(),
        ),
        _ => Decoded::Ignore,
    }
}

/// Decode little-endian i16 PCM bytes into samples (drops a trailing odd byte).
fn pcm_from_le_bytes(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_body_matches_cartesia_schema() {
        let body = build_request("hello there", "ctx-1", "sonic-2", "voice-x", 24_000);
        assert_eq!(body["transcript"], "hello there");
        assert_eq!(body["model_id"], "sonic-2");
        assert_eq!(body["context_id"], "ctx-1");
        assert_eq!(body["continue"], false);
        assert_eq!(body["voice"]["mode"], "id");
        assert_eq!(body["voice"]["id"], "voice-x");
        assert_eq!(body["output_format"]["container"], "raw");
        assert_eq!(body["output_format"]["encoding"], "pcm_s16le");
        assert_eq!(body["output_format"]["sample_rate"], 24_000);
    }

    #[test]
    fn url_uses_the_fixed_cartesia_host_with_version() {
        let c = CartesiaTts::new("k", "v");
        let url = c.url();
        assert!(url.starts_with("wss://api.cartesia.ai/tts/websocket?"));
        assert!(url.contains("cartesia_version=2026-03-01"));
    }

    #[test]
    fn decode_chunk_message_into_tts_audio() {
        // Two LE i16 samples: 1 and -1 → bytes [1,0, 255,255].
        let b64 = base64::engine::general_purpose::STANDARD.encode([1u8, 0, 255, 255]);
        let ctx: Arc<str> = Arc::from("ctx-1");
        let msg = json!({ "type": "chunk", "data": b64, "context_id": "ctx-1" });
        match decode_message(&msg, 24_000, &ctx) {
            Decoded::Audio(Frame::TtsAudio { audio, context_id }) => {
                assert_eq!(audio.pcm, vec![1, -1]);
                assert_eq!(audio.sample_rate, 24_000);
                assert_eq!(context_id.as_deref(), Some("ctx-1"));
            }
            _ => panic!("expected TtsAudio"),
        }
    }

    #[test]
    fn decode_done_and_error_and_ignore() {
        let ctx: Arc<str> = Arc::from("ctx-1");
        assert!(matches!(
            decode_message(&json!({"type": "done"}), 24_000, &ctx),
            Decoded::Done
        ));
        match decode_message(
            &json!({"type": "error", "error": "bad voice"}),
            24_000,
            &ctx,
        ) {
            Decoded::Error(e) => assert_eq!(e, "bad voice"),
            _ => panic!("expected Error"),
        }
        // A timestamps message → ignored.
        assert!(matches!(
            decode_message(
                &json!({"type": "timestamps", "word_timestamps": {}}),
                24_000,
                &ctx
            ),
            Decoded::Ignore
        ));
    }

    #[test]
    fn pcm_from_le_bytes_drops_trailing_odd_byte() {
        assert_eq!(pcm_from_le_bytes(&[1, 0, 2, 0, 99]), vec![1, 2]);
    }

    /// Live smoke (requires `CARTESIA_API_KEY` + `CARTESIA_VOICE_ID`): synthesize
    /// one short utterance and confirm audio came back. Run:
    /// `CARTESIA_API_KEY=… CARTESIA_VOICE_ID=… cargo test -p flowcat-services --features tts-cartesia -- --ignored cartesia_live`
    #[tokio::test]
    #[ignore = "requires CARTESIA_API_KEY + CARTESIA_VOICE_ID"]
    async fn cartesia_live_synthesizes_audio() {
        let key = std::env::var("CARTESIA_API_KEY").expect("CARTESIA_API_KEY");
        let voice = std::env::var("CARTESIA_VOICE_ID").expect("CARTESIA_VOICE_ID");
        let mut tts = CartesiaTts::new(key, voice);
        tts.start(&StartParams::default()).await.expect("connect");
        let frames = tts.run_tts("Hello from flowcat.").await.expect("run_tts");
        let audio_chunks = frames
            .iter()
            .filter(|f| matches!(f, Frame::TtsAudio { .. }))
            .count();
        assert!(audio_chunks > 0, "expected at least one TtsAudio chunk");
    }
}
