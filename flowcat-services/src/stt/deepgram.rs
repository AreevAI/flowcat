// SPDX-License-Identifier: Apache-2.0
//
//! Deepgram streaming STT over the `/v1/listen` WebSocket.
//!
//! Wire protocol (cross-checked against pipecat
//! `services/deepgram/stt.py`): connect to
//! `wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate=…&channels=1&model=…&interim_results=true`
//! with an `Authorization: Token <key>` header, stream raw little-endian PCM as
//! **binary** WS frames, and receive JSON `Results` messages:
//!
//! ```json
//! { "type": "Results",
//!   "is_final": true,
//!   "channel": { "alternatives": [ { "transcript": "hello", "confidence": 0.99 } ] } }
//! ```
//!
//! A persistent reader task (the streaming-reader pattern, mirroring the Gemini
//! client) decodes each server frame into [`Frame`]s and pushes them onto an
//! mpsc queue; [`SttService::run_stt`] forwards the queued frames for the
//! triggering audio chunk (and sends the chunk's PCM up the socket). The decode
//! is a **pure function** ([`decode_results`]) so the wire format is unit-tested
//! without a socket.

use std::sync::Arc;

use async_trait::async_trait;
use futures::{SinkExt, StreamExt};
use serde_json::Value;
use tokio::sync::mpsc;
use tokio::sync::Mutex as AsyncMutex;
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message;

use flowcat_core::error::{FlowcatError, Result};
use flowcat_core::processor::frame::{AudioFrame, Frame, StartParams};
use flowcat_core::service::SttService;

/// Deepgram's streaming-listen WSS base. The query string (encoding/rate/model)
/// is appended at connect time. The **host is fixed** — only the API key and a
/// small set of validated query params are caller-controlled (no SSRF surface).
pub const DEEPGRAM_WSS_BASE: &str = "wss://api.deepgram.com/v1/listen";

/// Builder for [`DeepgramStt`]: API key + model + sample rate, with sane
/// defaults (nova-2, 16 kHz, interim results on).
#[derive(Debug, Clone)]
pub struct DeepgramSttBuilder {
    api_key: String,
    model: String,
    sample_rate: u32,
    interim_results: bool,
}

impl DeepgramSttBuilder {
    /// Start a builder bound to `api_key`.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: "nova-2".to_string(),
            sample_rate: 16_000,
            interim_results: true,
        }
    }

    /// Override the model (default `nova-2`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the input sample rate (default 16 kHz).
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Toggle interim (partial) results (default on).
    pub fn interim_results(mut self, on: bool) -> Self {
        self.interim_results = on;
        self
    }

    /// Build the (not-yet-connected) client.
    pub fn build(self) -> DeepgramStt {
        DeepgramStt {
            api_key: self.api_key,
            model: self.model,
            sample_rate: self.sample_rate,
            interim_results: self.interim_results,
            conn: None,
            muted: false,
        }
    }

    /// The query string for this config (testable without a socket).
    pub(crate) fn query(&self) -> String {
        format!(
            "encoding=linear16&sample_rate={}&channels=1&model={}&interim_results={}",
            self.sample_rate,
            self.model,
            if self.interim_results {
                "true"
            } else {
                "false"
            }
        )
    }
}

/// Live socket state, present once [`SttService::start`] connected.
struct Connection {
    /// Write half — raw PCM goes out as binary frames.
    sink: Arc<AsyncMutex<futures::stream::SplitSink<ClientSocket, Message>>>,
    /// Decoded transcription frames, drained per `run_stt`.
    frames: mpsc::UnboundedReceiver<Frame>,
    /// Background reader task; aborted on drop.
    reader: JoinHandle<()>,
}

impl Drop for Connection {
    fn drop(&mut self) {
        self.reader.abort();
    }
}

type ClientSocket =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// A native Deepgram streaming-STT session.
pub struct DeepgramStt {
    api_key: String,
    model: String,
    sample_rate: u32,
    interim_results: bool,
    conn: Option<Connection>,
    muted: bool,
}

impl DeepgramStt {
    /// Construct with the default model/rate. Use [`DeepgramSttBuilder`] for
    /// non-default settings.
    pub fn new(api_key: impl Into<String>) -> Self {
        DeepgramSttBuilder::new(api_key).build()
    }

    fn builder(&self) -> DeepgramSttBuilder {
        DeepgramSttBuilder {
            api_key: self.api_key.clone(),
            model: self.model.clone(),
            sample_rate: self.sample_rate,
            interim_results: self.interim_results,
        }
    }

    fn url(&self) -> String {
        format!("{DEEPGRAM_WSS_BASE}?{}", self.builder().query())
    }

    /// Open the socket + spawn the decode reader.
    async fn open(&mut self) -> Result<()> {
        let mut request = self
            .url()
            .into_client_request()
            .map_err(|e| FlowcatError::Network(format!("deepgram url: {e}")))?;
        // Deepgram authenticates with `Authorization: Token <key>`.
        request.headers_mut().insert(
            "Authorization",
            format!("Token {}", self.api_key)
                .parse()
                .map_err(|e| FlowcatError::Network(format!("deepgram auth header: {e}")))?,
        );
        let (socket, _resp) = tokio_tungstenite::connect_async(request)
            .await
            .map_err(|e| FlowcatError::Network(format!("deepgram connect: {e}")))?;
        let (sink, stream) = socket.split();
        let (tx, rx) = mpsc::unbounded_channel();
        let reader = tokio::spawn(reader_task(stream, tx));
        self.conn = Some(Connection {
            sink: Arc::new(AsyncMutex::new(sink)),
            frames: rx,
            reader,
        });
        Ok(())
    }

    /// PCM → little-endian bytes for the binary WS frame.
    fn pcm_bytes(audio: &AudioFrame) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(audio.pcm.len() * 2);
        for s in &audio.pcm {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        bytes
    }
}

#[async_trait]
impl SttService for DeepgramStt {
    fn name(&self) -> &str {
        "deepgram"
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        // Lazy connect: do NOT open the WebSocket here. `start()` runs inside the
        // pipeline's Start handshake (every processor's `start()` must complete
        // before any audio — or the greeting `ClientConnected` — is pumped to the
        // head), so a slow/hung Deepgram connect would stall the whole call. We
        // open on the first `run_stt` instead. (The realtime path has no STT
        // socket on its Start path, which is why it never stalled.)
        Ok(())
    }

    async fn run_stt(&mut self, audio: Arc<AudioFrame>) -> Result<Vec<Frame>> {
        // Lazy connect (see `start`): open the WS on the first audio chunk.
        if self.conn.is_none() {
            self.open().await?;
        }
        let muted = self.muted;
        // While muted, feed SILENCE (not the mic): Deepgram idle-closes after ~10s
        // of no audio, so a long bot turn would kill the socket. Silence keeps it
        // warm without transcribing the bot's echo; decoded frames are dropped below.
        let bytes = if muted {
            vec![0u8; audio.pcm.len() * 2] // i16-LE silence, same chunk length
        } else {
            Self::pcm_bytes(&audio)
        };
        let send_res = {
            let conn = self
                .conn
                .as_mut()
                .ok_or_else(|| FlowcatError::Network("deepgram: not connected".into()))?;
            let mut sink = conn.sink.lock().await;
            sink.send(Message::binary(bytes)).await
        };
        // Send failed → socket is dead; drop it so the next chunk reconnects.
        if let Err(e) = send_res {
            tracing::warn!(error = %e, "deepgram send failed; dropping socket to reconnect");
            self.conn = None;
            return Ok(vec![]);
        }
        // Drain whatever the reader has decoded; drop it while muted.
        let mut out = Vec::new();
        if let Some(conn) = self.conn.as_mut() {
            while let Ok(f) = conn.frames.try_recv() {
                out.push(f);
            }
        }
        if muted {
            out.clear();
        }
        Ok(out)
    }

    async fn set_muted(&mut self, muted: bool) {
        self.muted = muted;
    }
}

/// The persistent reader: parse each server frame into transcription [`Frame`]s
/// and queue them. Mirrors the Gemini client's reader-task→mpsc bridge.
async fn reader_task(
    mut stream: futures::stream::SplitStream<ClientSocket>,
    tx: mpsc::UnboundedSender<Frame>,
) {
    while let Some(msg) = stream.next().await {
        let text = match msg {
            Ok(Message::Text(t)) => t.to_string(),
            Ok(Message::Binary(_)) | Ok(Message::Ping(_)) | Ok(Message::Pong(_)) => continue,
            Ok(Message::Close(_)) | Err(_) => break,
            Ok(_) => continue,
        };
        let Ok(value) = serde_json::from_str::<Value>(&text) else {
            continue;
        };
        for frame in decode_results(&value) {
            if tx.send(frame).is_err() {
                return; // consumer gone
            }
        }
    }
}

/// Decode one Deepgram server JSON message into transcription frames. A `Results`
/// message yields an [`Frame::InterimTranscription`] (when `is_final` is false)
/// or a final [`Frame::Transcription`]. Empty transcripts and non-`Results`
/// messages (Metadata, SpeechStarted, …) yield nothing. **Pure** — the seam the
/// wire-fixture tests drive.
pub(crate) fn decode_results(value: &Value) -> Vec<Frame> {
    if value.get("type").and_then(|t| t.as_str()) != Some("Results") {
        return vec![];
    }
    let transcript = value
        .get("channel")
        .and_then(|c| c.get("alternatives"))
        .and_then(|a| a.as_array())
        .and_then(|a| a.first())
        .and_then(|alt| alt.get("transcript"))
        .and_then(|t| t.as_str())
        .unwrap_or("");
    if transcript.is_empty() {
        return vec![];
    }
    let is_final = value
        .get("is_final")
        .and_then(|f| f.as_bool())
        .unwrap_or(false);
    let user_id: Arc<str> = Arc::from("user");
    if is_final {
        vec![Frame::Transcription {
            text: transcript.to_string(),
            user_id,
            language: None,
            final_: true,
        }]
    } else {
        vec![Frame::InterimTranscription {
            text: transcript.to_string(),
            user_id,
            language: None,
        }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn query_encodes_the_listen_params() {
        let q = DeepgramSttBuilder::new("k")
            .model("nova-3")
            .sample_rate(8000)
            .interim_results(false)
            .query();
        assert_eq!(
            q,
            "encoding=linear16&sample_rate=8000&channels=1&model=nova-3&interim_results=false"
        );
    }

    #[test]
    fn url_uses_the_fixed_deepgram_host() {
        let c = DeepgramStt::new("secret");
        assert!(c.url().starts_with("wss://api.deepgram.com/v1/listen?"));
        // The API key is never in the URL (it goes in the Authorization header).
        assert!(!c.url().contains("secret"));
    }

    #[test]
    fn decode_final_results_message() {
        let msg = json!({
            "type": "Results",
            "is_final": true,
            "channel": { "alternatives": [ { "transcript": "book a dentist", "confidence": 0.98 } ] }
        });
        let frames = decode_results(&msg);
        assert_eq!(frames.len(), 1);
        match &frames[0] {
            Frame::Transcription { text, final_, .. } => {
                assert_eq!(text, "book a dentist");
                assert!(final_);
            }
            other => panic!("expected final Transcription, got {}", other.name()),
        }
    }

    #[test]
    fn decode_interim_results_message() {
        let msg = json!({
            "type": "Results",
            "is_final": false,
            "channel": { "alternatives": [ { "transcript": "book a", "confidence": 0.5 } ] }
        });
        let frames = decode_results(&msg);
        assert_eq!(frames.len(), 1);
        assert!(matches!(frames[0], Frame::InterimTranscription { .. }));
    }

    #[test]
    fn decode_ignores_empty_and_non_results() {
        // Empty transcript → nothing.
        let empty = json!({
            "type": "Results", "is_final": true,
            "channel": { "alternatives": [ { "transcript": "" } ] }
        });
        assert!(decode_results(&empty).is_empty());
        // Metadata / other message types → nothing.
        let meta = json!({ "type": "Metadata", "request_id": "abc" });
        assert!(decode_results(&meta).is_empty());
        let speech = json!({ "type": "SpeechStarted", "timestamp": 1.0 });
        assert!(decode_results(&speech).is_empty());
    }

    #[test]
    fn pcm_bytes_are_little_endian() {
        let af = AudioFrame::mono(vec![1, -2, 256], 16_000);
        let bytes = DeepgramStt::pcm_bytes(&af);
        assert_eq!(bytes, vec![1, 0, 254, 255, 0, 1]);
    }

    /// Live smoke (requires `DEEPGRAM_API_KEY`): connect, send a beat of silence,
    /// confirm the socket stays open. Run with:
    /// `DEEPGRAM_API_KEY=… cargo test -p flowcat-services --features stt-deepgram -- --ignored deepgram_live`
    #[tokio::test]
    #[ignore = "requires DEEPGRAM_API_KEY"]
    async fn deepgram_live_connects_and_streams() {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let key = std::env::var("DEEPGRAM_API_KEY").expect("DEEPGRAM_API_KEY");
        let mut stt = DeepgramStt::new(key);
        stt.start(&StartParams::default()).await.expect("connect");
        // 100 ms of silence at 16 kHz.
        let silence = Arc::new(AudioFrame::mono(vec![0i16; 1600], 16_000));
        let _ = stt.run_stt(silence).await.expect("run_stt");
    }
}
