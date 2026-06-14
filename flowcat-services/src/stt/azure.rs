// SPDX-License-Identifier: Apache-2.0
//
//! **Azure Speech** streaming STT (Cognitive Services Speech-to-Text WS).
//!
//! A **(D)istinct** streaming-WebSocket client over Azure's documented Speech
//! Service WS endpoint
//! `wss://{region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language=…&format=detailed`,
//! authenticated with the `Ocp-Apim-Subscription-Key: <key>` header. Audio is
//! streamed as raw little-endian PCM **binary** frames.
//!
//! Azure frames each WS **text** message as an HTTP-ish header block, a blank
//! line, then a JSON body (the `Path` header names the message). The
//! transcription-bearing messages are:
//!
//! ```text
//! X-RequestId:…
//! Path:speech.hypothesis
//! Content-Type:application/json
//!
//! { "Text": "book a", "Offset": 100, "Duration": 50 }
//! ```
//!
//! `speech.hypothesis` → [`Frame::InterimTranscription`]; `speech.phrase` with
//! `RecognitionStatus == "Success"` → final [`Frame::Transcription`] (its
//! `DisplayText`, or the top `NBest[0].Display`). Every other path
//! (`turn.start`, `speech.startDetected`, `turn.end`, …) yields nothing. The
//! header-split + JSON map is a **pure** function ([`decode_frame`]) so the wire
//! format is unit-tested without a socket — and it never panics on a malformed
//! frame.
//!
//! This provider does its own frame parsing (Azure's header-framed text differs
//! from the bare-JSON sibling WS providers), but reuses the shared transport's
//! PCM helper and the same defensive untrusted-data discipline.

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

#[allow(clippy::duplicate_mod)] // each WS provider owns its own copy (feature-independent)
#[path = "ws_stt_common.rs"]
pub mod ws_stt;

type ClientSocket =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;
type Sink = futures::stream::SplitSink<ClientSocket, Message>;

/// Azure Speech streaming-STT session.
pub struct AzureStt {
    api_key: String,
    region: String,
    sample_rate: u32,
    language: String,
    conn: Option<Connection>,
    muted: bool,
}

struct Connection {
    sink: Arc<AsyncMutex<Sink>>,
    frames: mpsc::UnboundedReceiver<Frame>,
    reader: JoinHandle<()>,
}

impl Drop for Connection {
    fn drop(&mut self) {
        self.reader.abort();
    }
}

impl AzureStt {
    /// Construct bound to `api_key` + Azure `region` (e.g. `eastus`). Default
    /// 16 kHz input, English.
    pub fn new(api_key: impl Into<String>, region: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            region: region.into(),
            sample_rate: 16_000,
            language: "en-US".to_string(),
            muted: false,
            conn: None,
        }
    }

    /// Override the input sample rate (default 16 kHz).
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Override the recognition language (default `en-US`).
    pub fn language(mut self, lang: impl Into<String>) -> Self {
        self.language = lang.into();
        self
    }

    /// The connect URL for this config (testable without a socket). The key is
    /// **never** placed in the URL (it travels in `Ocp-Apim-Subscription-Key`).
    pub(crate) fn url(&self) -> String {
        format!(
            "wss://{}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language={}&format=detailed",
            self.region, self.language
        )
    }
}

#[async_trait]
impl SttService for AzureStt {
    fn name(&self) -> &str {
        "azure"
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        let mut request = self
            .url()
            .into_client_request()
            .map_err(|e| FlowcatError::Network(format!("azure url: {e}")))?;
        request.headers_mut().insert(
            "Ocp-Apim-Subscription-Key",
            self.api_key
                .parse()
                .map_err(|e| FlowcatError::Network(format!("azure auth header: {e}")))?,
        );
        let (socket, _resp) = tokio_tungstenite::connect_async(request)
            .await
            .map_err(|e| FlowcatError::Network(format!("azure connect: {e}")))?;
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

    async fn run_stt(&mut self, audio: Arc<AudioFrame>) -> Result<Vec<Frame>> {
        if self.muted {
            return Ok(vec![]);
        }
        let conn = self
            .conn
            .as_mut()
            .ok_or_else(|| FlowcatError::Network("azure: run_stt before start".into()))?;
        let bytes = ws_stt::pcm_le_bytes(&audio);
        {
            let mut sink = conn.sink.lock().await;
            sink.send(Message::binary(bytes))
                .await
                .map_err(|e| FlowcatError::Network(format!("azure send: {e}")))?;
        }
        let mut out = Vec::new();
        while let Ok(f) = conn.frames.try_recv() {
            out.push(f);
        }
        Ok(out)
    }

    async fn set_muted(&mut self, muted: bool) {
        self.muted = muted;
    }
}

/// Split an Azure WS text frame's `Path` header value and JSON body. Returns
/// `(path, body_value)` or `None` for a malformed/header-only frame. Pure.
fn split_azure_frame(frame: &str) -> Option<(String, Value)> {
    // Header block and JSON body are separated by a blank line (CRLF or LF).
    let (headers, body) = frame
        .split_once("\r\n\r\n")
        .or_else(|| frame.split_once("\n\n"))?;
    let mut path = None;
    for line in headers.lines() {
        if let Some((name, value)) = line.split_once(':') {
            if name.trim().eq_ignore_ascii_case("path") {
                path = Some(value.trim().to_string());
            }
        }
    }
    let path = path?;
    let value = serde_json::from_str::<Value>(body.trim()).ok()?;
    Some((path, value))
}

/// Decode one Azure WS text frame into transcription frames. **Pure** — the seam
/// the wire-fixture tests drive. `speech.hypothesis` → interim; a successful
/// `speech.phrase` → final; everything else → nothing. Never panics on a
/// malformed/header-only/non-JSON frame.
pub(crate) fn decode_frame(frame: &str) -> Vec<Frame> {
    let Some((path, body)) = split_azure_frame(frame) else {
        return vec![];
    };
    let user_id: Arc<str> = Arc::from("user");
    match path.as_str() {
        "speech.hypothesis" => {
            let text = body.get("Text").and_then(|t| t.as_str()).unwrap_or("");
            if text.is_empty() {
                return vec![];
            }
            vec![Frame::InterimTranscription {
                text: text.to_string(),
                user_id,
                language: None,
            }]
        }
        "speech.phrase" => {
            let status = body
                .get("RecognitionStatus")
                .and_then(|s| s.as_str())
                .unwrap_or("");
            if status != "Success" {
                return vec![];
            }
            let text = body
                .get("DisplayText")
                .and_then(|t| t.as_str())
                .or_else(|| {
                    body.get("NBest")
                        .and_then(|n| n.as_array())
                        .and_then(|a| a.first())
                        .and_then(|b| b.get("Display"))
                        .and_then(|d| d.as_str())
                })
                .unwrap_or("");
            if text.is_empty() {
                return vec![];
            }
            vec![Frame::Transcription {
                text: text.to_string(),
                user_id,
                language: None,
                final_: true,
            }]
        }
        _ => vec![],
    }
}

/// The persistent reader: decode each Azure text frame; ignore malformed frames
/// and binary/control frames; end on close/error.
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
        for frame in decode_frame(&text) {
            if tx.send(frame).is_err() {
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_uses_region_host_and_omits_key() {
        let c = AzureStt::new("secret", "eastus").language("en-US");
        let u = c.url();
        assert!(u.starts_with("wss://eastus.stt.speech.microsoft.com/speech/recognition/"));
        assert!(u.contains("language=en-US"));
        assert!(!u.contains("secret"));
    }

    #[test]
    fn decode_speech_phrase_success_is_final() {
        let frame =
            "X-RequestId:abc\r\nPath:speech.phrase\r\nContent-Type:application/json\r\n\r\n\
            {\"RecognitionStatus\":\"Success\",\"DisplayText\":\"book a dentist\",\"Offset\":0}";
        match &decode_frame(frame)[..] {
            [Frame::Transcription { text, final_, .. }] => {
                assert_eq!(text, "book a dentist");
                assert!(final_);
            }
            other => panic!("expected final, got {other:?}"),
        }
    }

    #[test]
    fn decode_speech_phrase_falls_back_to_nbest_display() {
        let frame = "Path:speech.phrase\r\n\r\n\
            {\"RecognitionStatus\":\"Success\",\"NBest\":[{\"Display\":\"hello world\"}]}";
        assert!(matches!(&decode_frame(frame)[..],
            [Frame::Transcription { text, .. }] if text == "hello world"));
    }

    #[test]
    fn decode_hypothesis_is_interim() {
        let frame = "Path:speech.hypothesis\r\n\r\n{\"Text\":\"book a\",\"Offset\":0}";
        assert!(matches!(
            decode_frame(frame).as_slice(),
            [Frame::InterimTranscription { .. }]
        ));
    }

    #[test]
    fn decode_ignores_control_nomatch_empty_and_malformed() {
        // Non-success phrase → nothing.
        assert!(
            decode_frame("Path:speech.phrase\r\n\r\n{\"RecognitionStatus\":\"NoMatch\"}")
                .is_empty()
        );
        // turn.start / speech.startDetected → nothing.
        assert!(decode_frame("Path:turn.start\r\n\r\n{\"context\":{}}").is_empty());
        assert!(decode_frame("Path:speech.startDetected\r\n\r\n{\"Offset\":0}").is_empty());
        // Empty hypothesis text → nothing.
        assert!(decode_frame("Path:speech.hypothesis\r\n\r\n{\"Text\":\"\"}").is_empty());
        // Header-only / no body separator → no panic, nothing.
        assert!(decode_frame("Path:speech.phrase\r\nContent-Type:application/json").is_empty());
        // Malformed JSON body → nothing.
        assert!(decode_frame("Path:speech.phrase\r\n\r\nnot json").is_empty());
        // No Path header → nothing.
        assert!(decode_frame("X-RequestId:abc\r\n\r\n{\"Text\":\"x\"}").is_empty());
        // Totally empty → nothing.
        assert!(decode_frame("").is_empty());
    }

    #[test]
    fn pcm_helper_is_little_endian() {
        let af = AudioFrame::mono(vec![1, -2, 256], 16_000);
        assert_eq!(ws_stt::pcm_le_bytes(&af), vec![1, 0, 254, 255, 0, 1]);
    }

    /// Live smoke (requires `AZURE_SPEECH_KEY` + `AZURE_SPEECH_REGION`). Run:
    /// `AZURE_SPEECH_KEY=… AZURE_SPEECH_REGION=eastus cargo test -p flowcat-services --features stt-azure -- --ignored azure_live`
    #[tokio::test]
    #[ignore = "requires AZURE_SPEECH_KEY + AZURE_SPEECH_REGION"]
    async fn azure_live_connects_and_streams() {
        let key = std::env::var("AZURE_SPEECH_KEY").expect("AZURE_SPEECH_KEY");
        let region = std::env::var("AZURE_SPEECH_REGION").expect("AZURE_SPEECH_REGION");
        let mut stt = AzureStt::new(key, region);
        stt.start(&StartParams::default()).await.expect("connect");
        let silence = Arc::new(AudioFrame::mono(vec![0i16; 1600], 16_000));
        let _ = stt.run_stt(silence).await.expect("run_stt");
    }
}
