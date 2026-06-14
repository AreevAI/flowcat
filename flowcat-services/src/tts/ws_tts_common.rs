// SPDX-License-Identifier: Apache-2.0
//
//! Shared streaming-WebSocket TTS transport (Group F).
//!
//! The one transport seam reused by every distinct-schema streaming-WS TTS
//! provider in this group (elevenlabs, deepgram, rime, asyncai, gradium, soniox,
//! resemble). It is **not** a `mod` in `tts/mod.rs` (which the fan-out must not
//! edit) — each provider file pulls it in with
//! `#[path = "ws_tts_common.rs"] mod ws_tts;`, so a single-feature build (e.g.
//! `--features tts-rime`) compiles its own private copy and never depends on a
//! sibling provider's feature.
//!
//! Shape (the Cartesia reference impl, generalized): [`TtsService::run_tts`] is a
//! **per-utterance request/response** — text in → all of this utterance's frames
//! out. So this helper, unlike the STT reader-task seam, reads **inline**:
//! [`WsTtsSession::synthesize`] sends the provider's synthesis message(s) and then
//! reads server frames, running the provider's **pure** [`Decode`] fn over each,
//! until that fn says [`Decoded::Done`] (the provider's terminal marker) or the
//! socket closes. It frames the run with [`Frame::TtsStarted`] / [`Frame::TtsStopped`]
//! and turns decoded PCM into [`Frame::TtsAudio`], decoded word timings into
//! [`Frame::TtsText`] (where the provider supplies them).
//!
//! **Security.** Every server frame is untrusted: a non-JSON or malformed message
//! decodes to [`Decoded::Ignore`] (the decode fn never panics), and a closed /
//! errored socket simply ends the read. The connect host is fixed by the calling
//! provider; only the API key (sent as a header, query param, or in an init/config
//! body) and validated numeric params are caller-controlled, so there is no SSRF
//! surface.
//!
//! Each provider compiles its **own** copy of this file (via `#[path]`) and uses a
//! subset of the API (binary-audio providers don't need the base64 decoder; the
//! header-auth providers don't use the query key, etc.) — so the per-provider
//! `dead_code` for the unused parts is expected and sanctioned here at module scope.
#![allow(dead_code)]

use std::sync::Arc;

use base64::Engine as _;
use futures::{SinkExt, StreamExt};
use serde_json::Value;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message;

use flowcat_core::error::{FlowcatError, Result};
use flowcat_core::processor::frame::{AudioFrame, Frame};

type ClientSocket =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// One WS message to send (a provider may send several per utterance: a config /
/// text / flush trio, etc.). Text is JSON; binary is rare on the request side.
pub enum OutMsg {
    /// A text (JSON) frame.
    Text(String),
    /// A binary frame.
    Binary(Vec<u8>),
}

impl OutMsg {
    fn into_ws(self) -> Message {
        match self {
            OutMsg::Text(t) => Message::text(t),
            OutMsg::Binary(b) => Message::binary(b),
        }
    }
}

/// The decode of one TTS server message — the provider's pure wire-fixture seam.
pub enum Decoded {
    /// A decoded PCM audio chunk (little-endian i16 samples).
    Audio(Vec<i16>),
    /// Word/segment timings: `(word_text, start_seconds)` pairs.
    Words(Vec<(String, f32)>),
    /// The terminal marker for this utterance.
    Done,
    /// A provider error message (surfaced as a network error).
    Error(String),
    /// Anything else (metadata, keep-alive acks, ready, a foreign context).
    Ignore,
}

/// A provider's decoder: one server message → [`Decoded`]. Receives the message
/// as either decoded JSON (`Some`) or, for a **binary** WS frame, the raw bytes
/// (`None` JSON + the bytes). Pure — it never touches the socket and never panics
/// on malformed input.
pub type Decode = fn(json: Option<&Value>, binary: Option<&[u8]>) -> Decoded;

/// Per-provider connect configuration for [`WsTtsSession::connect`].
pub struct WsTtsConfig {
    /// The full `wss://…` connect URL (host fixed per provider; a key only ever
    /// appears here for the providers whose API *requires* it as a query param).
    pub url: String,
    /// Extra request headers (e.g. `("xi-api-key", key)`), inserted verbatim.
    pub headers: Vec<(String, String)>,
    /// An optional JSON/text message sent once, immediately after connect (the
    /// init/config handshake some providers require, e.g. AsyncAI).
    pub init_message: Option<String>,
    /// The **pure** decoder: one server message → [`Decoded`].
    pub decode: Decode,
}

/// A live streaming-WS TTS session: just the socket + the provider's decoder.
/// One socket per provider instance; reused across utterances.
pub struct WsTtsSession {
    socket: ClientSocket,
    decode: Decode,
}

impl WsTtsSession {
    /// Open the socket and send the optional init/config message.
    pub async fn connect(cfg: WsTtsConfig) -> Result<Self> {
        let mut request = cfg
            .url
            .into_client_request()
            .map_err(|e| FlowcatError::Network(format!("ws-tts url: {e}")))?;
        for (name, value) in &cfg.headers {
            let header_name: tokio_tungstenite::tungstenite::http::HeaderName = name
                .parse()
                .map_err(|e| FlowcatError::Network(format!("ws-tts header name: {e}")))?;
            let header_value = value
                .parse()
                .map_err(|e| FlowcatError::Network(format!("ws-tts header value: {e}")))?;
            request.headers_mut().insert(header_name, header_value);
        }
        let (mut socket, _resp) = tokio_tungstenite::connect_async(request)
            .await
            .map_err(|e| FlowcatError::Network(format!("ws-tts connect: {e}")))?;
        if let Some(init) = cfg.init_message {
            socket
                .send(Message::text(init))
                .await
                .map_err(|e| FlowcatError::Network(format!("ws-tts init send: {e}")))?;
        }
        Ok(Self {
            socket,
            decode: cfg.decode,
        })
    }

    /// Send the synthesis message(s) for one utterance, then read+decode server
    /// frames inline until the provider's terminal marker (or the socket closes),
    /// framing the run with [`Frame::TtsStarted`] / [`Frame::TtsStopped`].
    ///
    /// `context_id` tags every emitted frame; `rate` is the output sample rate.
    pub async fn synthesize(
        &mut self,
        msgs: Vec<OutMsg>,
        context_id: Arc<str>,
        rate: u32,
    ) -> Result<Vec<Frame>> {
        for msg in msgs {
            self.socket
                .send(msg.into_ws())
                .await
                .map_err(|e| FlowcatError::Network(format!("ws-tts send: {e}")))?;
        }

        let mut out = vec![Frame::TtsStarted {
            context_id: Some(context_id.clone()),
        }];
        while let Some(msg) = self.socket.next().await {
            let decoded = match msg {
                Ok(Message::Text(t)) => match serde_json::from_str::<Value>(&t) {
                    Ok(v) => (self.decode)(Some(&v), None),
                    // untrusted server data: ignore malformed JSON
                    Err(_) => Decoded::Ignore,
                },
                Ok(Message::Binary(b)) => (self.decode)(None, Some(&b)),
                Ok(Message::Ping(_)) | Ok(Message::Pong(_)) => continue,
                Ok(Message::Close(_)) | Err(_) => break,
                Ok(_) => continue,
            };
            match decoded {
                Decoded::Audio(pcm) => {
                    if !pcm.is_empty() {
                        out.push(Frame::TtsAudio {
                            audio: Arc::new(AudioFrame::mono(pcm, rate)),
                            context_id: Some(context_id.clone()),
                        });
                    }
                }
                Decoded::Words(words) => {
                    for (text, _start) in words {
                        if !text.is_empty() {
                            out.push(Frame::TtsText {
                                text,
                                context_id: Some(context_id.clone()),
                            });
                        }
                    }
                }
                Decoded::Done => break,
                Decoded::Error(e) => return Err(FlowcatError::Network(format!("ws-tts: {e}"))),
                Decoded::Ignore => {}
            }
        }
        out.push(Frame::TtsStopped {
            context_id: Some(context_id),
        });
        Ok(out)
    }
}

/// Borrow the live session or return a uniform "run_tts before start" error.
pub fn require<'a>(
    session: &'a mut Option<WsTtsSession>,
    provider: &str,
) -> Result<&'a mut WsTtsSession> {
    session
        .as_mut()
        .ok_or_else(|| FlowcatError::Network(format!("{provider}: run_tts before start")))
}

/// Standard (RFC 4648) base64-decode of a server audio field → bytes. Returns an
/// empty vec on malformed input (never panics — server data is untrusted).
pub fn base64_decode(input: &str) -> Vec<u8> {
    base64::engine::general_purpose::STANDARD
        .decode(input)
        .unwrap_or_default()
}

/// Decode little-endian i16 PCM bytes into samples (drops a trailing odd byte).
pub fn pcm_from_le_bytes(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Pull a base64 `audio` field out of a JSON message and decode it to PCM. Returns
/// an empty vec when the field is absent or malformed.
pub fn pcm_from_b64_field(value: &Value, field: &str) -> Vec<i16> {
    value
        .get(field)
        .and_then(|d| d.as_str())
        .map(|b64| pcm_from_le_bytes(&base64_decode(b64)))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm_from_le_bytes_round_trips_and_drops_odd_tail() {
        // 1 and -1 → [1,0, 255,255]; a trailing odd byte is dropped.
        assert_eq!(pcm_from_le_bytes(&[1, 0, 255, 255, 9]), vec![1, -1]);
        assert!(pcm_from_le_bytes(&[]).is_empty());
    }

    #[test]
    fn base64_decode_is_lenient_on_garbage() {
        assert_eq!(base64_decode("AQA="), vec![1, 0]);
        assert!(base64_decode("!!! not base64 !!!").is_empty());
    }

    #[test]
    fn pcm_from_b64_field_handles_present_absent_and_bad() {
        let b64 = base64::engine::general_purpose::STANDARD.encode([1u8, 0, 255, 255]);
        let msg = serde_json::json!({ "audio": b64 });
        assert_eq!(pcm_from_b64_field(&msg, "audio"), vec![1, -1]);
        // Missing field → empty (no panic).
        assert!(pcm_from_b64_field(&serde_json::json!({}), "audio").is_empty());
        // Non-string field → empty.
        assert!(pcm_from_b64_field(&serde_json::json!({ "audio": 7 }), "audio").is_empty());
    }
}
