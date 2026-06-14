// SPDX-License-Identifier: Apache-2.0
//
//! Shared streaming-WebSocket STT transport.
//!
//! This is the one transport seam reused by every distinct-schema streaming-WS
//! STT provider in this group (assemblyai, gladia, soniox, speechmatics,
//! cartesia, gradium). It is **not** a `mod` in `stt/mod.rs` (which the fan-out
//! must not edit) — each provider file pulls it in with
//! `#[path = "ws_stt_common.rs"] mod ws_stt;`, so a single-feature build (e.g.
//! `--features stt-gladia`) compiles its own private copy and never depends on a
//! sibling provider's feature.
//!
//! Shape (the Deepgram reference impl, generalized): a persistent connection +
//! a reader task that runs the provider's **pure** JSON decoder over each server
//! frame and queues the resulting [`Frame`]s onto an mpsc channel;
//! `run_stt` sends the chunk and drains whatever the reader has decoded so far
//! (it never blocks on a round-trip). Each provider supplies its own
//! URL / headers / optional init-handshake / audio-encoding / decode fn.
//!
//! **Security.** Every server frame is untrusted: a non-JSON or malformed
//! message is silently dropped (the decode fn returns `vec![]`), nothing panics,
//! and a closed/errored socket simply ends the reader task. The connect host is
//! fixed by the calling provider; only the API key (sent as a header or in the
//! init body) and validated numeric query params are caller-controlled, so there
//! is no SSRF surface and the key never appears in the URL.
//!
//! Each WS provider compiles its **own** copy of this file (via `#[path]`) and
//! uses a subset of the API (binary-PCM senders don't need `base64_encode`, etc.)
//! — so the per-provider `dead_code` for the unused parts is expected and
//! sanctioned here at module scope.
#![allow(dead_code)]

use std::sync::Arc;

use futures::{SinkExt, StreamExt};
use serde_json::Value;
use tokio::sync::mpsc;
use tokio::sync::Mutex as AsyncMutex;
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message;

use flowcat_core::error::{FlowcatError, Result};
use flowcat_core::processor::frame::{AudioFrame, Frame};

type ClientSocket =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;
type Sink = futures::stream::SplitSink<ClientSocket, Message>;

/// Per-provider connect configuration for [`WsSttSession`]. `Clone` so the session
/// can keep it for lazy connect + reconnect.
#[derive(Clone)]
pub struct WsSttConfig {
    /// The full `wss://…` connect URL (host fixed per provider; key never here).
    pub url: String,
    /// Extra request headers (e.g. `("Authorization", key)`), inserted verbatim.
    pub headers: Vec<(String, String)>,
    /// An optional JSON message sent once, immediately after connect (the
    /// init/config handshake some providers require, e.g. Soniox).
    pub init_message: Option<String>,
    /// The **pure** decoder: one server JSON message → transcription frames.
    pub decode: fn(&Value) -> Vec<Frame>,
}

/// A streaming-WS STT session. Holds its [`WsSttConfig`] so it can connect
/// **lazily** (on the first send — keeping the socket connect OFF the pipeline Start
/// handshake, which an eager connect would stall) and **reconnect** if the socket
/// dies (idle-close during a long muted turn, or a network blip). The live
/// connection lives in [`Conn`].
pub struct WsSttSession {
    cfg: WsSttConfig,
    conn: Option<Conn>,
}

/// One live connection: the write half + the reader-decoded frame queue + the
/// reader task handle (aborted on drop).
struct Conn {
    sink: Arc<AsyncMutex<Sink>>,
    frames: mpsc::UnboundedReceiver<Frame>,
    reader: JoinHandle<()>,
}

impl Drop for Conn {
    fn drop(&mut self) {
        self.reader.abort();
    }
}

impl WsSttSession {
    /// Lazy: store the config; connect on the first send. Use this from a
    /// provider's `start()` so the socket connect never stalls the Start handshake.
    pub fn lazy(cfg: WsSttConfig) -> Self {
        Self { cfg, conn: None }
    }

    /// Eager (back-compat): connect immediately.
    pub async fn connect(cfg: WsSttConfig) -> Result<Self> {
        let mut s = Self::lazy(cfg);
        s.ensure().await?;
        Ok(s)
    }

    /// Open the socket, send the optional init message, and spawn the decode reader
    /// — unless already connected.
    async fn ensure(&mut self) -> Result<()> {
        if self.conn.is_some() {
            return Ok(());
        }
        let mut request = self
            .cfg
            .url
            .clone()
            .into_client_request()
            .map_err(|e| FlowcatError::Network(format!("ws-stt url: {e}")))?;
        for (name, value) in &self.cfg.headers {
            let header_name: tokio_tungstenite::tungstenite::http::HeaderName = name
                .parse()
                .map_err(|e| FlowcatError::Network(format!("ws-stt header name: {e}")))?;
            let header_value = value
                .parse()
                .map_err(|e| FlowcatError::Network(format!("ws-stt header value: {e}")))?;
            request.headers_mut().insert(header_name, header_value);
        }
        let (socket, _resp) = tokio_tungstenite::connect_async(request)
            .await
            .map_err(|e| FlowcatError::Network(format!("ws-stt connect: {e}")))?;
        let (mut sink, stream) = socket.split();
        if let Some(init) = &self.cfg.init_message {
            sink.send(Message::text(init.clone()))
                .await
                .map_err(|e| FlowcatError::Network(format!("ws-stt init send: {e}")))?;
        }
        let (tx, rx) = mpsc::unbounded_channel();
        let reader = tokio::spawn(reader_task(stream, tx, self.cfg.decode));
        self.conn = Some(Conn {
            sink: Arc::new(AsyncMutex::new(sink)),
            frames: rx,
            reader,
        });
        Ok(())
    }

    /// Send a chunk's PCM as one **binary** WS frame (little-endian i16).
    pub async fn send_pcm_binary(&mut self, audio: &AudioFrame) -> Result<()> {
        let bytes = pcm_le_bytes(audio);
        self.send_message(Message::binary(bytes)).await
    }

    /// Send an arbitrary already-encoded WS message (text or binary). Used by
    /// providers that wrap audio in a JSON envelope (Gladia, Sarvam).
    pub async fn send_message(&mut self, msg: Message) -> Result<()> {
        self.ensure().await?;
        let res = {
            let conn = self.conn.as_ref().expect("ensure() connected");
            let mut sink = conn.sink.lock().await;
            sink.send(msg).await
        };
        if let Err(e) = res {
            // Dead socket (idle-closed / network blip): drop it so the next send
            // lazily reconnects, instead of wedging on a dead connection.
            self.conn = None;
            return Err(FlowcatError::Network(format!("ws-stt send: {e}")));
        }
        Ok(())
    }

    /// Send a text control/command (e.g. Cartesia `"finalize"`/`"done"`).
    pub async fn send_text(&mut self, text: impl Into<String>) -> Result<()> {
        self.send_message(Message::text(text.into())).await
    }

    /// Drain every transcription frame the reader has decoded so far. The reader
    /// runs ahead of `run_stt`; this never blocks on a round-trip.
    pub fn drain(&mut self) -> Vec<Frame> {
        let mut out = Vec::new();
        if let Some(conn) = self.conn.as_mut() {
            while let Ok(f) = conn.frames.try_recv() {
                out.push(f);
            }
        }
        out
    }
}

/// Borrow the live session or return a uniform "run_stt before start" error.
pub fn require<'a>(
    session: &'a mut Option<WsSttSession>,
    provider: &str,
) -> Result<&'a mut WsSttSession> {
    session
        .as_mut()
        .ok_or_else(|| FlowcatError::Network(format!("{provider}: run_stt before start")))
}

/// PCM `i16` → little-endian bytes for a binary WS frame.
pub fn pcm_le_bytes(audio: &AudioFrame) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(audio.pcm.len() * 2);
    for s in &audio.pcm {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    bytes
}

/// Standard (RFC 4648) base64-encode — for the providers that wrap PCM in a
/// base64 JSON envelope (Gladia, Sarvam). Hand-rolled so the WS features don't
/// have to pull the optional `base64` crate (kept here, the one shared file, so
/// it isn't duplicated per provider).
pub fn base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[((n >> 18) & 63) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 63) as usize] as char);
        if chunk.len() > 1 {
            out.push(ALPHABET[((n >> 6) & 63) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(ALPHABET[(n & 63) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

/// The persistent reader: parse each **text** server frame as JSON, run the
/// provider's pure decoder, and queue the resulting frames. Non-JSON / malformed
/// messages are dropped (never panic); a close/error ends the task.
async fn reader_task(
    mut stream: futures::stream::SplitStream<ClientSocket>,
    tx: mpsc::UnboundedSender<Frame>,
    decode: fn(&Value) -> Vec<Frame>,
) {
    while let Some(msg) = stream.next().await {
        let text = match msg {
            Ok(Message::Text(t)) => t.to_string(),
            Ok(Message::Binary(_)) | Ok(Message::Ping(_)) | Ok(Message::Pong(_)) => continue,
            Ok(Message::Close(_)) | Err(_) => break,
            Ok(_) => continue,
        };
        let Ok(value) = serde_json::from_str::<Value>(&text) else {
            continue; // untrusted server data: ignore malformed JSON
        };
        for frame in decode(&value) {
            if tx.send(frame).is_err() {
                return; // consumer gone
            }
        }
    }
}
