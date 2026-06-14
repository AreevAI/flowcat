// SPDX-License-Identifier: Apache-2.0
//
//! Generic WebSocket media transport.
//!
//! [`WsTransport`] is a [`MediaTransport`](flowcat_core::MediaTransport) over a
//! raw bidirectional WebSocket. It is the **simplest** transport: inbound binary
//! frames are interpreted as little-endian 16-bit mono PCM at a fixed
//! `carrier_rate` and surfaced as [`MediaIn::Audio`]; outbound
//! [`AudioChunk`](flowcat_core::types::AudioChunk)s are written as binary PCM
//! frames. It is useful for tests and for non-WebRTC WS media bridges.
//!
//! This is the *generic, non-carrier* WS path — carrier-specific JSON framing
//! (Plivo `start`/`media`/`stop`, base64 μ-law, …) lives in `flowcat-telephony`
//! behind its [`MediaSerializer`](flowcat_core::MediaSerializer). Behind the
//! `ws` feature (pulls `tokio-tungstenite`).
//!
//! ## Framing
//!
//! - **Binary frame** → little-endian i16 mono PCM at `carrier_rate` →
//!   [`MediaIn::Audio`]. An odd-length binary frame (not a whole number of i16
//!   samples) drops its trailing byte (defensive — never panics).
//! - **Text frame** → ignored (a generic control message Flowcat doesn't model
//!   on this path); the loop keeps reading.
//! - **Close / stream end / transport error** → [`MediaIn::Stop`].
//!
//! The transport is generic over any tungstenite-`Message` stream/sink so a host
//! can plug in a connected client socket (via [`WsTransport::connect`]) or an
//! adapted server socket (via [`WsTransport::new`]), mirroring
//! `flowcat_core::transport::WsMediaTransport`.

use std::fmt::Display;

use async_trait::async_trait;
use futures::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;

use flowcat_core::error::FlowcatError;
use flowcat_core::transport::{MediaIn, MediaTransport};
use flowcat_core::types::AudioChunk;

/// A generic WebSocket [`MediaTransport`] carrying raw L16 mono PCM.
///
/// `S` is the underlying duplex socket — generic so both a connected
/// `tokio_tungstenite::WebSocketStream` (client) and a host-adapted server
/// socket satisfy `Stream<Item = Result<Message, E>> + Sink<Message>`.
pub struct WsTransport<S> {
    socket: S,
    carrier_rate: u32,
    /// `StreamStart` is emitted once, before any audio.
    started: bool,
    /// The carrier-side call id surfaced in `StreamStart`.
    call_id: String,
}

impl<S> WsTransport<S> {
    /// Wrap an already-established duplex WS stream/sink (server path / mocks).
    ///
    /// `carrier_rate` is the sample rate inbound PCM is tagged with and outbound
    /// PCM is expected at. `call_id` is surfaced once as
    /// [`MediaIn::StreamStart`].
    pub fn new(socket: S, carrier_rate: u32, call_id: impl Into<String>) -> Self {
        Self {
            socket,
            carrier_rate,
            started: false,
            call_id: call_id.into(),
        }
    }

    /// Consume the transport and return the underlying socket.
    pub fn into_inner(self) -> S {
        self.socket
    }
}

/// The concrete socket type produced by [`tokio_tungstenite::connect_async`].
type ClientSocket =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

impl WsTransport<ClientSocket> {
    /// Connect to `url` (`ws://` / `wss://`) and wrap the socket (client path).
    pub async fn connect(
        url: &str,
        carrier_rate: u32,
        call_id: impl Into<String>,
    ) -> Result<Self, FlowcatError> {
        let (socket, _resp) = tokio_tungstenite::connect_async(url)
            .await
            .map_err(|e| FlowcatError::Transport(format!("ws connect_async({url}): {e}")))?;
        Ok(Self::new(socket, carrier_rate, call_id))
    }
}

/// Decode a binary frame of little-endian i16 mono PCM into samples. A trailing
/// odd byte (not a whole sample) is dropped rather than panicking.
fn bytes_to_pcm16(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Encode i16 mono PCM into a little-endian byte buffer for a binary WS frame.
fn pcm16_to_bytes(pcm: &[i16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(pcm.len() * 2);
    for &s in pcm {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

#[async_trait]
impl<S, E> MediaTransport for WsTransport<S>
where
    S: futures::Stream<Item = Result<Message, E>>
        + futures::Sink<Message, Error = E>
        + Unpin
        + Send,
    E: Display + Send,
{
    async fn recv(&mut self) -> Option<MediaIn> {
        // Emit StreamStart exactly once, before any audio.
        if !self.started {
            self.started = true;
            return Some(MediaIn::StreamStart {
                call_id: self.call_id.clone(),
            });
        }

        // Loop so keepalives / unmodelled Text frames don't surface as events.
        loop {
            match self.socket.next().await {
                Some(Ok(Message::Binary(b))) => {
                    let pcm = bytes_to_pcm16(&b);
                    // An empty/odd frame yields no samples — keep reading rather
                    // than emit an empty Audio event.
                    if pcm.is_empty() {
                        continue;
                    }
                    return Some(MediaIn::Audio(AudioChunk::new(pcm, self.carrier_rate)));
                }
                // Generic WS text control frames are not modelled on this path.
                Some(Ok(Message::Text(_))) => continue,
                Some(Ok(Message::Close(_))) => return Some(MediaIn::Stop),
                // Keepalives + raw frames carry no media; ignore.
                Some(Ok(Message::Ping(_)))
                | Some(Ok(Message::Pong(_)))
                | Some(Ok(Message::Frame(_))) => continue,
                // A transport error is terminal: treat like a peer close.
                Some(Err(_)) => return Some(MediaIn::Stop),
                None => return Some(MediaIn::Stop),
            }
        }
    }

    async fn send_audio(&mut self, chunk: AudioChunk) -> Result<(), FlowcatError> {
        if chunk.sample_rate != self.carrier_rate {
            return Err(FlowcatError::Codec(format!(
                "ws send_audio: expected carrier rate {} Hz, got {} Hz",
                self.carrier_rate, chunk.sample_rate
            )));
        }
        let bytes = pcm16_to_bytes(&chunk.pcm);
        self.socket
            .send(Message::binary(bytes))
            .await
            .map_err(|e| FlowcatError::Transport(format!("ws send_audio: {e}")))
    }

    async fn send_clear(&mut self) -> Result<(), FlowcatError> {
        // A raw WS has no carrier-side playout buffer to flush; barge-in upstream
        // simply stops producing audio. No-op.
        Ok(())
    }

    fn carrier_rate(&self) -> u32 {
        self.carrier_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use std::collections::VecDeque;
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};
    use std::task::{Context, Poll};

    /// A duplex mock: a canned inbound stream + an outbound capture buffer.
    /// Mirrors the `flowcat_core::transport::ws_media` test mock.
    struct MockSocket {
        inbound: stream::Iter<std::vec::IntoIter<Result<Message, String>>>,
        sent: Arc<Mutex<VecDeque<Message>>>,
    }

    impl MockSocket {
        fn new(inbound: Vec<Result<Message, String>>) -> (Self, Arc<Mutex<VecDeque<Message>>>) {
            let sent = Arc::new(Mutex::new(VecDeque::new()));
            (
                Self {
                    inbound: stream::iter(inbound),
                    sent: sent.clone(),
                },
                sent,
            )
        }
    }

    impl futures::Stream for MockSocket {
        type Item = Result<Message, String>;
        fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            Pin::new(&mut self.inbound).poll_next(cx)
        }
    }

    impl futures::Sink<Message> for MockSocket {
        type Error = String;
        fn poll_ready(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<(), String>> {
            Poll::Ready(Ok(()))
        }
        fn start_send(self: Pin<&mut Self>, item: Message) -> Result<(), String> {
            self.sent.lock().unwrap().push_back(item);
            Ok(())
        }
        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<(), String>> {
            Poll::Ready(Ok(()))
        }
        fn poll_close(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Result<(), String>> {
            Poll::Ready(Ok(()))
        }
    }

    /// StreamStart first, then binary PCM frames decode to Audio at the carrier
    /// rate, text/keepalives are skipped, and Close → Stop.
    #[tokio::test]
    async fn recv_emits_streamstart_then_audio_then_stop() {
        let pcm = vec![0i16, 1, -1, 1000, -1000, 32767, -32768];
        let bytes = pcm16_to_bytes(&pcm);
        let inbound = vec![
            Ok(Message::Ping(bytes::Bytes::from_static(b"hb"))), // skipped
            Ok(Message::text("control")),                        // skipped (unmodelled)
            Ok(Message::binary(bytes.clone())),
            Ok(Message::Close(None)),
        ];
        let (mock, _sent) = MockSocket::new(inbound);
        let mut t = WsTransport::new(mock, 16000, "ws-call-1");

        // StreamStart{call_id} first.
        match t.recv().await {
            Some(MediaIn::StreamStart { call_id }) => assert_eq!(call_id, "ws-call-1"),
            other => panic!("expected StreamStart, got {other:?}"),
        }
        // Then the binary frame decodes back to our exact PCM @ 16 kHz.
        match t.recv().await {
            Some(MediaIn::Audio(chunk)) => {
                assert_eq!(chunk.sample_rate, 16000);
                assert_eq!(chunk.pcm, pcm);
            }
            other => panic!("expected Audio, got {other:?}"),
        }
        // Then Close → Stop.
        assert_eq!(t.recv().await, Some(MediaIn::Stop));
    }

    /// An odd-length binary frame drops its trailing byte rather than panicking,
    /// and an empty frame is skipped.
    #[tokio::test]
    async fn odd_and_empty_binary_frames_are_handled_safely() {
        let inbound = vec![
            Ok(Message::binary(vec![])),                 // empty → skipped
            Ok(Message::binary(vec![0x01, 0x02, 0x03])), // 3 bytes → one sample (0x0201), drop 0x03
            Ok(Message::Close(None)),
        ];
        let (mock, _sent) = MockSocket::new(inbound);
        let mut t = WsTransport::new(mock, 8000, "ws-call-2");
        assert!(matches!(t.recv().await, Some(MediaIn::StreamStart { .. })));
        match t.recv().await {
            Some(MediaIn::Audio(chunk)) => {
                assert_eq!(chunk.pcm, vec![i16::from_le_bytes([0x01, 0x02])]);
            }
            other => panic!("expected one-sample Audio, got {other:?}"),
        }
        assert_eq!(t.recv().await, Some(MediaIn::Stop));
    }

    /// A stream error is treated as a peer close (Stop).
    #[tokio::test]
    async fn stream_error_surfaces_as_stop() {
        let inbound = vec![Err("boom".to_string()), Ok(Message::text("unreached"))];
        let (mock, _sent) = MockSocket::new(inbound);
        let mut t = WsTransport::new(mock, 16000, "ws-call-3");
        assert!(matches!(t.recv().await, Some(MediaIn::StreamStart { .. })));
        assert_eq!(t.recv().await, Some(MediaIn::Stop));
    }

    /// send_audio writes a binary PCM frame that round-trips back to the input
    /// PCM; a wrong-rate chunk is rejected.
    #[tokio::test]
    async fn send_audio_emits_binary_pcm_and_rejects_wrong_rate() {
        let (mock, sent) = MockSocket::new(vec![]);
        let mut t = WsTransport::new(mock, 16000, "ws-call-4");

        let pcm = vec![100i16, -100, 2000, -2000];
        t.send_audio(AudioChunk::new(pcm.clone(), 16000))
            .await
            .unwrap();

        // Inspect the captured frame in a scope that drops the guard before the
        // next await (no MutexGuard held across `.await`).
        {
            let q = sent.lock().unwrap();
            match &q[0] {
                Message::Binary(b) => assert_eq!(bytes_to_pcm16(b), pcm),
                other => panic!("expected Binary, got {other:?}"),
            }
        }

        // Wrong rate → Codec error, no panic.
        let err = t
            .send_audio(AudioChunk::new(vec![0i16; 10], 8000))
            .await
            .unwrap_err();
        assert!(matches!(err, FlowcatError::Codec(_)));
    }

    #[test]
    fn pcm_byte_roundtrip_is_lossless() {
        let pcm = vec![0i16, 1, -1, i16::MAX, i16::MIN, 12345, -12345];
        assert_eq!(bytes_to_pcm16(&pcm16_to_bytes(&pcm)), pcm);
    }
}
