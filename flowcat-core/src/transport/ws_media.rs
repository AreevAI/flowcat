// SPDX-License-Identifier: Apache-2.0
//
//! `WsMediaTransport` — a [`MediaSocket`] over a `tokio-tungstenite` WebSocket.
//!
//! Flowcat does not own the socket: it is **generic over any tungstenite-message
//! stream/sink** so a host can plug in whatever it has. The standalone client
//! path ([`WsMediaTransport::connect`]) uses `tokio_tungstenite::connect_async`;
//! an embedder can adapt its own inbound (e.g. axum) WS into the same
//! `Stream<Item = Result<Message, _>> + Sink<Message>` shape via a thin wrapper
//! and construct a `WsMediaTransport` over that. See DESIGN.md "Crate layout".

use std::fmt::Display;

use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;

use crate::error::FlowcatError;
use crate::transport::socket::MediaSocket;
use crate::types::WsIn;

/// A [`MediaSocket`] backed by any tungstenite-`Message` WebSocket stream/sink.
///
/// `S` is the underlying duplex socket. It is intentionally generic so Flowcat
/// stays host-agnostic: both a `tokio_tungstenite::WebSocketStream` (client) and
/// an axum-WS adapter (server) satisfy
/// `Stream<Item = Result<Message, E>> + Sink<Message, Error = E>`.
///
/// `WebSocketStream` uses one error type (`tungstenite::Error`) for *both* the
/// `Stream` and the `Sink`, so a single `E` bound is sufficient and keeps the
/// `From`/error handling simple.
pub struct WsMediaTransport<S> {
    socket: S,
}

impl<S> WsMediaTransport<S> {
    /// Wrap an already-established duplex WS stream/sink (server path, mocks,
    /// or any pre-built socket) as a `MediaSocket`.
    pub fn new(socket: S) -> Self {
        Self { socket }
    }

    /// Consume the transport and return the underlying socket.
    pub fn into_inner(self) -> S {
        self.socket
    }
}

/// The concrete socket type produced by [`tokio_tungstenite::connect_async`].
type ClientSocket =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

impl WsMediaTransport<ClientSocket> {
    /// Connect to `url` (`ws://` or `wss://`) and wrap the socket (client path).
    ///
    /// This is the standalone/CLI path; servers should construct via
    /// [`WsMediaTransport::new`] over their own adapted socket instead.
    pub async fn connect(url: &str) -> Result<Self, FlowcatError> {
        let (socket, _resp) = tokio_tungstenite::connect_async(url)
            .await
            .map_err(|e| FlowcatError::Transport(format!("connect_async({url}): {e}")))?;
        Ok(Self::new(socket))
    }
}

#[async_trait]
impl<S, E> MediaSocket for WsMediaTransport<S>
where
    S: futures_util::Stream<Item = Result<Message, E>>
        + futures_util::Sink<Message, Error = E>
        + Unpin
        + Send,
    E: Display + Send,
{
    async fn recv(&mut self) -> Option<WsIn> {
        // Loop so control frames (Ping/Pong) and the rare protocol `Frame` don't
        // surface as actionable inbound events — keep reading until we hit a
        // Text/Binary/Close, a stream error, or end-of-stream.
        loop {
            match self.socket.next().await {
                Some(Ok(Message::Text(t))) => return Some(WsIn::Text(t.as_str().to_owned())),
                Some(Ok(Message::Binary(b))) => return Some(WsIn::Binary(b.to_vec())),
                Some(Ok(Message::Close(_))) => return Some(WsIn::Close),
                // Keepalives + raw frames carry no media for Flowcat; ignore.
                Some(Ok(Message::Ping(_)))
                | Some(Ok(Message::Pong(_)))
                | Some(Ok(Message::Frame(_))) => continue,
                // A transport error is terminal for the call: treat like a peer
                // close so the pipeline finalizes cleanly.
                Some(Err(_)) => return Some(WsIn::Close),
                None => return None,
            }
        }
    }

    async fn send_text(&mut self, s: String) -> Result<(), FlowcatError> {
        self.socket
            .send(Message::text(s))
            .await
            .map_err(|e| FlowcatError::Transport(format!("send_text: {e}")))
    }

    async fn send_binary(&mut self, b: Vec<u8>) -> Result<(), FlowcatError> {
        self.socket
            .send(Message::binary(b))
            .await
            .map_err(|e| FlowcatError::Transport(format!("send_binary: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::stream;
    use std::collections::VecDeque;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    /// A tiny duplex mock: a canned inbound stream + an outbound capture buffer.
    /// Implements `Stream<Item = Result<Message, String>>` and
    /// `Sink<Message, Error = String>` so it matches the `MediaSocket` bounds.
    struct MockSocket {
        inbound: stream::Iter<std::vec::IntoIter<Result<Message, String>>>,
        sent: std::sync::Arc<std::sync::Mutex<VecDeque<Message>>>,
    }

    impl MockSocket {
        fn new(
            inbound: Vec<Result<Message, String>>,
        ) -> (Self, std::sync::Arc<std::sync::Mutex<VecDeque<Message>>>) {
            let sent = std::sync::Arc::new(std::sync::Mutex::new(VecDeque::new()));
            (
                Self {
                    inbound: stream::iter(inbound),
                    sent: sent.clone(),
                },
                sent,
            )
        }
    }

    impl futures_util::Stream for MockSocket {
        type Item = Result<Message, String>;
        fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            Pin::new(&mut self.inbound).poll_next(cx)
        }
    }

    impl futures_util::Sink<Message> for MockSocket {
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

    #[tokio::test]
    async fn recv_maps_text_binary_close_and_skips_keepalives() {
        let inbound = vec![
            Ok(Message::Ping(bytes::Bytes::from_static(b"hb"))), // skipped
            Ok(Message::text("hello")),
            Ok(Message::Pong(bytes::Bytes::new())), // skipped
            Ok(Message::binary(vec![1u8, 2, 3])),
            Ok(Message::Close(None)),
        ];
        let (mock, _sent) = MockSocket::new(inbound);
        let mut t = WsMediaTransport::new(mock);

        assert_eq!(t.recv().await, Some(WsIn::Text("hello".to_string())));
        assert_eq!(t.recv().await, Some(WsIn::Binary(vec![1, 2, 3])));
        assert_eq!(t.recv().await, Some(WsIn::Close));
        // After the canned stream is exhausted, recv yields None.
        assert_eq!(t.recv().await, None);
    }

    #[tokio::test]
    async fn recv_treats_stream_error_as_close() {
        let inbound = vec![Err("boom".to_string()), Ok(Message::text("unreached"))];
        let (mock, _sent) = MockSocket::new(inbound);
        let mut t = WsMediaTransport::new(mock);
        assert_eq!(t.recv().await, Some(WsIn::Close));
    }

    #[tokio::test]
    async fn send_text_and_binary_emit_matching_messages() {
        let (mock, sent) = MockSocket::new(vec![]);
        let mut t = WsMediaTransport::new(mock);

        t.send_text("ping".to_string()).await.unwrap();
        t.send_binary(vec![9, 8, 7]).await.unwrap();

        let q = sent.lock().unwrap();
        match &q[0] {
            Message::Text(s) => assert_eq!(s.as_str(), "ping"),
            other => panic!("expected Text, got {other:?}"),
        }
        match &q[1] {
            Message::Binary(b) => assert_eq!(b.as_ref(), &[9u8, 8, 7]),
            other => panic!("expected Binary, got {other:?}"),
        }
    }
}
