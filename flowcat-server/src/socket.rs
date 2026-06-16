// SPDX-License-Identifier: Apache-2.0
//
//! [`AxumWsSocket`] — a flowcat-core [`MediaSocket`] over an inbound axum WebSocket.
//!
//! flowcat-core's media path is generic over `T: MediaSocket`, so the cleanest
//! server-side adapter is a direct impl over [`axum::extract::ws::WebSocket`] — no
//! tungstenite round-trip (flowcat ships a client-side WS transport; for an
//! inbound carrier WS we adapt the axum socket straight to the trait).
//!
//! ## Why the socket is split (read half here, write half in a writer task)
//!
//! The bidirectional carrier WS is shared (behind one async mutex) between the
//! recv pump and the audio sender. If `recv` and `send` used the *same* socket, a
//! `send` blocked on WS-write **backpressure** (the model's replies are bursty and
//! can fill the write buffer faster than the carrier drains it) would hold the
//! shared lock and **starve the recv pump** — caller audio would stop reaching the
//! model and the realtime session would idle-abort. So we `split()` the socket:
//! `recv` reads the **stream** half, and `send_*` is a **non-blocking enqueue** to a
//! dedicated writer task that owns the **sink** half and absorbs backpressure off
//! the shared lock.

use async_trait::async_trait;
use axum::extract::ws::{Message, WebSocket};
use futures_util::stream::SplitStream;
use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;

use flowcat_core::{FlowcatError, MediaSocket, WsIn};

/// Wraps an accepted axum [`WebSocket`] as a flowcat-core [`MediaSocket`], with the
/// write half driven by a separate task so sends never block reads.
pub struct AxumWsSocket {
    /// Read half — `recv` reads carrier frames here.
    rx: SplitStream<WebSocket>,
    /// Outbound queue drained by the writer task (owns the sink half). Unbounded so
    /// `send_*` is a synchronous enqueue that can't block the shared lock; the
    /// writer task is where real WS-write backpressure is absorbed.
    tx: mpsc::UnboundedSender<Message>,
}

impl AxumWsSocket {
    /// Adapt an already-`accept`-ed axum WebSocket. Spawns the writer task that
    /// drains queued outbound frames into the split sink (requires a Tokio runtime
    /// — always true on the media-WS request path).
    pub fn new(ws: WebSocket) -> Self {
        let (mut sink, rx) = ws.split();
        let (tx, mut out_rx) = mpsc::unbounded_channel::<Message>();
        tokio::spawn(async move {
            // Drain queued frames to the carrier; WS-write backpressure is absorbed
            // here (NOT under the recv/send shared lock). Exits when the queue is
            // closed (socket dropped on call end) or the carrier write fails.
            while let Some(msg) = out_rx.recv().await {
                if sink.send(msg).await.is_err() {
                    break;
                }
            }
            let _ = sink.close().await;
        });
        Self { rx, tx }
    }
}

#[async_trait]
impl MediaSocket for AxumWsSocket {
    async fn recv(&mut self) -> Option<WsIn> {
        // Loop so keepalive frames (Ping/Pong) never surface as actionable media:
        // keep reading until a Text/Binary/Close, a stream error, or end-of-stream.
        loop {
            match self.rx.next().await {
                Some(Ok(Message::Text(t))) => return Some(WsIn::Text(t.as_str().to_owned())),
                Some(Ok(Message::Binary(b))) => return Some(WsIn::Binary(b.to_vec())),
                Some(Ok(Message::Close(_))) => return Some(WsIn::Close),
                // Carriers send pings to keep the socket warm — not media.
                Some(Ok(Message::Ping(_))) | Some(Ok(Message::Pong(_))) => continue,
                // A transport error is terminal for the call: treat it like a peer
                // close so the pipeline tears down + finalizes cleanly.
                Some(Err(_)) => return Some(WsIn::Close),
                None => return None,
            }
        }
    }

    async fn send_text(&mut self, s: String) -> Result<(), FlowcatError> {
        // Non-blocking enqueue; the writer task performs the actual WS write.
        self.tx
            .send(Message::text(s))
            .map_err(|_| FlowcatError::Transport("axum ws send_text: writer task closed".into()))
    }

    async fn send_binary(&mut self, b: Vec<u8>) -> Result<(), FlowcatError> {
        self.tx
            .send(Message::binary(b))
            .map_err(|_| FlowcatError::Transport("axum ws send_binary: writer task closed".into()))
    }
}
