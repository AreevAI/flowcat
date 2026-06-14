// SPDX-License-Identifier: Apache-2.0
//
//! The `MediaSocket` trait — Flowcat's transport-agnostic media seam.
//!
//! Flowcat does not own the socket; the host adapts whatever it has (an axum WS,
//! a tungstenite client, a mock) to this trait. See DESIGN.md "Trait contracts".

use async_trait::async_trait;

use crate::error::FlowcatError;
use crate::types::WsIn;

/// A bidirectional media socket carrying text/binary WebSocket frames.
///
/// Implementations are driven (via [`MediaTransport`](crate::transport::MediaTransport))
/// by the [S2S pipeline](crate::pipeline::build_s2s_task): `recv` yields inbound
/// frames; `send_text`/`send_binary` push outbound frames.
#[async_trait]
pub trait MediaSocket: Send {
    /// Receive the next inbound frame, or `None` once the socket is exhausted.
    async fn recv(&mut self) -> Option<WsIn>;

    /// Send a UTF-8 text frame.
    async fn send_text(&mut self, s: String) -> Result<(), FlowcatError>;

    /// Send a binary frame.
    async fn send_binary(&mut self, b: Vec<u8>) -> Result<(), FlowcatError>;
}
