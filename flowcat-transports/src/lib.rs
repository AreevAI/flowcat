// SPDX-License-Identifier: Apache-2.0
//
//! # flowcat-transports
//!
//! Concrete media **transports** for the Flowcat runtime: str0m WebRTC + Opus,
//! a WebSocket transport, Daily/LiveKit, a local mic/speaker, and avatar
//! transports.
//!
//! The transport-agnostic seams —
//! [`MediaTransport`](flowcat_core::MediaTransport),
//! [`MediaSerializer`](flowcat_core::MediaSerializer), and the
//! `TransportInput`/`TransportOutput` processors — stay frozen in flowcat-core.
//! Each transport here implements them and composes the
//! [`SourcePump`](flowcat_core::SourcePump) helper for its input (read) leg.
//! flowcat-core never depends back on this crate.
//!
//! Every transport is behind its own `dep:`-gated Cargo feature, so a default
//! build compiles only these stubs — no `str0m`/`audiopus`/`tokio-tungstenite`/
//! `reqwest` is pulled until a feature is enabled.

/// str0m WebRTC offer/answer + Opus. Behind `webrtc-str0m`.
#[cfg(feature = "webrtc-str0m")]
pub mod webrtc;

/// The str0m WebRTC transport + its signaling peer + Opus codec (convenience
/// re-exports so callers can `use flowcat_transports::WebRtcTransport`).
#[cfg(feature = "webrtc-str0m")]
pub use webrtc::{
    opus::{OpusDecoder, OpusEncoder},
    signaling::{PeerChannels, PeerEvent, WebRtcPeer},
    WebRtcTransport,
};

/// WebSocket transport. Behind `ws`.
#[cfg(feature = "ws")]
pub mod ws;

/// The generic WebSocket transport (convenience re-export).
#[cfg(feature = "ws")]
pub use ws::WsTransport;

/// The (stub) local mic/speaker transport (convenience re-export).
#[cfg(feature = "local")]
pub use local::LocalTransport;

/// Daily transport. Behind `daily`.
#[cfg(feature = "daily")]
pub mod daily;

/// LiveKit transport. Behind `livekit`.
#[cfg(feature = "livekit")]
pub mod livekit;

/// Local mic/speaker transport. Behind `local`.
#[cfg(feature = "local")]
pub mod local;

/// Avatar transports (Tavus/HeyGen/Simli/LemonSlice). Always-present doc-only
/// home; submodules add their `dep:`-gated features when implemented.
pub mod avatar;
