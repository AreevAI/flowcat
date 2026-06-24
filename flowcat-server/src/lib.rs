// SPDX-License-Identifier: Apache-2.0
//
//! # flowcat-server
//!
//! A config-driven single-agent server for the Flowcat runtime: point it at a
//! YAML/JSON config describing one agent (graph + topology + transport) and run
//! that agent with **no control plane**.
//!
//! - [`config`] — the agent/server config schema + loader ([`ServerConfig`]).
//! - [`session`] — a control-plane-free [`session::StaticSession`].
//! - [`run`] — assemble + run the flowcat pipeline for one call from the config
//!   topology (realtime or cascaded), resolving provider keys from the env.
//!
//! The axum HTTP/transport framework — `build_router`, `AppState`, the media WS, and
//! (with WebRTC) the reusable [`webrtc::handle_offer`] signaling helper — is gated
//! behind the **`server-helper`** feature; the browser playground surface adds
//! **`webrtc-helper`**. Neither helper pulls a `flowcat-services/*-all` bundle, so an
//! embedder reuses this surface with its OWN curated connector set (the factory
//! unifies in whatever `flowcat-services/*` features the embedder enables). The
//! standalone **binary** is built by the **`server`** feature (`server-helper` + every
//! connector bundle) — or **`webrtc`** (`webrtc-helper` + `server`) for the full
//! browser playground binary. The default (no-feature) build is library-only and
//! pulls no axum/tokio.
//!
//! ## A reusable framework, not just a binary
//!
//! With the `server-helper` feature, the HTTP/transport surface (`build_router` over
//! `AppState`) is **generic over the embedder's `SessionSource` and `AgentBrain`**.
//! The zero-config default (a [`StaticSession`] resolving one configured agent plus
//! a `DeclarativeBrain` factory) is built by `AppState::new` and powers the
//! playground and `--config agent.yaml`. A platform with its own control plane
//! injects its own session and per-call `BrainFactory` via `AppState::with_parts`,
//! reusing the same router, media WebSocket, and live-events WS with no fork of the
//! axum bootstrap.
//!
//! For the **native SIP/RTP** path (no carrier WS), the [`sip`] module's
//! `serve_sip_inbound` + `sip_originate` are the analogue of `media_ws` +
//! `run_call_with`: the same `SessionSource` + `BrainFactory` + topology wiring drives
//! the inbound-INVITE pump and the outbound originate over a `SipAgent`, with the
//! dialed-identifier→run mapping behind the embedder's `SipInboundResolver`.

pub mod config;
pub mod run;
pub mod session;

#[cfg(feature = "server-helper")]
pub mod server;
#[cfg(feature = "server-helper")]
pub mod sip;
#[cfg(feature = "server-helper")]
pub mod socket;

#[cfg(feature = "webrtc-helper")]
pub mod events;
#[cfg(feature = "webrtc-helper")]
pub mod webrtc;

pub use config::{ConfigError, ServerConfig, TopologyConfig};
pub use run::{env_spec_resolver, run_call, run_call_with, SpecResolver};
pub use session::StaticSession;

#[cfg(feature = "server-helper")]
pub use server::{build_router, AppState, BrainFactory};

// The generic SIP inbound/originate orchestration — the SIP analogue of the
// carrier media-WS path (`media_ws` + `run_call_with`).
#[cfg(feature = "server-helper")]
pub use sip::{
    serve_sip_inbound, sip_originate, SipInboundResolver, SipOrchestrator, SipRun, TopologyResolver,
};

#[cfg(feature = "webrtc-helper")]
pub use webrtc::{handle_offer, OfferParams};

// The live-events surface: the publish side (register/publish per call) plus the
// receiver/stream half an embedder serving its OWN gated events WS needs
// ([`EventRegistry::take_receiver`] + [`stream_events`]). flowcat-server's own
// un-gated [`events::events_ws`] route stays the standalone-server convenience.
#[cfg(feature = "webrtc-helper")]
pub use events::{stream_events, CallEvents, EventRegistry, RegistryGuard, RtfSink};
