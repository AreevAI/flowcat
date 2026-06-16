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
//! With the `server` Cargo feature, the crate also builds the **binary**: an axum
//! HTTP server (health, the Plivo media WebSocket, the Plivo answer XML) that runs
//! the configured agent. The default (no-feature) build is library-only and pulls
//! no axum/tokio.
//!
//! ## A reusable framework, not just a binary
//!
//! With the `server` feature, the HTTP/transport surface (`build_router` over
//! `AppState`) is **generic over the embedder's `SessionSource` and `AgentBrain`**.
//! The zero-config default (a [`StaticSession`] resolving one configured agent plus
//! a `DeclarativeBrain` factory) is built by `AppState::new` and powers the
//! playground and `--config agent.yaml`. A platform with its own control plane
//! injects its own session and per-call `BrainFactory` via `AppState::with_parts`,
//! reusing the same router, media WebSocket, and live-events WS with no fork of the
//! axum bootstrap.

pub mod config;
pub mod run;
pub mod session;

#[cfg(feature = "server")]
pub mod server;
#[cfg(feature = "server")]
pub mod socket;

#[cfg(feature = "webrtc")]
pub mod events;
#[cfg(feature = "webrtc")]
pub mod webrtc;

pub use config::{ConfigError, ServerConfig, TopologyConfig};
pub use run::{env_spec_resolver, run_call, run_call_with, SpecResolver};
pub use session::StaticSession;

#[cfg(feature = "server")]
pub use server::{build_router, AppState, BrainFactory};
