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

pub mod config;
pub mod run;
pub mod session;

#[cfg(feature = "server")]
pub mod server;
#[cfg(feature = "server")]
pub mod socket;

pub use config::{ConfigError, ServerConfig, TopologyConfig};
pub use session::StaticSession;
