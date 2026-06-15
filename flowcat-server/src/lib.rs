// SPDX-License-Identifier: Apache-2.0
//
//! # flowcat-server
//!
//! A config-driven single-agent server for the Flowcat runtime: point it at a
//! YAML/JSON config describing one agent (graph + topology + transport) and run
//! that agent with **no control plane**.
//!
//! This crate exposes the building blocks the server wires together:
//! - [`config`] — the agent/server config schema + loader ([`ServerConfig`]).
//! - [`session`] — a control-plane-free [`session::StaticSession`] that satisfies
//!   `flowcat_core::SessionSource` from the local config (resolve returns the
//!   configured agent; artifacts are written to a local directory).
//!
//! The server binary + transport wiring (media WebSocket / WebRTC / SIP) build on
//! these and land in a later change.

pub mod config;
pub mod session;

pub use config::{ConfigError, ServerConfig, TopologyConfig};
pub use session::StaticSession;
