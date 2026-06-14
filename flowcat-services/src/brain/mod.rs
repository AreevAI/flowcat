// SPDX-License-Identifier: Apache-2.0
//
//! Remote "brain" adapters — `AgentBrain` impls that drive the conversation
//! policy from an out-of-process service over the network.
//!
//! The conversation decision-maker trait ([`flowcat_core::AgentBrain`]) is
//! synchronous and lives in `flowcat-core`. An embedder normally implements it
//! in-process over its own engine. This module ships a **reference HTTP webhook
//! adapter** ([`remote::RemoteBrain`]) so a Python (or any-language) service can
//! own the policy and be consulted per turn over a small JSON wire contract,
//! without writing any Rust or using in-process bindings.
//!
//! Gated behind the `brain-http` feature (`dep:`-gated on `reqwest` + the
//! multi-threaded `tokio` runtime), so a default build pulls nothing.

pub mod remote;
