// SPDX-License-Identifier: Apache-2.0
//
//! `flowcat-agent` — a declarative, graph-based conversation agent for Flowcat.
//!
//! Define an agent as a node/edge graph (the same `graph_spec` JSON an editor
//! produces) instead of hand-writing an `AgentBrain` in Rust. The [`Engine`] is
//! pure logic: parse the graph, track the active node, compose each turn's system
//! prompt + the callable transitions, interpolate `{{variables}}`, and advance on
//! a transition/tool call.
//!
//! With the `brain` feature (on by default) the crate also ships a
//! `DeclarativeBrain`, a ready-to-use `flowcat_core::AgentBrain` backed by the
//! engine — so a config-driven host can run an agent with no Rust agent code.
//! Turn the feature off (`default-features = false`) to use the pure engine
//! without depending on `flowcat-core`.

#[cfg(feature = "brain")]
pub mod brain;
pub mod engine;
pub mod graph;

pub use engine::{
    validate, validate_detailed, var_interpolate, var_interpolate_value, Action, Engine,
    EngineError, ExtractionSpec, ExtractionVar, NodeInfo, NodePrompt, TransferMode, TransferTarget,
    TransitionSpec, VarType,
};
pub use graph::{Edge, Graph, GraphError, Node, NodeKind, ValidationError};

#[cfg(feature = "brain")]
pub use brain::{DeclarativeBrain, END_CALL_TOOL};
