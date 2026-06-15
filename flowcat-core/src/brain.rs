// SPDX-License-Identifier: Apache-2.0
//
//! The conversation decision-maker seam.
//!
//! The embedder implements [`AgentBrain`] over its own engine; flowcat-core
//! ships a demo brain. The brain owns the system prompt + tool set for the
//! current state and decides what a tool call means (see DESIGN.md
//! "Trait contracts").
//!
//! [`BrainAction`] and [`ToolDecl`] are defined in [`crate::types`] (the shared
//! data-type module) and re-exported here, which is their contract location.

use serde_json::Value;

// Re-export so callers can use `crate::brain::{BrainAction, ToolDecl}` as the
// design lays out, while the type definitions live in `frame`.
pub use crate::types::{BrainAction, ToolDecl};

/// Drives the conversation: provides the active prompt + tools, interprets tool
/// calls into [`BrainAction`]s, and reports completion + collected variables.
///
/// Synchronous on purpose — the brain is pure decision logic; any I/O it needs
/// (e.g. the embedder's engine) is done by the implementor before/around these calls.
pub trait AgentBrain: Send {
    /// The system prompt for the current conversation state.
    fn system_prompt(&self) -> String;

    /// Tools available in the current state (transitions + endCall, + later node tools).
    fn tools(&self) -> Vec<ToolDecl>;

    /// The id of the brain's current conversation node. The pipeline uses this to
    /// scope the control-plane lookup of the node's MCP/HTTP workflow tools
    /// (`SessionSource::node_tools` / `tool_call`).
    fn current_node_id(&self) -> String;

    /// The display name of the brain's current conversation node. Used to label
    /// the transition marker written into the transcript so the stored/historical
    /// view shows the node name (e.g. "Conversation") rather than the internal
    /// transition slug. Defaults to [`current_node_id`](Self::current_node_id) for
    /// brains without a separate display name.
    fn current_node_name(&self) -> String {
        self.current_node_id()
    }

    /// Interpret a tool/function call from the model into a [`BrainAction`].
    fn on_tool_call(&mut self, name: &str, args: &Value) -> BrainAction;

    /// Whether the conversation has reached a terminal state.
    fn is_finished(&self) -> bool;

    /// The variables collected so far (folded into the finalize payload).
    fn collected_vars(&self) -> Value;
}
