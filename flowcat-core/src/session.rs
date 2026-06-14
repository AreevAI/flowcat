// SPDX-License-Identifier: Apache-2.0
//
//! Call bootstrap + finalize seam.
//!
//! The embedder implements [`SessionSource`] over its own control-plane HTTP
//! API (resolve a run+token into a call config, upload artifacts, write the
//! finalize). flowcat-core only sees the opaque shapes (see DESIGN.md
//! "Trait contracts"). [`ResolvedCall`], [`Finalize`], [`UploadTarget`], and
//! [`Usage`] are defined in [`crate::types`] and re-exported here.

use async_trait::async_trait;

use crate::error::FlowcatError;

// Re-export so callers use `crate::session::{ResolvedCall, Finalize, ...}` as
// the design lays out; the definitions live in `frame`.
pub use crate::types::{Finalize, ResolvedCall, ToolDecl, UploadTarget, Usage};

/// Bootstraps a call from a run id + per-call token and writes results back.
///
/// `Send + Sync` because it is shared across the spawned per-leg tasks of a call.
#[async_trait]
pub trait SessionSource: Send + Sync {
    /// Resolve a run id + per-call token into the call's configuration.
    async fn resolve(&self, run_id: i64, token: &str) -> Result<ResolvedCall, FlowcatError>;

    /// Mark the run complete and persist usage / collected vars / artifact URLs.
    async fn complete(&self, run_id: i64, token: &str, fin: Finalize) -> Result<(), FlowcatError>;

    /// Obtain a (pre-signed) upload target for an artifact (`kind` = recording/transcript/…).
    async fn artifact_upload_url(
        &self,
        run_id: i64,
        token: &str,
        kind: &str,
    ) -> Result<UploadTarget, FlowcatError>;

    /// PUT raw bytes to a (pre-signed) upload URL with the given content type.
    async fn put_bytes(
        &self,
        url: &str,
        bytes: Vec<u8>,
        content_type: &str,
    ) -> Result<(), FlowcatError>;

    /// Fetch the current node's MCP/HTTP **workflow tools** (distinct from the
    /// brain's graph transitions). These are the tools the control plane will
    /// execute on the agent's behalf (see [`SessionSource::tool_call`]).
    ///
    /// Implementations **degrade gracefully**: any HTTP/parse failure returns
    /// `Ok(vec![])` (the call proceeds with no node tools) rather than aborting
    /// the live call. `params` is the tool's JSON-Schema (`input_schema`,
    /// defaulting to an empty object).
    async fn node_tools(
        &self,
        run_id: i64,
        token: &str,
        node_id: &str,
    ) -> Result<Vec<ToolDecl>, FlowcatError>;

    /// Relay a workflow tool call (name + args) to the control plane, which runs
    /// the MCP/HTTP egress and returns the tool result `content` (fed back to the
    /// model). The `is_error` flag from the control plane is folded into the
    /// returned text — the model handles failures conversationally — so this
    /// returns a single string in all cases. On a transport error it returns a
    /// short "temporarily unavailable" message so the call continues.
    async fn tool_call(
        &self,
        run_id: i64,
        token: &str,
        node_id: &str,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> Result<String, FlowcatError>;
}
