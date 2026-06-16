// SPDX-License-Identifier: Apache-2.0
//
//! Call bootstrap + finalize seam.
//!
//! The embedder implements [`SessionSource`] over its own control-plane HTTP
//! API (resolve a run+token into a call config, upload artifacts, write the
//! finalize). flowcat-core only sees the opaque shapes (see DESIGN.md
//! "Trait contracts"). [`ResolvedCall`], [`Finalize`], [`UploadTarget`], and
//! [`Usage`] are defined in [`crate::types`] and re-exported here.

use std::sync::Arc;

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

/// Blanket impl for any `Arc`-wrapped source (including `Arc<dyn SessionSource>`).
///
/// A real deployment holds **one** session (its control-plane HTTP client, pools,
/// caches) and reuses it across every call. The per-call pipeline builders take a
/// `SessionSource` *by value*, so this impl lets a shared `Arc<S>` be cloned cheaply
/// per call and passed in without forcing the concrete source to be `Clone` — and,
/// via `?Sized`, lets an embedder erase the source to `Arc<dyn SessionSource>`.
#[async_trait]
impl<T: SessionSource + ?Sized> SessionSource for Arc<T> {
    async fn resolve(&self, run_id: i64, token: &str) -> Result<ResolvedCall, FlowcatError> {
        (**self).resolve(run_id, token).await
    }

    async fn complete(&self, run_id: i64, token: &str, fin: Finalize) -> Result<(), FlowcatError> {
        (**self).complete(run_id, token, fin).await
    }

    async fn artifact_upload_url(
        &self,
        run_id: i64,
        token: &str,
        kind: &str,
    ) -> Result<UploadTarget, FlowcatError> {
        (**self).artifact_upload_url(run_id, token, kind).await
    }

    async fn put_bytes(
        &self,
        url: &str,
        bytes: Vec<u8>,
        content_type: &str,
    ) -> Result<(), FlowcatError> {
        (**self).put_bytes(url, bytes, content_type).await
    }

    async fn node_tools(
        &self,
        run_id: i64,
        token: &str,
        node_id: &str,
    ) -> Result<Vec<ToolDecl>, FlowcatError> {
        (**self).node_tools(run_id, token, node_id).await
    }

    async fn tool_call(
        &self,
        run_id: i64,
        token: &str,
        node_id: &str,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> Result<String, FlowcatError> {
        (**self)
            .tool_call(run_id, token, node_id, tool_name, args)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    /// Minimal source that records how many times `resolve` was called and echoes a
    /// fixed provider, so the blanket `Arc` impl can be observed delegating.
    struct CountingSource {
        calls: std::sync::atomic::AtomicUsize,
    }

    #[async_trait]
    impl SessionSource for CountingSource {
        async fn resolve(&self, _run_id: i64, _token: &str) -> Result<ResolvedCall, FlowcatError> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(ResolvedCall {
                provider: "counting".to_string(),
                brain_config: json!({ "graph_spec": {} }),
                is_completed: false,
            })
        }
        async fn complete(
            &self,
            _run_id: i64,
            _token: &str,
            _fin: Finalize,
        ) -> Result<(), FlowcatError> {
            Ok(())
        }
        async fn artifact_upload_url(
            &self,
            _run_id: i64,
            _token: &str,
            _kind: &str,
        ) -> Result<UploadTarget, FlowcatError> {
            Err(FlowcatError::Session("not supported".into()))
        }
        async fn put_bytes(
            &self,
            _url: &str,
            _bytes: Vec<u8>,
            _content_type: &str,
        ) -> Result<(), FlowcatError> {
            Ok(())
        }
        async fn node_tools(
            &self,
            _run_id: i64,
            _token: &str,
            _node_id: &str,
        ) -> Result<Vec<ToolDecl>, FlowcatError> {
            Ok(vec![])
        }
        async fn tool_call(
            &self,
            _run_id: i64,
            _token: &str,
            _node_id: &str,
            tool_name: &str,
            _args: &Value,
        ) -> Result<String, FlowcatError> {
            Ok(tool_name.to_string())
        }
    }

    // Generic over `SessionSource` by value — the call site that motivates the
    // blanket impl (e.g. the per-call pipeline builders).
    async fn resolve_via<S: SessionSource>(s: S) -> ResolvedCall {
        s.resolve(7, "tok").await.unwrap()
    }

    #[tokio::test]
    async fn arc_delegates_and_satisfies_session_source_by_value() {
        let inner = Arc::new(CountingSource {
            calls: std::sync::atomic::AtomicUsize::new(0),
        });
        // Concrete `Arc<S>`: clone it per "call" and pass each by value.
        assert_eq!(resolve_via(inner.clone()).await.provider, "counting");
        assert_eq!(resolve_via(inner.clone()).await.provider, "counting");
        // Type-erased `Arc<dyn SessionSource>` works through the same impl.
        let erased: Arc<dyn SessionSource> = inner.clone();
        assert_eq!(resolve_via(erased).await.provider, "counting");
        assert_eq!(inner.calls.load(std::sync::atomic::Ordering::SeqCst), 3);
    }
}
