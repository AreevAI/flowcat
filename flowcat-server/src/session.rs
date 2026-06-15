// SPDX-License-Identifier: Apache-2.0
//
//! [`StaticSession`] — a control-plane-free [`SessionSource`].
//!
//! flowcat's runtime resolves each call through a `SessionSource` (fetch the
//! agent config, upload recordings/transcripts, write the finalize). Embedders
//! back that with their control-plane HTTP API — but there was no default, so you
//! could not run a call without writing one. `StaticSession` fills that gap: it
//! serves a single agent from a local config and writes artifacts to a local
//! directory, so any flowcat user can run a real call with no control plane.

use std::path::PathBuf;

use async_trait::async_trait;
use serde_json::{json, Map, Value};
use tracing::info;

use flowcat_core::session::{Finalize, ResolvedCall, ToolDecl, UploadTarget};
use flowcat_core::{FlowcatError, SessionSource};

/// A [`SessionSource`] backed by a single in-memory agent config (no control
/// plane). `resolve` always returns the configured agent; artifacts are written
/// under a local directory; workflow-tool relay is a no-op (declarative graph
/// transitions come from the brain, not the control plane).
pub struct StaticSession {
    /// `{ "graph_spec": …, "runtime_options": {}, "seed_vars": … }`.
    brain_config: Value,
    /// Provider label echoed back in the resolved call.
    provider: String,
    /// Directory artifacts (recording/transcript) are written to.
    artifact_dir: PathBuf,
}

impl StaticSession {
    /// Build from the agent's `graph_spec` + seed variables. `provider` is the
    /// label echoed in [`ResolvedCall::provider`] (e.g. the transport kind).
    pub fn new(
        graph_spec: Value,
        seed_vars: Map<String, Value>,
        provider: impl Into<String>,
    ) -> Self {
        Self {
            brain_config: json!({
                "graph_spec": graph_spec,
                "runtime_options": {},
                "seed_vars": seed_vars,
            }),
            provider: provider.into(),
            artifact_dir: PathBuf::from("flowcat-artifacts"),
        }
    }

    /// Override where artifacts are written (default `./flowcat-artifacts`).
    pub fn with_artifact_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.artifact_dir = dir.into();
        self
    }
}

#[async_trait]
impl SessionSource for StaticSession {
    async fn resolve(&self, _run_id: i64, _token: &str) -> Result<ResolvedCall, FlowcatError> {
        Ok(ResolvedCall {
            provider: self.provider.clone(),
            brain_config: self.brain_config.clone(),
            is_completed: false,
        })
    }

    async fn complete(&self, run_id: i64, _token: &str, fin: Finalize) -> Result<(), FlowcatError> {
        info!(
            run_id,
            usage = %fin.usage,
            collected_vars = %fin.collected_vars,
            recording = ?fin.recording_url,
            transcript = ?fin.transcript_url,
            "static session: run complete"
        );
        Ok(())
    }

    async fn artifact_upload_url(
        &self,
        run_id: i64,
        _token: &str,
        kind: &str,
    ) -> Result<UploadTarget, FlowcatError> {
        std::fs::create_dir_all(&self.artifact_dir)
            .map_err(|e| FlowcatError::Session(format!("create artifact dir: {e}")))?;
        let key = format!("run-{run_id}-{kind}");
        let url = format!("file://{}", self.artifact_dir.join(&key).display());
        let content_type = match kind {
            "recording" => "audio/wav",
            "transcript" => "application/json",
            _ => "application/octet-stream",
        }
        .to_string();
        Ok(UploadTarget {
            url,
            key,
            content_type,
        })
    }

    async fn put_bytes(
        &self,
        url: &str,
        bytes: Vec<u8>,
        _content_type: &str,
    ) -> Result<(), FlowcatError> {
        let path = url.strip_prefix("file://").ok_or_else(|| {
            FlowcatError::Session(format!(
                "static session only supports file:// upload targets, got {url:?}"
            ))
        })?;
        std::fs::write(path, bytes)
            .map_err(|e| FlowcatError::Session(format!("write artifact {path}: {e}")))?;
        Ok(())
    }

    async fn node_tools(
        &self,
        _run_id: i64,
        _token: &str,
        _node_id: &str,
    ) -> Result<Vec<ToolDecl>, FlowcatError> {
        // No control-plane workflow tools in the static session; the declarative
        // graph's transitions are supplied by the brain itself.
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
        Ok(format!(
            "tool {tool_name:?} is not available in the static/local session"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session() -> StaticSession {
        StaticSession::new(
            json!({ "nodes": [{ "id": "s", "type": "startCall" }], "edges": [] }),
            Map::new(),
            "local",
        )
    }

    #[tokio::test]
    async fn resolve_returns_the_configured_agent() {
        let r = session().resolve(1, "tok").await.unwrap();
        assert_eq!(r.provider, "local");
        assert!(!r.is_completed);
        assert_eq!(r.brain_config["graph_spec"]["nodes"][0]["id"], "s");
        assert!(r.brain_config["runtime_options"].is_object());
    }

    #[tokio::test]
    async fn node_tools_are_empty_and_tool_call_is_a_noop_message() {
        let s = session();
        assert!(s.node_tools(1, "t", "s").await.unwrap().is_empty());
        let msg = s
            .tool_call(1, "t", "s", "lookup", &json!({}))
            .await
            .unwrap();
        assert!(msg.contains("not available"), "got: {msg}");
        assert!(msg.contains("lookup"));
    }

    #[tokio::test]
    async fn artifact_round_trips_to_a_local_file() {
        let dir = std::env::temp_dir().join("flowcat-server-session-test");
        let _ = std::fs::remove_dir_all(&dir);
        let s = session().with_artifact_dir(&dir);

        let target = s.artifact_upload_url(7, "tok", "transcript").await.unwrap();
        assert!(target.url.starts_with("file://"));
        assert_eq!(target.key, "run-7-transcript");
        assert_eq!(target.content_type, "application/json");

        s.put_bytes(&target.url, b"hello".to_vec(), &target.content_type)
            .await
            .unwrap();
        let written = std::fs::read(dir.join(&target.key)).unwrap();
        assert_eq!(written, b"hello");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn put_bytes_rejects_non_file_urls() {
        let err = session()
            .put_bytes(
                "https://example.com/x",
                b"x".to_vec(),
                "application/octet-stream",
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("file://"), "got: {err}");
    }
}
