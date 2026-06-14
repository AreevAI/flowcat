// SPDX-License-Identifier: Apache-2.0
//
//! **Google Vertex AI** (Gemini) LLM — the Vertex `streamGenerateContent` client.
//!
//! Speaks the **identical** Gemini wire format as [`GoogleLlm`](super::GoogleLlm) — it
//! reuses that module's [`gemini_request_body`](super::google::gemini_request_body)
//! body builder and [`sse_to_frames`](super::google::sse_to_frames) decoder — and
//! differs only in the **surface**:
//!
//! - **Host/path:** a regional `{location}-aiplatform.googleapis.com` (or the global
//!   `aiplatform.googleapis.com` when `location = "global"`) and a
//!   `projects/{project}/locations/{location}/publishers/google/models/{model}` path,
//!   versus AI Studio's `generativelanguage.googleapis.com/v1beta/models/{model}`.
//! - **Auth:** OAuth2 `Authorization: Bearer <access_token>`, versus AI Studio's
//!   `x-goog-api-key`.
//!
//! The bearer token is an **operator-supplied** GCP OAuth2 access token (e.g. from
//! `gcloud auth print-access-token`, or minted from a service account out of band) —
//! config, never request-derived, so there is no request-controlled SSRF surface.
//! Vertex tokens are short-lived; the operator is responsible for supplying a fresh
//! one. Behind the `llm-google-vertex` feature (which enables `llm-google`).

use async_trait::async_trait;
use futures::stream::BoxStream;

use flowcat_core::error::{FlowcatError, Result};
use flowcat_core::processor::frame::{Frame, LlmContext, StartParams};
use flowcat_core::service::{LlmService, Tool};

use super::google::{gemini_request_body, sse_to_frames};

/// Default Vertex location when none is configured.
pub const VERTEX_DEFAULT_LOCATION: &str = "us-central1";
/// Default Vertex Gemini model.
const DEFAULT_MODEL: &str = "gemini-2.5-flash";

/// Google Vertex AI (Gemini) LLM service.
pub struct GoogleVertexLlm {
    http: reqwest::Client,
    /// OAuth2 access token (operator-supplied; `Authorization: Bearer`).
    access_token: String,
    project: String,
    location: String,
    model: String,
    tools: Vec<Tool>,
}

impl GoogleVertexLlm {
    /// Construct bound to an OAuth2 `access_token`, a GCP `project`, and a `location`
    /// (region, e.g. `us-central1`; empty → [`VERTEX_DEFAULT_LOCATION`]). Default model
    /// [`DEFAULT_MODEL`]; override with [`Self::model`].
    pub fn new(
        access_token: impl Into<String>,
        project: impl Into<String>,
        location: impl Into<String>,
    ) -> Self {
        let location = location.into();
        let location = if location.trim().is_empty() {
            VERTEX_DEFAULT_LOCATION.to_string()
        } else {
            location
        };
        Self {
            http: reqwest::Client::new(),
            access_token: access_token.into(),
            project: project.into(),
            location,
            model: DEFAULT_MODEL.to_string(),
            tools: Vec::new(),
        }
    }

    /// Override the model (default [`DEFAULT_MODEL`]).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// The Vertex host: regional `{location}-aiplatform.googleapis.com`, or the global
    /// `aiplatform.googleapis.com` when `location = "global"`.
    fn host(&self) -> String {
        if self.location == "global" {
            "aiplatform.googleapis.com".to_string()
        } else {
            format!("{}-aiplatform.googleapis.com", self.location)
        }
    }

    /// The full `streamGenerateContent` URL (`?alt=sse` for an SSE body, matching the
    /// AI-Studio client's decode).
    fn url(&self) -> String {
        format!(
            "https://{}/v1/projects/{}/locations/{}/publishers/google/models/{}:streamGenerateContent?alt=sse",
            self.host(),
            self.project,
            self.location,
            self.model,
        )
    }
}

#[async_trait]
impl LlmService for GoogleVertexLlm {
    fn name(&self) -> &str {
        "google_vertex"
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        Ok(())
    }

    async fn run_llm<'a>(&'a mut self, ctx: &'a LlmContext) -> Result<BoxStream<'a, Frame>> {
        let body = gemini_request_body(ctx, &self.tools);
        let url = self.url();
        let resp = self
            .http
            .post(&url)
            .bearer_auth(&self.access_token)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| FlowcatError::Network(format!("google_vertex send: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(FlowcatError::Network(format!(
                "google_vertex {status}: {text}"
            )));
        }
        Ok(sse_to_frames(resp.bytes_stream()))
    }

    fn set_tools(&mut self, tools: Vec<Tool>) {
        self.tools = tools;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regional_url_uses_location_host_and_project_path() {
        let llm = GoogleVertexLlm::new("tok", "my-proj", "us-central1").model("gemini-2.5-pro");
        assert_eq!(llm.name(), "google_vertex");
        assert_eq!(
            llm.url(),
            "https://us-central1-aiplatform.googleapis.com/v1/projects/my-proj/locations/us-central1/publishers/google/models/gemini-2.5-pro:streamGenerateContent?alt=sse"
        );
    }

    #[test]
    fn global_location_drops_the_region_prefix() {
        let llm = GoogleVertexLlm::new("tok", "my-proj", "global");
        assert_eq!(
            llm.url(),
            "https://aiplatform.googleapis.com/v1/projects/my-proj/locations/global/publishers/google/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
        );
    }

    #[test]
    fn empty_location_falls_back_to_default() {
        let llm = GoogleVertexLlm::new("tok", "my-proj", "   ");
        assert!(llm
            .url()
            .starts_with("https://us-central1-aiplatform.googleapis.com/"));
    }
}
