// SPDX-License-Identifier: Apache-2.0
//
//! **Azure OpenAI Realtime** (speech-to-speech) client.
//!
//! Azure's Realtime endpoint speaks the **same wire protocol** as OpenAI
//! Realtime, differing only in (a) the endpoint URL — a full Azure WSS URL that
//! already carries `?api-version=…&deployment=…` — and (b) authentication, which
//! uses an `api-key: <key>` header instead of `Authorization: Bearer …`. So this
//! is a thin wrapper over [`OpenAiRealtime`](super::openai::OpenAiRealtime)
//! configured with the Azure base URL + header (mirrors the reference
//! `AzureRealtimeLLMService(OpenAIRealtimeLLMService)` subclass).
//!
//! Behind the `realtime-azure` feature (which enables `realtime-openai`).
//!
//! ## Keys / auth (security note)
//!
//! The key is sent **only** in the `api-key` header at connect (never logged,
//! never in the URL/query). The full deployment URL is provided by the caller.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use flowcat_core::error::FlowcatError;
use flowcat_core::processor::frame::AudioFrame;
use flowcat_core::service::{RealtimeLlmService, RealtimeServiceSetup, Tool};
use flowcat_core::types::RealtimeEvent;

use super::openai::OpenAiRealtime;

/// Azure OpenAI Realtime session — an [`OpenAiRealtime`] pointed at an Azure
/// deployment URL and authenticated with the `api-key` header.
pub struct AzureRealtime {
    inner: OpenAiRealtime,
}

impl AzureRealtime {
    /// Construct an Azure Realtime client.
    ///
    /// - `api_key`: the Azure OpenAI key (sent as the `api-key` header).
    /// - `endpoint_url`: the full Azure WSS realtime URL, e.g.
    ///   `wss://<resource>.openai.azure.com/openai/realtime?api-version=2025-04-01-preview&deployment=<deployment>`
    ///   (the `?api-version` + `?deployment` query is preserved as-is; the model
    ///   is selected by the deployment, so no `?model=` is appended).
    pub fn new(api_key: impl Into<String>, endpoint_url: impl Into<String>) -> Self {
        let key = api_key.into();
        let inner = OpenAiRealtime::new(key.clone())
            .with_base_url(endpoint_url)
            // Azure auth header (no Bearer).
            .with_headers(vec![("api-key".to_string(), key)]);
        Self { inner }
    }
}

#[async_trait]
impl RealtimeLlmService for AzureRealtime {
    async fn connect(&mut self, setup: RealtimeServiceSetup) -> Result<(), FlowcatError> {
        self.inner.connect(setup).await
    }

    async fn send_audio(&mut self, chunk: Arc<AudioFrame>) -> Result<(), FlowcatError> {
        self.inner.send_audio(chunk).await
    }

    async fn update_system(
        &mut self,
        prompt: String,
        tools: Vec<Tool>,
    ) -> Result<(), FlowcatError> {
        self.inner.update_system(prompt, tools).await
    }

    async fn send_tool_result(&mut self, id: String, result: Value) -> Result<(), FlowcatError> {
        self.inner.send_tool_result(id, result).await
    }

    async fn next_event(&mut self) -> Option<RealtimeEvent> {
        self.inner.next_event().await
    }

    fn input_sample_rate(&self) -> u32 {
        self.inner.input_sample_rate()
    }

    async fn kickoff(&mut self) -> Result<(), FlowcatError> {
        self.inner.kickoff().await
    }

    fn event_notify(&self) -> Option<std::sync::Arc<tokio::sync::Notify>> {
        self.inner.event_notify()
    }

    async fn poll_event(&mut self) -> flowcat_core::realtime::PollEvent {
        self.inner.poll_event().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn azure_uses_api_key_header_and_preserves_deployment_url() {
        // The wrapper must point the inner client at the Azure URL (deployment
        // query preserved) and carry the `api-key` header — never Bearer.
        let url = "wss://r.openai.azure.com/openai/realtime?api-version=2025-04-01&deployment=rt";
        let az = AzureRealtime::new("azure-secret", url);
        // The inner OpenAI client holds the same key (used only in the header).
        assert_eq!(az.inner.api_key(), "azure-secret");
    }

    /// `AZURE_OPENAI_API_KEY=… AZURE_OPENAI_REALTIME_URL=wss://… cargo test \
    ///   -p flowcat-services --features realtime-azure -- \
    ///   realtime::azure::tests::live_azure_realtime_smoke --ignored --nocapture`
    #[tokio::test]
    #[ignore = "live: needs AZURE_OPENAI_API_KEY + AZURE_OPENAI_REALTIME_URL"]
    async fn live_azure_realtime_smoke() {
        let key = std::env::var("AZURE_OPENAI_API_KEY").expect("AZURE_OPENAI_API_KEY");
        let url = std::env::var("AZURE_OPENAI_REALTIME_URL").expect("AZURE_OPENAI_REALTIME_URL");
        let mut c = AzureRealtime::new(key, url);
        c.connect(RealtimeServiceSetup {
            model: String::new(), // selected by the deployment
            system_prompt: "You are a helpful agent.".into(),
            tools: vec![],
            input_sample_rate: 24_000,
            output_sample_rate: 24_000,
        })
        .await
        .expect("connect");
        c.kickoff().await.expect("kickoff");
        while let Some(ev) = c.next_event().await {
            if matches!(ev, RealtimeEvent::Closed) {
                break;
            }
        }
    }
}
