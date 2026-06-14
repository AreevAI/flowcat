// SPDX-License-Identifier: Apache-2.0
//
//! **xAI Grok Realtime** (speech-to-speech) client.
//!
//! xAI's Realtime endpoint speaks the **OpenAI Realtime wire protocol** over
//! `wss://api.x.ai/v1/realtime?model=<model>` with `Authorization: Bearer <key>`
//! auth (the reference `GrokRealtimeLLMService` aliases the xAI service, which
//! subclasses the OpenAI Realtime service). So this is a thin wrapper over
//! [`OpenAiRealtime`](super::openai::OpenAiRealtime) pointed at the xAI base URL.
//!
//! Behind the `realtime-grok` feature (which enables `realtime-openai`).
//!
//! ## Keys / auth (security note)
//!
//! The key is sent **only** in the `Authorization: Bearer` header at connect
//! (never logged, never in the query). Default model `grok-realtime`.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use flowcat_core::error::FlowcatError;
use flowcat_core::processor::frame::AudioFrame;
use flowcat_core::service::{RealtimeLlmService, RealtimeServiceSetup, Tool};
use flowcat_core::types::RealtimeEvent;

use super::openai::OpenAiRealtime;

/// Base WSS endpoint for the xAI Grok Realtime service.
pub const GROK_REALTIME_WSS_BASE: &str = "wss://api.x.ai/v1/realtime";

/// xAI Grok Realtime session — an [`OpenAiRealtime`] pointed at the xAI base URL.
pub struct GrokRealtime {
    inner: OpenAiRealtime,
}

impl GrokRealtime {
    /// Construct a Grok Realtime client bound to the given xAI API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        let inner = OpenAiRealtime::new(api_key).with_base_url(GROK_REALTIME_WSS_BASE);
        Self { inner }
    }

    /// Override the base URL (e.g. an enterprise/staging xAI endpoint).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.inner = self.inner.with_base_url(base_url);
        self
    }
}

#[async_trait]
impl RealtimeLlmService for GrokRealtime {
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
    fn grok_targets_the_xai_endpoint_and_holds_the_key() {
        let c = GrokRealtime::new("xai-secret");
        assert_eq!(c.inner.api_key(), "xai-secret");
        assert_eq!(GROK_REALTIME_WSS_BASE, "wss://api.x.ai/v1/realtime");
    }

    /// `XAI_API_KEY=… cargo test -p flowcat-services --features realtime-grok -- \
    ///   realtime::grok::tests::live_grok_realtime_smoke --ignored --nocapture`
    #[tokio::test]
    #[ignore = "live: needs XAI_API_KEY + network (xAI Grok Realtime)"]
    async fn live_grok_realtime_smoke() {
        let key = std::env::var("XAI_API_KEY").expect("XAI_API_KEY");
        let mut c = GrokRealtime::new(key);
        c.connect(RealtimeServiceSetup {
            model: "grok-realtime".into(),
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
