// SPDX-License-Identifier: Apache-2.0
//
//! **Inworld Realtime** (speech-to-speech) client.
//!
//! Inworld's Realtime endpoint speaks an **OpenAI-Realtime-style** event
//! protocol (`session.update` / `input_audio_buffer.append` / `response.*`) over
//! `wss://api.inworld.ai/api/v1/realtime/session` with `Authorization: Bearer
//! <key>` auth (the reference `InworldRealtimeLLMService` mirrors the OpenAI
//! Realtime event surface). So this is a thin wrapper over
//! [`OpenAiRealtime`](super::openai::OpenAiRealtime) pointed at the Inworld base
//! URL.
//!
//! Behind the `realtime-inworld` feature (which enables `realtime-openai`).
//!
//! ## Keys / auth (security note)
//!
//! The key is sent **only** in the `Authorization: Bearer` header at connect
//! (never logged, never in the query).

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use flowcat_core::error::FlowcatError;
use flowcat_core::processor::frame::AudioFrame;
use flowcat_core::service::{RealtimeLlmService, RealtimeServiceSetup, Tool};
use flowcat_core::types::RealtimeEvent;

use super::openai::OpenAiRealtime;

/// Base WSS endpoint for the Inworld Realtime service.
pub const INWORLD_REALTIME_WSS_BASE: &str = "wss://api.inworld.ai/api/v1/realtime/session";

/// Inworld Realtime session — an [`OpenAiRealtime`] pointed at the Inworld URL.
pub struct InworldRealtime {
    inner: OpenAiRealtime,
}

impl InworldRealtime {
    /// Construct an Inworld Realtime client bound to the given Inworld API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        // Inworld selects the model via the session URL/params, not a `?model=`;
        // the base already lacks a `model=` so the wrapper appends one harmlessly
        // — Inworld ignores an unknown query. Callers can override the base.
        let inner = OpenAiRealtime::new(api_key).with_base_url(INWORLD_REALTIME_WSS_BASE);
        Self { inner }
    }

    /// Override the base URL (e.g. a session URL pre-issued by the Inworld REST
    /// API that already carries query params).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.inner = self.inner.with_base_url(base_url);
        self
    }
}

#[async_trait]
impl RealtimeLlmService for InworldRealtime {
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
    fn inworld_targets_its_endpoint_and_holds_the_key() {
        let c = InworldRealtime::new("inworld-secret");
        assert_eq!(c.inner.api_key(), "inworld-secret");
        assert!(INWORLD_REALTIME_WSS_BASE.starts_with("wss://api.inworld.ai/"));
    }

    /// `INWORLD_API_KEY=… cargo test -p flowcat-services --features realtime-inworld -- \
    ///   realtime::inworld::tests::live_inworld_realtime_smoke --ignored --nocapture`
    #[tokio::test]
    #[ignore = "live: needs INWORLD_API_KEY + network (Inworld Realtime)"]
    async fn live_inworld_realtime_smoke() {
        let key = std::env::var("INWORLD_API_KEY").expect("INWORLD_API_KEY");
        let mut c = InworldRealtime::new(key);
        c.connect(RealtimeServiceSetup {
            model: String::new(),
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
