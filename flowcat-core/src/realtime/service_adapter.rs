// SPDX-License-Identifier: Apache-2.0
//
//! Bridge: [`RealtimeLlmService`] → [`RealtimeLlm`] + [`RealtimeKickoff`].
//!
//! flowcat has two realtime trait surfaces: the s2s pipeline
//! ([`build_s2s_task`](crate::pipeline::build_s2s_task)) is generic over
//! [`RealtimeLlm`] + [`RealtimeKickoff`] (which [`GeminiLive`](crate::GeminiLive)
//! implements directly), while the streaming realtime *connectors* (OpenAI, Grok,
//! Ultravox, … in `flowcat-services`) implement the frozen
//! [`RealtimeLlmService`]. [`ServiceRealtimeAdapter`] wraps any `RealtimeLlmService`
//! and presents it as a `RealtimeLlm + RealtimeKickoff`, so a connector drops into the
//! pipeline with **zero glue in the consuming app** — the realtime equivalent of how
//! a cascaded `LlmService`/`SttService`/`TtsService` is consumed directly.
//!
//! The only structural difference between the two traits is `send_audio`'s argument
//! (`AudioChunk` on the pipeline side vs `Arc<AudioFrame>` on the service side);
//! `RealtimeSetup`/`RealtimeEvent`/`ToolDecl` are shared/aliased, so everything else
//! is a straight delegate. The service trait has no lock-free `event_notify`/
//! `poll_event`, so the adapter inherits [`RealtimeLlm`]'s blocking defaults (which
//! the trait documents as the legacy-OK path).

use std::sync::Arc;

use async_trait::async_trait;

use crate::error::FlowcatError;
use crate::processor::frame::AudioFrame;
use crate::realtime::{RealtimeKickoff, RealtimeLlm};
use crate::service::RealtimeLlmService;
use crate::types::{AudioChunk, RealtimeEvent, RealtimeSetup, ToolDecl};

/// Adapts any [`RealtimeLlmService`] connector into the pipeline's
/// [`RealtimeLlm`] + [`RealtimeKickoff`] surface. See the module docs.
pub struct ServiceRealtimeAdapter<S: RealtimeLlmService> {
    inner: S,
}

impl<S: RealtimeLlmService> ServiceRealtimeAdapter<S> {
    /// Wrap a realtime connector.
    pub fn new(inner: S) -> Self {
        Self { inner }
    }

    /// Unwrap back to the underlying connector.
    pub fn into_inner(self) -> S {
        self.inner
    }
}

#[async_trait]
impl<S: RealtimeLlmService> RealtimeLlm for ServiceRealtimeAdapter<S> {
    async fn connect(&mut self, setup: RealtimeSetup) -> Result<(), FlowcatError> {
        self.inner.connect(setup).await
    }

    async fn send_audio(&mut self, chunk: AudioChunk) -> Result<(), FlowcatError> {
        // The only real shape difference: pipeline `AudioChunk` → service
        // `Arc<AudioFrame>` (`AudioFrame: From<AudioChunk>`).
        self.inner
            .send_audio(Arc::new(AudioFrame::from(chunk)))
            .await
    }

    async fn update_system(
        &mut self,
        prompt: String,
        tools: Vec<ToolDecl>,
    ) -> Result<(), FlowcatError> {
        // `Tool` is a type alias of `ToolDecl`, so the vec passes straight through.
        self.inner.update_system(prompt, tools).await
    }

    async fn rebase_session(
        &mut self,
        prompt: String,
        tools: Vec<ToolDecl>,
    ) -> Result<(), FlowcatError> {
        // Delegate to the connector's re-base (the audio-dropping reopen for the
        // OpenAI family; the default in-session swap otherwise).
        self.inner.rebase_session(prompt, tools).await
    }

    async fn send_tool_result(
        &mut self,
        id: String,
        result: serde_json::Value,
    ) -> Result<(), FlowcatError> {
        self.inner.send_tool_result(id, result).await
    }

    async fn next_event(&mut self) -> Option<RealtimeEvent> {
        self.inner.next_event().await
    }

    fn input_sample_rate(&self) -> u32 {
        self.inner.input_sample_rate()
    }

    // Delegate the lock-free event path to the connector. Without this the pipeline
    // would use the blocking `next_event` (holding the session lock across the idle
    // wait between turns), which starves `send_audio` and stalls caller audio after
    // the greeting. A connector that doesn't override these gets the service trait's
    // blocking defaults (same as before) — no regression.
    fn event_notify(&self) -> Option<std::sync::Arc<tokio::sync::Notify>> {
        self.inner.event_notify()
    }

    async fn poll_event(&mut self) -> crate::realtime::PollEvent {
        self.inner.poll_event().await
    }
}

#[async_trait]
impl<S: RealtimeLlmService> RealtimeKickoff for ServiceRealtimeAdapter<S> {
    async fn kickoff(&mut self) -> Result<(), FlowcatError> {
        self.inner.kickoff().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal `RealtimeLlmService` that records what it received, so we can assert
    /// the adapter delegates + converts correctly without a live socket.
    #[derive(Default)]
    struct SpyService {
        connected: bool,
        last_audio_rate: Option<u32>,
        kicked: bool,
    }

    #[async_trait]
    impl RealtimeLlmService for SpyService {
        async fn connect(&mut self, _setup: RealtimeSetup) -> Result<(), FlowcatError> {
            self.connected = true;
            Ok(())
        }
        async fn send_audio(&mut self, chunk: Arc<AudioFrame>) -> Result<(), FlowcatError> {
            self.last_audio_rate = Some(chunk.sample_rate);
            Ok(())
        }
        async fn update_system(
            &mut self,
            _p: String,
            _t: Vec<ToolDecl>,
        ) -> Result<(), FlowcatError> {
            Ok(())
        }
        async fn send_tool_result(
            &mut self,
            _id: String,
            _r: serde_json::Value,
        ) -> Result<(), FlowcatError> {
            Ok(())
        }
        async fn next_event(&mut self) -> Option<RealtimeEvent> {
            None
        }
        fn input_sample_rate(&self) -> u32 {
            24_000
        }
        async fn kickoff(&mut self) -> Result<(), FlowcatError> {
            self.kicked = true;
            Ok(())
        }
    }

    #[tokio::test]
    async fn adapter_delegates_and_converts_audio() {
        let mut a = ServiceRealtimeAdapter::new(SpyService::default());
        // input_sample_rate flows from the connector (24k here).
        assert_eq!(a.input_sample_rate(), 24_000);
        // send_audio: AudioChunk(24k) → Arc<AudioFrame>(24k) reaches the service.
        a.send_audio(AudioChunk::new(vec![0i16; 10], 24_000))
            .await
            .unwrap();
        assert_eq!(a.inner.last_audio_rate, Some(24_000));
        // kickoff routes through to the connector's kickoff.
        RealtimeKickoff::kickoff(&mut a).await.unwrap();
        assert!(a.inner.kicked);
    }
}
