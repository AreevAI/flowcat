// SPDX-License-Identifier: Apache-2.0
//
//! **Speaches** TTS — a **(W)rapper** over the OpenAI-TTS-HTTP family, self-hosted.
//!
//! Speaches ([speaches.ai](https://speaches.ai/), `speaches-ai/speaches`) is an
//! OpenAI-API-compatible speech server you run yourself — its `/v1/audio/speech`
//! takes the same `{input, model, voice, response_format}` body as OpenAI. So this
//! is the [`OpenAiTts`] client pointed at the operator's instance via a configurable
//! `base_url` (config, never request-derived → no SSRF surface). Default model
//! `speaches-ai/Kokoro-82M` (the voice id is the Kokoro voice, e.g. `af_heart`),
//! raw PCM @ 24 kHz. Behind the `tts-speaches` feature.

use async_trait::async_trait;

use flowcat_core::error::Result;
use flowcat_core::processor::frame::{Frame, StartParams};
use flowcat_core::service::TtsService;

use super::openai::{OpenAiTts, OpenAiTtsBuilder};

/// Default Speaches base URL when none is configured (the common local dev port).
pub const SPEACHES_DEFAULT_BASE: &str = "http://localhost:8000/v1";

/// Speaches TTS — the OpenAI-TTS-HTTP client pointed at a self-hosted instance.
pub struct SpeachesTts {
    inner: OpenAiTts,
}

impl SpeachesTts {
    /// Construct bound to `api_key` + `base_url` (the self-hosted instance, e.g.
    /// `http://host:8000/v1`; empty → the localhost default) + `voice`. Default model
    /// `speaches-ai/Kokoro-82M-v1.0-ONNX` (the Kokoro id in Speaches' registry; install
    /// it once via `POST /v1/models/{id}`), 24 kHz raw PCM (the OpenAI `/audio/speech`
    /// shape). Voice ids are Kokoro voices (e.g. `af_heart`).
    pub fn new(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        voice: impl Into<String>,
    ) -> Self {
        let base = base_url.into();
        let base = if base.trim().is_empty() {
            SPEACHES_DEFAULT_BASE.to_string()
        } else {
            base
        };
        Self {
            inner: OpenAiTtsBuilder::new(api_key, voice)
                .name("speaches")
                .base_url(base)
                .model("speaches-ai/Kokoro-82M-v1.0-ONNX")
                .build(),
        }
    }
}

#[async_trait]
impl TtsService for SpeachesTts {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn sample_rate(&self) -> u32 {
        self.inner.sample_rate()
    }

    async fn start(&mut self, params: &StartParams) -> Result<()> {
        self.inner.start(params).await
    }

    async fn run_tts(&mut self, text: &str) -> Result<Vec<Frame>> {
        self.inner.run_tts(text).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn points_client_at_the_configured_base() {
        let tts = SpeachesTts::new("k", "http://my-host:8000/v1", "af_heart");
        assert_eq!(tts.name(), "speaches");
        assert_eq!(tts.inner.url(), "http://my-host:8000/v1/audio/speech");
        assert_eq!(tts.sample_rate(), 24_000);
    }

    #[test]
    fn empty_base_falls_back_to_localhost() {
        let tts = SpeachesTts::new("k", "   ", "af_heart");
        assert_eq!(tts.inner.url(), "http://localhost:8000/v1/audio/speech");
    }
}
