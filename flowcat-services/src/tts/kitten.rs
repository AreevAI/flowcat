// SPDX-License-Identifier: Apache-2.0
//
//! **KittenTTS** — a **(W)rapper** over the OpenAI-TTS-HTTP family, self-hosted.
//!
//! KittenTTS (`KittenML/KittenTTS`) is a lightweight open-weights CPU TTS model
//! family (15M–80M params, ONNX). It ships as a Python library, not a server, so
//! the expected deployment is an OpenAI-compatible serving wrapper in front of it
//! (community servers exist, or a ~100-line FastAPI shim) exposing the standard
//! `/v1/audio/speech` `{input, model, voice, response_format}` body. This client
//! is [`OpenAiTts`] pointed at that instance via a configurable `base_url`
//! (config, never request-derived → no SSRF surface). Default model
//! `kitten-tts-mini` (the serving wrapper decides which checkpoint that maps to);
//! the voice id is a Kitten voice name (e.g. `Bella`). Raw PCM @ 24 kHz. Behind
//! the `tts-kitten` feature.

use async_trait::async_trait;

use flowcat_core::error::Result;
use flowcat_core::processor::frame::{Frame, StartParams};
use flowcat_core::service::TtsService;

use super::openai::{OpenAiTts, OpenAiTtsBuilder};

/// Default KittenTTS-server base URL when none is configured (the common local
/// dev port for OpenAI-compatible speech shims).
pub const KITTEN_DEFAULT_BASE: &str = "http://localhost:8880/v1";

/// KittenTTS — the OpenAI-TTS-HTTP client pointed at a self-hosted instance.
pub struct KittenTts {
    inner: OpenAiTts,
}

impl KittenTts {
    /// Construct bound to `api_key` (self-hosted instances typically ignore it,
    /// but the header is still sent) + `base_url` (e.g. `http://host:8880/v1`;
    /// empty → the localhost default) + `voice` (a Kitten voice name, e.g.
    /// `Bella`; empty → the server's default voice). Default model
    /// `kitten-tts-mini`, 24 kHz raw PCM (the OpenAI `/audio/speech` shape).
    pub fn new(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        voice: impl Into<String>,
    ) -> Self {
        let base = base_url.into();
        let base = if base.trim().is_empty() {
            KITTEN_DEFAULT_BASE.to_string()
        } else {
            base
        };
        Self {
            inner: OpenAiTtsBuilder::new(api_key, voice)
                .name("kitten")
                .base_url(base)
                .model("kitten-tts-mini")
                .build(),
        }
    }
}

#[async_trait]
impl TtsService for KittenTts {
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
        let tts = KittenTts::new("k", "http://my-host:8880/v1", "Bella");
        assert_eq!(tts.name(), "kitten");
        assert_eq!(tts.inner.url(), "http://my-host:8880/v1/audio/speech");
        assert_eq!(tts.sample_rate(), 24_000);
    }

    #[test]
    fn empty_base_falls_back_to_localhost() {
        let tts = KittenTts::new("k", "   ", "Bella");
        assert_eq!(tts.inner.url(), "http://localhost:8880/v1/audio/speech");
    }
}
