// SPDX-License-Identifier: Apache-2.0
//
//! **xAI** TTS — a **(W)rapper** over the OpenAI-TTS-HTTP family.
//!
//! xAI's batch HTTP TTS (`XAIHttpTTSService`) shares the OpenAI family transport —
//! a `Authorization: Bearer <key>` POST that returns raw `pcm` bytes — but at its
//! own endpoint `https://api.x.ai/v1/tts` with a different request body
//! (`{text, voice_id, output_format: {codec, sample_rate}}` rather than
//! `{input, model, voice, response_format}`). So it is the [`OpenAiTts`] client
//! pointed at xAI with the `/tts` endpoint + the [`BodyShape::XaiTts`] body — a
//! ~20-line config delegation (the `(W)` triage, PROVIDERS.md §3). Behind
//! the `tts-xai` feature.

use async_trait::async_trait;

use flowcat_core::error::Result;
use flowcat_core::processor::frame::{Frame, StartParams};
use flowcat_core::service::TtsService;

use super::openai::{BodyShape, OpenAiTts, OpenAiTtsBuilder};

/// xAI's API base. The key rides the `Authorization` header, never the URL.
pub const XAI_API_BASE: &str = "https://api.x.ai/v1";

/// xAI TTS — the OpenAI-TTS-HTTP client configured for xAI's `/tts` endpoint.
pub struct XaiTts {
    inner: OpenAiTts,
}

impl XaiTts {
    /// Construct bound to `api_key` + `voice_id` (default voice `eve`, 24 kHz raw
    /// PCM — the pipecat xAI defaults).
    pub fn new(api_key: impl Into<String>, voice_id: impl Into<String>) -> Self {
        Self {
            inner: OpenAiTtsBuilder::new(api_key, voice_id)
                .name("xai")
                .base_url(XAI_API_BASE)
                .endpoint("/tts")
                .body_shape(BodyShape::XaiTts)
                .sample_rate(24_000)
                .build(),
        }
    }
}

#[async_trait]
impl TtsService for XaiTts {
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
    fn wraps_openai_client_pointed_at_xai_tts() {
        let tts = XaiTts::new("k", "eve");
        assert_eq!(tts.name(), "xai");
        assert_eq!(tts.sample_rate(), 24_000);
        assert_eq!(tts.inner.url(), "https://api.x.ai/v1/tts");
    }

    /// Live smoke (requires `XAI_API_KEY`). Run:
    /// `XAI_API_KEY=… cargo test -p flowcat-services --features tts-xai -- --ignored xai_tts_live`
    #[tokio::test]
    #[ignore = "requires XAI_API_KEY"]
    async fn xai_tts_live_synthesizes_audio() {
        let key = std::env::var("XAI_API_KEY").expect("XAI_API_KEY");
        let mut tts = XaiTts::new(key, "eve");
        tts.start(&StartParams::default()).await.expect("start");
        let frames = tts.run_tts("Hello from flowcat.").await.expect("run_tts");
        assert!(frames.iter().any(|f| matches!(f, Frame::TtsAudio { .. })));
    }
}
