// SPDX-License-Identifier: Apache-2.0
//
//! **Groq** TTS — a **(W)rapper** over the OpenAI-TTS-HTTP family.
//!
//! Groq's TTS is the OpenAI `/audio/speech` wire shape (`Authorization: Bearer`,
//! `{input,model,voice,response_format}`) at `https://api.groq.com/openai/v1`,
//! differing only in `base_url`, default model/voice, sample rate, and that it
//! returns a **WAV** body (vs OpenAI's raw PCM). So it is the [`OpenAiTts`] client
//! pointed at Groq with the WAV [`AudioContainer`] — a ~20-line config delegation
//! (the `(W)` triage, PROVIDERS.md §3). Behind the `tts-groq` feature.

use async_trait::async_trait;

use flowcat_core::error::Result;
use flowcat_core::processor::frame::{Frame, StartParams};
use flowcat_core::service::TtsService;

use super::openai::{AudioContainer, OpenAiTts, OpenAiTtsBuilder};

/// Groq's OpenAI-compatible API base. The key rides the `Authorization` header,
/// never the URL.
pub const GROQ_API_BASE: &str = "https://api.groq.com/openai/v1";

/// Groq TTS — the OpenAI-TTS-HTTP client configured for Groq (WAV output, 48 kHz).
pub struct GroqTts {
    inner: OpenAiTts,
}

impl GroqTts {
    /// Construct bound to `api_key` + `voice_id` (default model
    /// `canopylabs/orpheus-v1-english`, 48 kHz WAV — the pipecat Groq defaults).
    pub fn new(api_key: impl Into<String>, voice_id: impl Into<String>) -> Self {
        Self {
            inner: OpenAiTtsBuilder::new(api_key, voice_id)
                .name("groq")
                .base_url(GROQ_API_BASE)
                .model("canopylabs/orpheus-v1-english")
                .sample_rate(48_000)
                .container(AudioContainer::Wav)
                .build(),
        }
    }
}

#[async_trait]
impl TtsService for GroqTts {
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
    fn wraps_openai_client_pointed_at_groq() {
        let tts = GroqTts::new("k", "autumn");
        assert_eq!(tts.name(), "groq");
        assert_eq!(tts.sample_rate(), 48_000);
        assert_eq!(
            tts.inner.url(),
            "https://api.groq.com/openai/v1/audio/speech"
        );
    }

    /// Live smoke (requires `GROQ_API_KEY`). Run:
    /// `GROQ_API_KEY=… cargo test -p flowcat-services --features tts-groq -- --ignored groq_tts_live`
    #[tokio::test]
    #[ignore = "requires GROQ_API_KEY"]
    async fn groq_tts_live_synthesizes_audio() {
        let key = std::env::var("GROQ_API_KEY").expect("GROQ_API_KEY");
        let mut tts = GroqTts::new(key, "autumn");
        tts.start(&StartParams::default()).await.expect("start");
        let frames = tts.run_tts("Hello from flowcat.").await.expect("run_tts");
        assert!(frames.iter().any(|f| matches!(f, Frame::TtsAudio { .. })));
    }
}
