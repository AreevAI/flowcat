// SPDX-License-Identifier: Apache-2.0
//
//! **Groq Whisper** STT — a **(W)rapper** over the Whisper-HTTP family.
//!
//! Groq exposes the OpenAI-compatible `/audio/transcriptions` endpoint at
//! `https://api.groq.com/openai/v1` with `Authorization: Bearer <key>`, so this is
//! the [`OpenAiStt`](super::OpenAiStt) family client ([`WhisperHttpStt`]) pointed at
//! the Groq base + the Groq default model (`whisper-large-v3-turbo`, matching
//! pipecat `services/groq/stt.py`). Behind the `stt-groq` feature.

use std::sync::Arc;

use async_trait::async_trait;

use flowcat_core::error::Result;
use flowcat_core::processor::frame::{AudioFrame, Frame, StartParams};
use flowcat_core::service::SttService;

use super::openai::{WhisperAuth, WhisperHttpStt, WhisperHttpSttBuilder};

/// Groq base URL for the OpenAI-compatible transcription endpoint.
pub const GROQ_API_BASE: &str = "https://api.groq.com/openai/v1";

/// Groq's default Whisper model (pipecat `GroqSTTService`).
pub const GROQ_DEFAULT_MODEL: &str = "whisper-large-v3-turbo";

/// Groq Whisper STT — a thin newtype over the Whisper-HTTP family client aimed at
/// the Groq base. Delegates the whole [`SttService`] contract to the inner client.
pub struct GroqStt(WhisperHttpStt);

impl GroqStt {
    /// Construct a Groq Whisper STT client bound to `api_key`.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self(
            WhisperHttpSttBuilder::new(api_key)
                .base_url(GROQ_API_BASE)
                .auth(WhisperAuth::Bearer)
                .model(GROQ_DEFAULT_MODEL)
                .build()
                .with_name("groq"),
        )
    }
}

#[async_trait]
impl SttService for GroqStt {
    fn name(&self) -> &str {
        self.0.name()
    }
    async fn start(&mut self, params: &StartParams) -> Result<()> {
        self.0.start(params).await
    }
    async fn run_stt(&mut self, audio: Arc<AudioFrame>) -> Result<Vec<Frame>> {
        self.0.run_stt(audio).await
    }
    async fn set_muted(&mut self, muted: bool) {
        self.0.set_muted(muted).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groq_targets_the_groq_base_and_model() {
        let stt = GroqStt::new("gsk_test");
        assert_eq!(stt.name(), "groq");
        // The wrapper points the family client at the Groq base + path.
        assert_eq!(
            WhisperHttpSttBuilder::new("k")
                .base_url(GROQ_API_BASE)
                .url(),
            "https://api.groq.com/openai/v1/audio/transcriptions"
        );
        assert_eq!(GROQ_DEFAULT_MODEL, "whisper-large-v3-turbo");
    }

    #[test]
    fn groq_uses_bearer_auth() {
        assert_eq!(WhisperAuth::Bearer.header_value("gsk_x"), "Bearer gsk_x");
    }

    /// Live smoke (requires `GROQ_API_KEY`). Run with:
    /// `GROQ_API_KEY=… cargo test -p flowcat-services --features stt-groq -- --ignored groq_stt_live`
    #[tokio::test]
    #[ignore = "requires GROQ_API_KEY"]
    async fn groq_stt_live_transcribes() {
        let key = std::env::var("GROQ_API_KEY").expect("GROQ_API_KEY");
        let mut stt = GroqStt::new(key);
        stt.start(&StartParams::default()).await.expect("start");
        let audio = Arc::new(AudioFrame::mono(vec![0i16; 16_000], 16_000));
        let _ = stt.run_stt(audio).await.expect("run_stt");
    }
}
