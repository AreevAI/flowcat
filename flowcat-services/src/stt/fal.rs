// SPDX-License-Identifier: Apache-2.0
//
//! **fal** STT — a **(W)rapper** over the Whisper-HTTP family.
//!
//! Per the PROVIDERS.md §2 triage, fal rides the Whisper-HTTP family client
//! ([`OpenAiStt`](super::OpenAiStt) = [`WhisperHttpStt`]) — the same
//! `/audio/transcriptions` request/response wire shape, pointed at fal's
//! OpenAI-compatible base with **`Authorization: Key <key>`** (fal's scheme;
//! pipecat's `services/fal/stt.py` uses the same `Key ` prefix). It differs from
//! pipecat's bespoke Wizper-JSON path, which is the deliberate family
//! consolidation: one Whisper-HTTP client, four config-only wrappers. Behind the
//! `stt-fal` feature.

use std::sync::Arc;

use async_trait::async_trait;

use flowcat_core::error::Result;
use flowcat_core::processor::frame::{AudioFrame, Frame, StartParams};
use flowcat_core::service::SttService;

use super::openai::{WhisperAuth, WhisperHttpStt, WhisperHttpSttBuilder};

/// fal's OpenAI-compatible transcription base.
pub const FAL_API_BASE: &str = "https://api.fal.ai/v1";

/// fal's default Whisper model (Wizper v3).
pub const FAL_DEFAULT_MODEL: &str = "whisper-1";

/// fal Whisper STT — a thin newtype over the Whisper-HTTP family client aimed at
/// the fal base, authenticating with `Authorization: Key <key>`.
pub struct FalStt(WhisperHttpStt);

impl FalStt {
    /// Construct a fal Whisper STT client bound to `api_key` (fal `Key` auth).
    pub fn new(api_key: impl Into<String>) -> Self {
        Self(
            WhisperHttpSttBuilder::new(api_key)
                .base_url(FAL_API_BASE)
                .auth(WhisperAuth::Key)
                .model(FAL_DEFAULT_MODEL)
                .build()
                .with_name("fal"),
        )
    }
}

#[async_trait]
impl SttService for FalStt {
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
    fn fal_targets_the_fal_base_and_model() {
        let stt = FalStt::new("fal_test");
        assert_eq!(stt.name(), "fal");
        assert_eq!(
            WhisperHttpSttBuilder::new("k").base_url(FAL_API_BASE).url(),
            "https://api.fal.ai/v1/audio/transcriptions"
        );
        assert_eq!(FAL_DEFAULT_MODEL, "whisper-1");
    }

    #[test]
    fn fal_uses_key_auth_not_bearer() {
        // fal is the one Whisper-HTTP provider with `Key ` auth (not `Bearer `).
        assert_eq!(WhisperAuth::Key.header_value("fal_x"), "Key fal_x");
    }

    /// Live smoke (requires `FAL_KEY`). Run with:
    /// `FAL_KEY=… cargo test -p flowcat-services --features stt-fal -- --ignored fal_stt_live`
    #[tokio::test]
    #[ignore = "requires FAL_KEY"]
    async fn fal_stt_live_transcribes() {
        let key = std::env::var("FAL_KEY").expect("FAL_KEY");
        let mut stt = FalStt::new(key);
        stt.start(&StartParams::default()).await.expect("start");
        let audio = Arc::new(AudioFrame::mono(vec![0i16; 16_000], 16_000));
        let _ = stt.run_stt(audio).await.expect("run_stt");
    }
}
