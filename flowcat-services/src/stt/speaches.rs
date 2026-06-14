// SPDX-License-Identifier: Apache-2.0
//
//! **Speaches** STT — a **(W)rapper** over the Whisper-HTTP family.
//!
//! Speaches is a self-hosted, OpenAI-compatible server exposing
//! `/v1/audio/transcriptions`, so it is the [`OpenAiStt`](super::OpenAiStt) family
//! client ([`WhisperHttpStt`]) pointed at the operator's Speaches base (default
//! `http://localhost:8000/v1`, matching pipecat `services/speaches/stt.py`) with
//! `Authorization: Bearer <key>`. Because the server is self-hosted, the API key is
//! often a placeholder — [`SpeachesStt::with_base_url`] lets the operator set their
//! own endpoint. Behind the `stt-speaches` feature.

use std::sync::Arc;

use async_trait::async_trait;

use flowcat_core::error::Result;
use flowcat_core::processor::frame::{AudioFrame, Frame, StartParams};
use flowcat_core::service::SttService;

use super::openai::{WhisperAuth, WhisperHttpStt, WhisperHttpSttBuilder};

/// Default Speaches base URL (the self-hosted server's OpenAI-compatible API).
pub const SPEACHES_API_BASE: &str = "http://localhost:8000/v1";

/// A common Speaches default model (faster-whisper). Override per deployment.
pub const SPEACHES_DEFAULT_MODEL: &str = "Systran/faster-whisper-small";

/// Speaches Whisper STT — a thin newtype over the Whisper-HTTP family client aimed
/// at a self-hosted Speaches base. Delegates the [`SttService`] contract.
pub struct SpeachesStt(WhisperHttpStt);

impl SpeachesStt {
    /// Construct a Speaches STT client bound to `api_key` at the default
    /// (`localhost`) base. Use [`SpeachesStt::with_base_url`] for a remote server.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_base_url(api_key, SPEACHES_API_BASE)
    }

    /// Construct against an operator-supplied Speaches `base_url`.
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self(
            WhisperHttpSttBuilder::new(api_key)
                .base_url(base_url)
                .auth(WhisperAuth::Bearer)
                .model(SPEACHES_DEFAULT_MODEL)
                .build()
                .with_name("speaches"),
        )
    }
}

#[async_trait]
impl SttService for SpeachesStt {
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
    fn speaches_targets_the_self_hosted_base() {
        let stt = SpeachesStt::new("none");
        assert_eq!(stt.name(), "speaches");
        assert_eq!(
            WhisperHttpSttBuilder::new("k")
                .base_url(SPEACHES_API_BASE)
                .url(),
            "http://localhost:8000/v1/audio/transcriptions"
        );
    }

    #[test]
    fn speaches_honours_a_custom_base_url() {
        // A remote Speaches server is reached by overriding the base.
        let url = WhisperHttpSttBuilder::new("k")
            .base_url("https://speaches.internal.example/v1/")
            .url();
        assert_eq!(
            url,
            "https://speaches.internal.example/v1/audio/transcriptions"
        );
        let _ = SpeachesStt::with_base_url("none", "https://speaches.internal.example/v1");
    }

    #[test]
    fn speaches_uses_bearer_auth() {
        assert_eq!(WhisperAuth::Bearer.header_value("none"), "Bearer none");
    }

    /// Live smoke (requires a running Speaches at `SPEACHES_BASE_URL`). Run with:
    /// `SPEACHES_BASE_URL=… cargo test -p flowcat-services --features stt-speaches -- --ignored speaches_stt_live`
    #[tokio::test]
    #[ignore = "requires a running Speaches server (SPEACHES_BASE_URL)"]
    async fn speaches_stt_live_transcribes() {
        let base = std::env::var("SPEACHES_BASE_URL").expect("SPEACHES_BASE_URL");
        let mut stt = SpeachesStt::with_base_url("none", base);
        stt.start(&StartParams::default()).await.expect("start");
        let audio = Arc::new(AudioFrame::mono(vec![0i16; 16_000], 16_000));
        let _ = stt.run_stt(audio).await.expect("run_stt");
    }
}
