// SPDX-License-Identifier: Apache-2.0
//
//! **xAI** STT — a **(W)rapper** over the Whisper-HTTP family.
//!
//! xAI exposes an OpenAI-compatible API at `https://api.x.ai/v1` with
//! `Authorization: Bearer <key>`, so per the PROVIDERS.md §2 triage (note ³)
//! this rides the Whisper-HTTP family client ([`OpenAiStt`](super::OpenAiStt) =
//! [`WhisperHttpStt`]) — the same `/audio/transcriptions` request/response shape,
//! just the xAI base + model. (pipecat additionally ships a streaming-WebSocket xAI
//! STT; the deliberate consolidation here groups xAI with the Whisper-HTTP
//! `(W)` cohort — one client, config-only wrappers.) Behind the `stt-xai` feature.

use std::sync::Arc;

use async_trait::async_trait;

use flowcat_core::error::Result;
use flowcat_core::processor::frame::{AudioFrame, Frame, StartParams};
use flowcat_core::service::SttService;

use super::openai::{WhisperAuth, WhisperHttpStt, WhisperHttpSttBuilder};

/// xAI base URL for the OpenAI-compatible API.
pub const XAI_API_BASE: &str = "https://api.x.ai/v1";

/// xAI's default transcription model.
pub const XAI_DEFAULT_MODEL: &str = "whisper-1";

/// xAI Whisper-HTTP STT — a thin newtype over the Whisper-HTTP family client aimed
/// at the xAI base. Delegates the whole [`SttService`] contract.
pub struct XaiStt(WhisperHttpStt);

impl XaiStt {
    /// Construct an xAI STT client bound to `api_key`.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self(
            WhisperHttpSttBuilder::new(api_key)
                .base_url(XAI_API_BASE)
                .auth(WhisperAuth::Bearer)
                .model(XAI_DEFAULT_MODEL)
                .build()
                .with_name("xai"),
        )
    }
}

#[async_trait]
impl SttService for XaiStt {
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
    fn xai_targets_the_xai_base() {
        let stt = XaiStt::new("xai-test");
        assert_eq!(stt.name(), "xai");
        assert_eq!(
            WhisperHttpSttBuilder::new("k").base_url(XAI_API_BASE).url(),
            "https://api.x.ai/v1/audio/transcriptions"
        );
        assert_eq!(XAI_DEFAULT_MODEL, "whisper-1");
    }

    #[test]
    fn xai_uses_bearer_auth() {
        assert_eq!(WhisperAuth::Bearer.header_value("xai_x"), "Bearer xai_x");
    }

    /// Live smoke (requires `XAI_API_KEY`). Run with:
    /// `XAI_API_KEY=… cargo test -p flowcat-services --features stt-xai -- --ignored xai_stt_live`
    #[tokio::test]
    #[ignore = "requires XAI_API_KEY"]
    async fn xai_stt_live_transcribes() {
        let key = std::env::var("XAI_API_KEY").expect("XAI_API_KEY");
        let mut stt = XaiStt::new(key);
        stt.start(&StartParams::default()).await.expect("start");
        let audio = Arc::new(AudioFrame::mono(vec![0i16; 16_000], 16_000));
        let _ = stt.run_stt(audio).await.expect("run_stt");
    }
}
