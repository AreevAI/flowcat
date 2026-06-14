// SPDX-License-Identifier: Apache-2.0
//
//! **Speechmatics** TTS — a **(D)istinct** HTTP-POST-audio client.
//!
//! Speechmatics synthesizes a whole utterance with a single POST to
//! `{base}/generate/{voice}?output_format=pcm_{sample_rate}` (cross-checked
//! against pipecat `services/speechmatics/tts.py`): `Authorization: Bearer <key>`,
//! a tiny JSON body `{ "text": "…" }`, and a response of raw little-endian
//! `pcm_s16le` bytes at the requested rate. The voice is part of the **URL path**,
//! so the host is fixed and only a validated voice id + numeric rate are
//! interpolated. The default endpoint is the public preview host
//! (`https://preview.tts.speechmatics.com`, 16 kHz). The request encode
//! ([`build_url`] / [`build_body`]) is a pure, unit-tested seam; the audio is
//! decoded by the shared [`http::pcm_from_le_bytes`]. Behind the
//! `tts-speechmatics` feature.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use flowcat_core::error::{FlowcatError, Result};
use flowcat_core::processor::frame::{Frame, StartParams};
use flowcat_core::service::TtsService;

#[allow(clippy::duplicate_mod)] // each HTTP-TTS provider owns its own copy (feature-independent)
#[path = "http_tts_common.rs"]
pub mod http;

use http::{pcm_from_le_bytes, tts_frames, HttpTtsBody, HttpTtsClient, HttpTtsRequest};

/// Speechmatics' default TTS host (the public preview endpoint).
pub const SPEECHMATICS_TTS_BASE: &str = "https://preview.tts.speechmatics.com";
/// Speechmatics TTS streams 16 kHz PCM by default.
const SPEECHMATICS_SAMPLE_RATE: u32 = 16_000;

/// Speechmatics HTTP TTS service (stateless request/response).
pub struct SpeechmaticsTts {
    client: HttpTtsClient,
    api_key: String,
    base_url: String,
    voice: String,
    sample_rate: u32,
    ctx_counter: u64,
}

impl SpeechmaticsTts {
    /// Construct bound to `api_key` + `voice` (default preview host, 16 kHz, voice
    /// `sarah`).
    pub fn new(api_key: impl Into<String>, voice: impl Into<String>) -> Self {
        Self {
            client: HttpTtsClient::new("speechmatics"),
            api_key: api_key.into(),
            base_url: SPEECHMATICS_TTS_BASE.to_string(),
            voice: voice.into(),
            sample_rate: SPEECHMATICS_SAMPLE_RATE,
            ctx_counter: 0,
        }
    }

    /// Override the API base (trailing slash trimmed).
    pub fn base_url(mut self, base: impl Into<String>) -> Self {
        self.base_url = base.into().trim_end_matches('/').to_string();
        self
    }

    /// Override the output sample rate (default 16 kHz).
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    fn url(&self) -> String {
        build_url(&self.base_url, &self.voice, self.sample_rate)
    }
}

#[async_trait]
impl TtsService for SpeechmaticsTts {
    fn name(&self) -> &str {
        "speechmatics"
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        if self.api_key.is_empty() {
            return Err(FlowcatError::Session(
                "speechmatics tts: empty api key".into(),
            ));
        }
        Ok(())
    }

    async fn run_tts(&mut self, text: &str) -> Result<Vec<Frame>> {
        self.ctx_counter += 1;
        let context_id: Arc<str> = Arc::from(format!("ctx-{}", self.ctx_counter));
        let req = HttpTtsRequest {
            url: self.url(),
            headers: vec![(
                "Authorization".to_string(),
                format!("Bearer {}", self.api_key),
            )],
            body: HttpTtsBody::Json(build_body(text)),
        };
        let body = self.client.post(req).await?;
        Ok(tts_frames(
            pcm_from_le_bytes(&body),
            self.sample_rate,
            context_id,
        ))
    }
}

/// Build the `/generate/{voice}?output_format=pcm_{rate}` URL (pure seam).
pub fn build_url(base_url: &str, voice: &str, sample_rate: u32) -> String {
    format!("{base_url}/generate/{voice}?output_format=pcm_{sample_rate}")
}

/// Build the `{ "text": … }` body (pure seam).
pub fn build_body(text: &str) -> Value {
    json!({ "text": text })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_embeds_voice_and_pcm_format() {
        let url = build_url(SPEECHMATICS_TTS_BASE, "sarah", 16_000);
        assert_eq!(
            url,
            "https://preview.tts.speechmatics.com/generate/sarah?output_format=pcm_16000"
        );
    }

    #[test]
    fn body_carries_text() {
        assert_eq!(build_body("hello there")["text"], "hello there");
    }

    #[test]
    fn client_defaults() {
        let tts = SpeechmaticsTts::new("k", "sarah");
        assert_eq!(tts.name(), "speechmatics");
        assert_eq!(tts.sample_rate(), 16_000);
        assert!(tts.url().contains("/generate/sarah"));
    }

    #[tokio::test]
    async fn start_rejects_empty_key() {
        let mut tts = SpeechmaticsTts::new("", "sarah");
        assert!(tts.start(&StartParams::default()).await.is_err());
    }

    /// Live smoke (requires `SPEECHMATICS_API_KEY`). Run:
    /// `SPEECHMATICS_API_KEY=… cargo test -p flowcat-services --features tts-speechmatics -- --ignored speechmatics_tts_live`
    #[tokio::test]
    #[ignore = "requires SPEECHMATICS_API_KEY"]
    async fn speechmatics_tts_live_synthesizes_audio() {
        let key = std::env::var("SPEECHMATICS_API_KEY").expect("SPEECHMATICS_API_KEY");
        let mut tts = SpeechmaticsTts::new(key, "sarah");
        tts.start(&StartParams::default()).await.expect("start");
        let frames = tts.run_tts("Hello from flowcat.").await.expect("run_tts");
        assert!(frames.iter().any(|f| matches!(f, Frame::TtsAudio { .. })));
    }
}
