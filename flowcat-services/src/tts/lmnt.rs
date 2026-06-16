// SPDX-License-Identifier: Apache-2.0
//
//! **LMNT** TTS — interruptible HTTP client (Group H).
//!
//! LMNT's streaming WebSocket (pipecat `services/lmnt/tts.py`) is mirrored here by
//! its REST `speech/bytes` endpoint, because the Group-H `tts-lmnt` feature enables
//! only `reqwest`+`tokio` (no `tokio-tungstenite`):
//!
//! ```text
//! POST https://api.lmnt.com/v1/ai/speech/bytes
//!   X-API-Key: <key>
//!   { "text": "...", "voice": "...", "format": "wav", "sample_rate": 24000,
//!     "model": "aurora", "language": "en" }
//! ```
//!
//! With `format: "wav"` the response body is a 16-bit little-endian PCM WAV at the
//! requested sample rate (LMNT streams it with placeholder `0xFFFFFFFF` chunk
//! sizes); [`strip_wav_header`](tail::strip_wav_header) extracts the s16 data. The
//! `"raw"` format is avoided because it returns 32-bit float PCM
//! (`application/vnd.lmnt.audio-fp32`), which the s16 framer would misread.
//! [`build_payload`] / [`LmntTts::url`] are the **pure** seam.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use flowcat_core::error::{FlowcatError, Result};
use flowcat_core::processor::frame::{Frame, StartParams};
use flowcat_core::service::TtsService;

#[path = "tail_tts_common.rs"]
#[allow(clippy::duplicate_mod)] // shared header pattern: included into each Group-H module
mod tail;

/// LMNT REST synthesis endpoint.
pub const LMNT_BYTES_URL: &str = "https://api.lmnt.com/v1/ai/speech/bytes";
/// Default model.
pub const LMNT_DEFAULT_MODEL: &str = "aurora";

/// LMNT TTS service (HTTP REST, raw PCM).
pub struct LmntTts {
    api_key: String,
    voice_id: String,
    sample_rate: u32,
    model: String,
    lang: String,
    url: String,
    http: reqwest::Client,
    ctx_counter: u64,
}

impl LmntTts {
    /// Construct bound to `api_key` + `voice_id` (default 24000 Hz, aurora, English).
    pub fn new(api_key: impl Into<String>, voice_id: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            voice_id: voice_id.into(),
            sample_rate: 24_000,
            model: LMNT_DEFAULT_MODEL.to_string(),
            lang: "en".to_string(),
            url: LMNT_BYTES_URL.to_string(),
            http: reqwest::Client::new(),
            ctx_counter: 0,
        }
    }

    /// Override the model (default `aurora`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the output sample rate (default 24000 Hz).
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Override the language code (default `en`).
    pub fn language(mut self, lang: impl Into<String>) -> Self {
        self.lang = lang.into();
        self
    }

    fn url(&self) -> &str {
        &self.url
    }
}

/// Build the LMNT synthesis JSON body (pure — the request seam). `format: "wav"`
/// requests a 16-bit little-endian PCM WAV at the chosen sample rate. (LMNT's
/// `"raw"` format returns 32-bit float PCM — `application/vnd.lmnt.audio-fp32` —
/// which the s16 framer would misread as buzz, so we take the WAV and strip its
/// header.)
fn build_payload(text: &str, voice: &str, sample_rate: u32, model: &str, lang: &str) -> Value {
    json!({
        "text": text,
        "voice": voice,
        "format": "wav",
        "sample_rate": sample_rate,
        "model": model,
        "language": lang,
    })
}

#[async_trait]
impl TtsService for LmntTts {
    fn name(&self) -> &str {
        "lmnt"
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        Ok(())
    }

    async fn run_tts(&mut self, text: &str) -> Result<Vec<Frame>> {
        self.ctx_counter += 1;
        let context_id: Arc<str> = Arc::from(format!("ctx-{}", self.ctx_counter));
        let payload = build_payload(
            text,
            &self.voice_id,
            self.sample_rate,
            &self.model,
            &self.lang,
        );

        let resp = self
            .http
            .post(self.url())
            .header("X-API-Key", &self.api_key)
            .json(&payload)
            .send()
            .await
            .map_err(|e| FlowcatError::Network(format!("lmnt send: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(FlowcatError::Network(format!("lmnt http {status}: {body}")));
        }
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| FlowcatError::Network(format!("lmnt body: {e}")))?;
        // `format: wav` returns a 16-bit PCM WAV; strip the header to raw s16 PCM.
        let pcm = tail::strip_wav_header(&bytes);
        Ok(tail::one_shot_frames(pcm, self.sample_rate, context_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_matches_lmnt_schema() {
        let p = build_payload("hi", "voice-x", 24_000, "aurora", "en");
        assert_eq!(p["text"], "hi");
        assert_eq!(p["voice"], "voice-x");
        assert_eq!(p["format"], "wav");
        assert_eq!(p["sample_rate"], 24_000);
        assert_eq!(p["model"], "aurora");
        assert_eq!(p["language"], "en");
    }

    #[test]
    fn url_is_the_bytes_endpoint() {
        let t = LmntTts::new("k", "v");
        assert_eq!(t.url(), "https://api.lmnt.com/v1/ai/speech/bytes");
    }

    #[test]
    fn raw_pcm_body_frames_audio() {
        let frames = tail::one_shot_frames(&[1, 0, 255, 255], 24_000, Arc::from("c"));
        match &frames[1] {
            Frame::TtsAudio { audio, .. } => assert_eq!(audio.pcm, vec![1, -1]),
            _ => panic!("expected TtsAudio"),
        }
    }

    /// Live smoke (requires `LMNT_API_KEY` + `LMNT_VOICE_ID`).
    #[tokio::test]
    #[ignore = "requires LMNT_API_KEY + LMNT_VOICE_ID"]
    async fn lmnt_live_synthesizes_audio() {
        let key = std::env::var("LMNT_API_KEY").expect("LMNT_API_KEY");
        let voice = std::env::var("LMNT_VOICE_ID").expect("LMNT_VOICE_ID");
        let mut tts = LmntTts::new(key, voice);
        tts.start(&StartParams::default()).await.expect("start");
        let frames = tts.run_tts("Hello from flowcat.").await.expect("run_tts");
        assert!(frames.iter().any(|f| matches!(f, Frame::TtsAudio { .. })));
    }
}
