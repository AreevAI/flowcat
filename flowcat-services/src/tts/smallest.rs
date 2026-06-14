// SPDX-License-Identifier: Apache-2.0
//
//! **Smallest** TTS — interruptible HTTP client (Group H).
//!
//! Smallest's streaming WebSocket (pipecat `services/smallest/tts.py`) is mirrored
//! here by its REST `get_speech` endpoint, because the Group-H `tts-smallest` feature
//! enables only `reqwest`+`tokio` (no `tokio-tungstenite`):
//!
//! ```text
//! POST https://waves-api.smallest.ai/api/v1/{model}/get_speech
//!   Authorization: Bearer <key>
//!   { "text": "...", "voice_id": "...", "sample_rate": 24000, "format": "wav" }
//! ```
//!
//! The response is a one-shot WAV body; we strip the RIFF header to raw PCM s16le.
//! Request-encode ([`build_payload`] / [`SmallestTts::url`]) is the **pure** seam.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use flowcat_core::error::{FlowcatError, Result};
use flowcat_core::processor::frame::{Frame, StartParams};
use flowcat_core::service::TtsService;

#[path = "tail_tts_common.rs"]
#[allow(clippy::duplicate_mod)] // shared header pattern: included into each Group-H module
mod tail;

/// Smallest REST base URL.
pub const SMALLEST_HTTP_BASE: &str = "https://waves-api.smallest.ai/api/v1";
/// Default model (the `get_speech` path segment).
pub const SMALLEST_DEFAULT_MODEL: &str = "lightning";

/// Smallest TTS service (HTTP REST).
pub struct SmallestTts {
    api_key: String,
    voice_id: String,
    sample_rate: u32,
    model: String,
    base_url: String,
    http: reqwest::Client,
    ctx_counter: u64,
}

impl SmallestTts {
    /// Construct bound to `api_key` + `voice_id` (default 24000 Hz, lightning model).
    pub fn new(api_key: impl Into<String>, voice_id: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            voice_id: voice_id.into(),
            sample_rate: 24_000,
            model: SMALLEST_DEFAULT_MODEL.to_string(),
            base_url: SMALLEST_HTTP_BASE.to_string(),
            http: reqwest::Client::new(),
            ctx_counter: 0,
        }
    }

    /// Override the model (default `lightning`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the output sample rate (default 24000 Hz).
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    fn url(&self) -> String {
        format!("{}/{}/get_speech", self.base_url, self.model)
    }
}

/// Build the Smallest synthesis JSON body (pure — the request seam).
fn build_payload(text: &str, voice_id: &str, sample_rate: u32) -> Value {
    json!({
        "text": text,
        "voice_id": voice_id,
        "sample_rate": sample_rate,
        "format": "wav",
    })
}

#[async_trait]
impl TtsService for SmallestTts {
    fn name(&self) -> &str {
        "smallest"
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        // Stateless REST — nothing to open.
        Ok(())
    }

    async fn run_tts(&mut self, text: &str) -> Result<Vec<Frame>> {
        self.ctx_counter += 1;
        let context_id: Arc<str> = Arc::from(format!("ctx-{}", self.ctx_counter));
        let payload = build_payload(text, &self.voice_id, self.sample_rate);

        let resp = self
            .http
            .post(self.url())
            .bearer_auth(&self.api_key)
            .json(&payload)
            .send()
            .await
            .map_err(|e| FlowcatError::Network(format!("smallest send: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(FlowcatError::Network(format!(
                "smallest http {status}: {body}"
            )));
        }
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| FlowcatError::Network(format!("smallest body: {e}")))?;
        let pcm = tail::strip_wav_header(&bytes);
        Ok(tail::one_shot_frames(pcm, self.sample_rate, context_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_matches_smallest_schema() {
        let p = build_payload("hi", "voice-x", 24_000);
        assert_eq!(p["text"], "hi");
        assert_eq!(p["voice_id"], "voice-x");
        assert_eq!(p["sample_rate"], 24_000);
        assert_eq!(p["format"], "wav");
    }

    #[test]
    fn url_embeds_the_model() {
        let t = SmallestTts::new("k", "v").model("lightning-v2");
        assert_eq!(
            t.url(),
            "https://waves-api.smallest.ai/api/v1/lightning-v2/get_speech"
        );
    }

    #[test]
    fn wav_body_is_stripped_to_pcm() {
        // Minimal WAV with a 4-byte data payload (two LE samples 1 and -1).
        let mut w = Vec::new();
        w.extend_from_slice(b"RIFF");
        w.extend_from_slice(&[0, 0, 0, 0]);
        w.extend_from_slice(b"WAVE");
        w.extend_from_slice(b"data");
        w.extend_from_slice(&4u32.to_le_bytes());
        w.extend_from_slice(&[1, 0, 255, 255]);
        let frames = tail::one_shot_frames(tail::strip_wav_header(&w), 24_000, Arc::from("c"));
        match &frames[1] {
            Frame::TtsAudio { audio, .. } => assert_eq!(audio.pcm, vec![1, -1]),
            _ => panic!("expected TtsAudio"),
        }
    }

    /// Live smoke (requires `SMALLEST_API_KEY` + `SMALLEST_VOICE_ID`).
    #[tokio::test]
    #[ignore = "requires SMALLEST_API_KEY + SMALLEST_VOICE_ID"]
    async fn smallest_live_synthesizes_audio() {
        let key = std::env::var("SMALLEST_API_KEY").expect("SMALLEST_API_KEY");
        let voice = std::env::var("SMALLEST_VOICE_ID").expect("SMALLEST_VOICE_ID");
        let mut tts = SmallestTts::new(key, voice);
        tts.start(&StartParams::default()).await.expect("start");
        let frames = tts.run_tts("Hello from flowcat.").await.expect("run_tts");
        assert!(frames.iter().any(|f| matches!(f, Frame::TtsAudio { .. })));
    }
}
