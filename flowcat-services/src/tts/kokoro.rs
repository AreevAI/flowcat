// SPDX-License-Identifier: Apache-2.0
//
//! **Kokoro** TTS — local-model HTTP-server client (Group H).
//!
//! Kokoro is a local ONNX voice (pipecat's `KokoroTTSService` loads `kokoro-v1.0.onnx`
//! in-process via `kokoro-onnx`). That in-process path needs a heavy native ONNX
//! runtime that is *not* on the `tts-kokoro` feature (which enables only
//! `reqwest`+`tokio`), so this client targets the widely-used **kokoro-fastapi**
//! server, which exposes an OpenAI-TTS-compatible endpoint:
//!
//! ```text
//! POST {base_url}/v1/audio/speech
//!   { "model": "kokoro", "input": "...", "voice": "<voice>", "response_format": "pcm" }
//!   → raw little-endian s16 PCM (24 kHz)
//! ```
//!
//! Point it at a running server with [`KokoroTts::with_base_url`]; without one,
//! `run_tts` returns a clear *"local model not wired"* error (no panic).
//! [`build_payload`] is the **pure** request seam.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use flowcat_core::error::{FlowcatError, Result};
use flowcat_core::processor::frame::{Frame, StartParams};
use flowcat_core::service::TtsService;

#[path = "tail_tts_common.rs"]
#[allow(clippy::duplicate_mod)] // shared header pattern: included into each Group-H module
mod tail;

/// Kokoro's native output sample rate (24 kHz).
pub const KOKORO_SAMPLE_RATE: u32 = 24_000;

/// Kokoro TTS service (local OpenAI-compatible HTTP server).
pub struct KokoroTts {
    voice_id: String,
    sample_rate: u32,
    base_url: Option<String>,
    http: reqwest::Client,
    ctx_counter: u64,
}

impl KokoroTts {
    /// Construct bound to `voice_id` (default 24000 Hz). Kokoro takes no API key;
    /// `_api_key` is accepted for a uniform constructor and ignored.
    pub fn new(_api_key: impl Into<String>, voice_id: impl Into<String>) -> Self {
        Self {
            voice_id: voice_id.into(),
            sample_rate: KOKORO_SAMPLE_RATE,
            base_url: None,
            http: reqwest::Client::new(),
            ctx_counter: 0,
        }
    }

    /// Point the client at a running kokoro-fastapi server (e.g.
    /// `http://localhost:8880`). Required.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        let mut url = base_url.into();
        while url.ends_with('/') {
            url.pop();
        }
        self.base_url = Some(url);
        self
    }

    /// Override the output sample rate (default 24000 Hz).
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }
}

/// Build the kokoro-fastapi (OpenAI-shaped) request body (pure — the request seam).
fn build_payload(text: &str, voice: &str) -> Value {
    json!({
        "model": "kokoro",
        "input": text,
        "voice": voice,
        "response_format": "pcm",
    })
}

#[async_trait]
impl TtsService for KokoroTts {
    fn name(&self) -> &str {
        "kokoro"
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        Ok(())
    }

    async fn run_tts(&mut self, text: &str) -> Result<Vec<Frame>> {
        let Some(base) = self.base_url.clone() else {
            return Err(FlowcatError::Other(
                "kokoro TTS: local model not wired — set a kokoro-fastapi URL via \
                 KokoroTts::with_base_url (in-process ONNX is not linked on tts-kokoro)"
                    .into(),
            ));
        };
        self.ctx_counter += 1;
        let context_id: Arc<str> = Arc::from(format!("ctx-{}", self.ctx_counter));
        let payload = build_payload(text, &self.voice_id);

        let resp = self
            .http
            .post(format!("{base}/v1/audio/speech"))
            .json(&payload)
            .send()
            .await
            .map_err(|e| FlowcatError::Network(format!("kokoro send: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(FlowcatError::Network(format!(
                "kokoro http {status}: {body}"
            )));
        }
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| FlowcatError::Network(format!("kokoro body: {e}")))?;
        let pcm = tail::strip_wav_header(&bytes);
        Ok(tail::one_shot_frames(pcm, self.sample_rate, context_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_matches_kokoro_fastapi_schema() {
        let p = build_payload("hi", "af_bella");
        assert_eq!(p["model"], "kokoro");
        assert_eq!(p["input"], "hi");
        assert_eq!(p["voice"], "af_bella");
        assert_eq!(p["response_format"], "pcm");
    }

    #[tokio::test]
    async fn without_base_url_returns_not_wired_seam() {
        let mut tts = KokoroTts::new("", "af_bella");
        let err = tts.run_tts("hi").await.unwrap_err();
        assert!(err.to_string().contains("local model not wired"));
    }

    #[test]
    fn with_base_url_trims_trailing_slash() {
        let tts = KokoroTts::new("", "v").with_base_url("http://localhost:8880//");
        assert_eq!(tts.base_url.as_deref(), Some("http://localhost:8880"));
    }

    /// Live smoke (requires a running kokoro-fastapi server at `KOKORO_BASE_URL`).
    #[tokio::test]
    #[ignore = "requires KOKORO_BASE_URL (kokoro-fastapi) + KOKORO_VOICE"]
    async fn kokoro_live_synthesizes_audio() {
        let base = std::env::var("KOKORO_BASE_URL").expect("KOKORO_BASE_URL");
        let voice = std::env::var("KOKORO_VOICE").expect("KOKORO_VOICE");
        let mut tts = KokoroTts::new("", voice).with_base_url(base);
        tts.start(&StartParams::default()).await.expect("start");
        let frames = tts.run_tts("Hello from flowcat.").await.expect("run_tts");
        assert!(frames.iter().any(|f| matches!(f, Frame::TtsAudio { .. })));
    }
}
