// SPDX-License-Identifier: Apache-2.0
//
//! **Piper** TTS — local-model HTTP-server client (Group H).
//!
//! Piper is a local ONNX voice. pipecat ships both an in-process variant (loads the
//! `.onnx` voice) and `PiperHttpTTSService`, which talks to Piper's bundled HTTP
//! server (`python -m piper.http_server`). In-process inference would need a heavy
//! native model dependency that is *not* on the `tts-piper` feature (which enables
//! only `reqwest`+`tokio`), so this client implements the **HTTP-server form** and
//! exposes a clear *"local in-process model not wired"* seam (no panic) for the
//! native path:
//!
//! ```text
//! POST {base_url}
//!   { "text": "...", "voice": "<voice-id>" }   →  WAV body
//! ```
//!
//! Point it at a running server with [`PiperTts::with_base_url`]. The WAV header is
//! stripped to raw PCM. [`build_payload`] is the **pure** request seam.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use flowcat_core::error::{FlowcatError, Result};
use flowcat_core::processor::frame::{Frame, StartParams};
use flowcat_core::service::TtsService;

#[path = "tail_tts_common.rs"]
#[allow(clippy::duplicate_mod)] // shared header pattern: included into each Group-H module
mod tail;

/// Piper TTS service (local HTTP server). Piper's native sample rate is voice-
/// dependent (commonly 22050 Hz); the server returns it in the WAV header.
pub struct PiperTts {
    voice_id: String,
    sample_rate: u32,
    /// The local Piper HTTP-server base URL, if configured.
    base_url: Option<String>,
    http: reqwest::Client,
    ctx_counter: u64,
}

impl PiperTts {
    /// Construct bound to `voice_id` (default 22050 Hz). Piper takes no API key; the
    /// `_api_key` arg is accepted for a uniform constructor and ignored.
    pub fn new(_api_key: impl Into<String>, voice_id: impl Into<String>) -> Self {
        Self {
            voice_id: voice_id.into(),
            sample_rate: 22_050,
            base_url: None,
            http: reqwest::Client::new(),
            ctx_counter: 0,
        }
    }

    /// Point the client at a running Piper HTTP server (e.g.
    /// `http://localhost:5000`). Required — without it `run_tts` returns a clear
    /// "local model not wired" error rather than panicking.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        let mut url = base_url.into();
        while url.ends_with('/') {
            url.pop();
        }
        self.base_url = Some(url);
        self
    }

    /// Override the output sample rate (default 22050 Hz).
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }
}

/// Build the Piper HTTP-server request body (pure — the request seam).
fn build_payload(text: &str, voice: &str) -> Value {
    json!({ "text": text, "voice": voice })
}

#[async_trait]
impl TtsService for PiperTts {
    fn name(&self) -> &str {
        "piper"
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        Ok(())
    }

    async fn run_tts(&mut self, text: &str) -> Result<Vec<Frame>> {
        let Some(base) = self.base_url.clone() else {
            // In-process ONNX inference is the heavy native path we don't link; the
            // HTTP-server seam needs a URL. Clear, non-panicking signal.
            return Err(FlowcatError::Other(
                "piper TTS: local model not wired — set a Piper HTTP-server URL via \
                 PiperTts::with_base_url (in-process ONNX is not linked on tts-piper)"
                    .into(),
            ));
        };
        self.ctx_counter += 1;
        let context_id: Arc<str> = Arc::from(format!("ctx-{}", self.ctx_counter));
        let payload = build_payload(text, &self.voice_id);

        let resp = self
            .http
            .post(&base)
            .json(&payload)
            .send()
            .await
            .map_err(|e| FlowcatError::Network(format!("piper send: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(FlowcatError::Network(format!(
                "piper http {status}: {body}"
            )));
        }
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| FlowcatError::Network(format!("piper body: {e}")))?;
        let pcm = tail::strip_wav_header(&bytes);
        Ok(tail::one_shot_frames(pcm, self.sample_rate, context_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_matches_piper_http_schema() {
        let p = build_payload("hi", "en_US-ryan-high");
        assert_eq!(p["text"], "hi");
        assert_eq!(p["voice"], "en_US-ryan-high");
    }

    #[tokio::test]
    async fn without_base_url_returns_not_wired_seam() {
        let mut tts = PiperTts::new("", "en_US-ryan-high");
        let err = tts.run_tts("hi").await.unwrap_err();
        assert!(err.to_string().contains("local model not wired"));
    }

    #[test]
    fn with_base_url_trims_trailing_slash() {
        let tts = PiperTts::new("", "v").with_base_url("http://localhost:5000/");
        assert_eq!(tts.base_url.as_deref(), Some("http://localhost:5000"));
    }

    #[test]
    fn wav_body_strips_to_pcm() {
        let mut w = Vec::new();
        w.extend_from_slice(b"RIFF");
        w.extend_from_slice(&[0, 0, 0, 0]);
        w.extend_from_slice(b"WAVE");
        w.extend_from_slice(b"data");
        w.extend_from_slice(&4u32.to_le_bytes());
        w.extend_from_slice(&[1, 0, 255, 255]);
        assert_eq!(tail::pcm_s16le(tail::strip_wav_header(&w)), vec![1, -1]);
    }

    /// Live smoke (requires a running Piper HTTP server at `PIPER_BASE_URL`).
    #[tokio::test]
    #[ignore = "requires PIPER_BASE_URL (a running piper.http_server) + PIPER_VOICE"]
    async fn piper_live_synthesizes_audio() {
        let base = std::env::var("PIPER_BASE_URL").expect("PIPER_BASE_URL");
        let voice = std::env::var("PIPER_VOICE").expect("PIPER_VOICE");
        let mut tts = PiperTts::new("", voice).with_base_url(base);
        tts.start(&StartParams::default()).await.expect("start");
        let frames = tts.run_tts("Hello from flowcat.").await.expect("run_tts");
        assert!(frames.iter().any(|f| matches!(f, Frame::TtsAudio { .. })));
    }
}
