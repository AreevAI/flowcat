// SPDX-License-Identifier: Apache-2.0
//
//! **ElevenLabs** STT (Scribe — segmented HTTP).
//!
//! A **(D)istinct** segmented-HTTP client (cross-checked against pipecat
//! `services/elevenlabs/stt.py`). It buffers PCM across `run_stt` calls and, once
//! a segment's worth has accumulated, wraps it in a WAV file and POSTs it as
//! `multipart/form-data` to `https://api.elevenlabs.io/v1/speech-to-text` with an
//! `xi-api-key: <api-key>` header and the form fields `model_id` (default
//! `scribe_v2`) + `language_code`. The JSON response is decoded by the **pure**
//! [`decode_response`]:
//!
//! ```json
//! { "text": "book a dentist", "language_code": "eng" }
//! ```
//!
//! A non-empty `text` → one final [`Frame::Transcription`]; an empty/absent
//! `text` (or any other shape) → nothing. The buffering + WAV + multipart
//! plumbing is the shared [`rest`] seam.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use flowcat_core::error::{FlowcatError, Result};
use flowcat_core::processor::frame::{AudioFrame, Frame, Language, StartParams};
use flowcat_core::service::SttService;

#[allow(clippy::duplicate_mod)] // each REST provider owns its own copy (feature-independent)
#[path = "rest_stt_common.rs"]
pub mod rest;

use rest::{multipart_body, Part, SegmentBuffer};

/// ElevenLabs' fixed Scribe STT endpoint. The **host is fixed**; the API key
/// travels only in the `xi-api-key` header, never in the URL.
pub const ELEVENLABS_STT_URL: &str = "https://api.elevenlabs.io/v1/speech-to-text";

/// Seconds of audio buffered per POSTed segment (the trait has no explicit
/// end-of-turn signal, so a segmented HTTP STT flushes on a size threshold).
const SEGMENT_SECS: f32 = 5.0;

/// ElevenLabs (Scribe) segmented-HTTP STT service.
pub struct ElevenLabsStt {
    api_key: String,
    sample_rate: u32,
    model: String,
    language: String,
    buffer: SegmentBuffer,
    muted: bool,
}

impl ElevenLabsStt {
    /// Construct bound to `api_key` (default 16 kHz input, `scribe_v2`, English).
    pub fn new(api_key: impl Into<String>) -> Self {
        let sample_rate = 16_000;
        Self {
            api_key: api_key.into(),
            sample_rate,
            model: "scribe_v2".to_string(),
            language: "eng".to_string(),
            buffer: SegmentBuffer::new(sample_rate, SEGMENT_SECS),
            muted: false,
        }
    }

    /// Override the input sample rate (default 16 kHz).
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self.buffer = SegmentBuffer::new(rate, SEGMENT_SECS);
        self
    }

    /// Override the Scribe model (default `scribe_v2`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the language code (default `eng`).
    pub fn language(mut self, lang: impl Into<String>) -> Self {
        self.language = lang.into();
        self
    }

    /// POST a buffered WAV segment and decode the transcript. Used by `run_stt`
    /// (on a full segment) and by [`flush`](Self::flush).
    async fn transcribe_segment(&mut self) -> Result<Vec<Frame>> {
        let Some(wav) = self.buffer.take_wav() else {
            return Ok(vec![]);
        };
        let boundary = "----flowcatElevenLabsBoundary7MA4YWxkTrZu0gW";
        let parts = vec![
            Part::file("file", "audio.wav", "audio/x-wav", wav),
            Part::text("model_id", self.model.clone()),
            Part::text("language_code", self.language.clone()),
        ];
        let (content_type, body) = multipart_body(boundary, &parts);
        let client = reqwest::Client::new();
        let resp = client
            .post(ELEVENLABS_STT_URL)
            .header("xi-api-key", &self.api_key)
            .header("Content-Type", content_type)
            .body(body)
            .send()
            .await
            .map_err(|e| FlowcatError::Network(format!("elevenlabs stt: {e}")))?;
        if !resp.status().is_success() {
            let code = resp.status();
            return Err(FlowcatError::Network(format!(
                "elevenlabs stt failed: HTTP {code}"
            )));
        }
        let body: Value = resp
            .json()
            .await
            .map_err(|e| FlowcatError::Network(format!("elevenlabs stt body: {e}")))?;
        Ok(decode_response(&body))
    }

    /// Transcribe any remaining buffered audio (call at end-of-turn / shutdown).
    pub async fn flush(&mut self) -> Result<Vec<Frame>> {
        self.transcribe_segment().await
    }
}

#[async_trait]
impl SttService for ElevenLabsStt {
    fn name(&self) -> &str {
        "elevenlabs"
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        // Segmented HTTP STT holds no persistent connection.
        Ok(())
    }

    async fn run_stt(&mut self, audio: Arc<AudioFrame>) -> Result<Vec<Frame>> {
        if self.muted {
            return Ok(vec![]);
        }
        self.buffer.push(&audio);
        if self.buffer.is_ready() {
            self.transcribe_segment().await
        } else {
            Ok(vec![])
        }
    }

    async fn set_muted(&mut self, muted: bool) {
        self.muted = muted;
    }
}

/// Decode the ElevenLabs Scribe JSON response. **Pure.** A non-empty `text`
/// becomes one final [`Frame::Transcription`]; anything else → nothing.
pub(crate) fn decode_response(body: &Value) -> Vec<Frame> {
    let text = body
        .get("text")
        .and_then(|t| t.as_str())
        .unwrap_or("")
        .trim();
    if text.is_empty() {
        return vec![];
    }
    let language = body
        .get("language_code")
        .and_then(|l| l.as_str())
        .map(|l| Language(l.to_string()));
    vec![Frame::Transcription {
        text: text.to_string(),
        user_id: Arc::from("user"),
        language,
        final_: true,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn decode_transcript_response() {
        let body = json!({ "text": "book a dentist appointment", "language_code": "eng" });
        match &decode_response(&body)[..] {
            [Frame::Transcription {
                text,
                final_,
                language,
                ..
            }] => {
                assert_eq!(text, "book a dentist appointment");
                assert!(final_);
                assert_eq!(language.as_ref().map(|l| l.0.as_str()), Some("eng"));
            }
            other => panic!("expected final, got {other:?}"),
        }
    }

    #[test]
    fn decode_ignores_empty_and_malformed() {
        assert!(decode_response(&json!({ "text": "" })).is_empty());
        assert!(decode_response(&json!({ "text": "   " })).is_empty());
        assert!(decode_response(&json!({ "language_code": "eng" })).is_empty());
        assert!(decode_response(&json!({ "error": "bad request" })).is_empty());
        assert!(decode_response(&json!("nope")).is_empty());
    }

    #[test]
    fn run_stt_buffers_until_segment_then_no_panic_on_small_chunks() {
        // Below threshold → nothing emitted, no network call.
        let mut stt = ElevenLabsStt::new("k");
        // We can only synchronously test the buffering decision via the buffer.
        let small = AudioFrame::mono(vec![0i16; 100], 16_000);
        stt.buffer.push(&small);
        assert!(!stt.buffer.is_ready());
    }

    /// Live smoke (requires `ELEVENLABS_API_KEY`): POST ~1s of silence, expect a
    /// (likely empty) transcript without error. Run:
    /// `ELEVENLABS_API_KEY=… cargo test -p flowcat-services --features stt-elevenlabs -- --ignored elevenlabs_live`
    #[tokio::test]
    #[ignore = "requires ELEVENLABS_API_KEY"]
    async fn elevenlabs_live_transcribes_a_segment() {
        let key = std::env::var("ELEVENLABS_API_KEY").expect("ELEVENLABS_API_KEY");
        let mut stt = ElevenLabsStt::new(key);
        stt.start(&StartParams::default()).await.expect("start");
        for _ in 0..6 {
            let chunk = Arc::new(AudioFrame::mono(vec![0i16; 16_000], 16_000));
            let _ = stt.run_stt(chunk).await.expect("run_stt");
        }
        let _ = stt.flush().await.expect("flush");
    }
}
