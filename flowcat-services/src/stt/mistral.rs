// SPDX-License-Identifier: Apache-2.0
//
//! **Mistral** STT (Voxtral — segmented REST).
//!
//! A **(D)istinct** segmented-HTTP client. It buffers PCM across `run_stt` calls
//! and, once a segment's worth has accumulated, wraps it in a WAV file and POSTs
//! it as `multipart/form-data` to `https://api.mistral.ai/v1/audio/transcriptions`
//! with an `Authorization: Bearer <api-key>` header and the form fields `model`
//! (default `voxtral-mini-2507`) + `file`. The JSON response is Whisper-shaped and
//! decoded by the **pure** [`decode_response`]:
//!
//! ```json
//! { "text": "book a dentist", "language": "en" }
//! ```
//!
//! A non-empty `text` → one final [`Frame::Transcription`]; anything else →
//! nothing. The buffering + WAV + multipart plumbing is the shared [`rest`] seam.

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

/// Mistral's fixed audio-transcription endpoint. The **host is fixed**; the API
/// key travels only in the `Authorization` header, never in the URL.
pub const MISTRAL_STT_URL: &str = "https://api.mistral.ai/v1/audio/transcriptions";

/// Seconds of audio buffered per POSTed segment.
const SEGMENT_SECS: f32 = 5.0;

/// Mistral (Voxtral) segmented-HTTP STT service.
pub struct MistralStt {
    api_key: String,
    sample_rate: u32,
    model: String,
    buffer: SegmentBuffer,
    muted: bool,
}

impl MistralStt {
    /// Construct bound to `api_key` (default 16 kHz input, `voxtral-mini-2507`).
    pub fn new(api_key: impl Into<String>) -> Self {
        let sample_rate = 16_000;
        Self {
            api_key: api_key.into(),
            sample_rate,
            model: "voxtral-mini-2507".to_string(),
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

    /// Override the transcription model (default `voxtral-mini-2507`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    async fn transcribe_segment(&mut self) -> Result<Vec<Frame>> {
        let Some(wav) = self.buffer.take_wav() else {
            return Ok(vec![]);
        };
        let boundary = "----flowcatMistralBoundary7MA4YWxkTrZu0gW";
        let parts = vec![
            Part::file("file", "audio.wav", "audio/x-wav", wav),
            Part::text("model", self.model.clone()),
        ];
        let (content_type, body) = multipart_body(boundary, &parts);
        let client = reqwest::Client::new();
        let resp = client
            .post(MISTRAL_STT_URL)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", content_type)
            .body(body)
            .send()
            .await
            .map_err(|e| FlowcatError::Network(format!("mistral stt: {e}")))?;
        if !resp.status().is_success() {
            let code = resp.status();
            return Err(FlowcatError::Network(format!(
                "mistral stt failed: HTTP {code}"
            )));
        }
        let body: Value = resp
            .json()
            .await
            .map_err(|e| FlowcatError::Network(format!("mistral stt body: {e}")))?;
        Ok(decode_response(&body))
    }

    /// Transcribe any remaining buffered audio (end-of-turn / shutdown).
    pub async fn flush(&mut self) -> Result<Vec<Frame>> {
        self.transcribe_segment().await
    }
}

#[async_trait]
impl SttService for MistralStt {
    fn name(&self) -> &str {
        "mistral"
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
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

/// Decode the Mistral (Whisper-shaped) JSON response. **Pure.** A non-empty
/// `text` becomes one final [`Frame::Transcription`]; anything else → nothing.
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
        .get("language")
        .and_then(|l| l.as_str())
        .filter(|l| !l.is_empty())
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
        let body = json!({ "text": "book a dentist", "language": "en" });
        match &decode_response(&body)[..] {
            [Frame::Transcription {
                text,
                final_,
                language,
                ..
            }] => {
                assert_eq!(text, "book a dentist");
                assert!(final_);
                assert_eq!(language.as_ref().map(|l| l.0.as_str()), Some("en"));
            }
            other => panic!("expected final, got {other:?}"),
        }
    }

    #[test]
    fn decode_ignores_empty_and_malformed() {
        assert!(decode_response(&json!({ "text": "" })).is_empty());
        assert!(decode_response(&json!({ "text": "  " })).is_empty());
        assert!(decode_response(&json!({ "language": "en" })).is_empty());
        assert!(decode_response(&json!({ "message": "Unauthorized" })).is_empty());
        assert!(decode_response(&json!("nope")).is_empty());
    }

    /// Live smoke (requires `MISTRAL_API_KEY`). Run:
    /// `MISTRAL_API_KEY=… cargo test -p flowcat-services --features stt-mistral -- --ignored mistral_live`
    #[tokio::test]
    #[ignore = "requires MISTRAL_API_KEY"]
    async fn mistral_live_transcribes_a_segment() {
        let key = std::env::var("MISTRAL_API_KEY").expect("MISTRAL_API_KEY");
        let mut stt = MistralStt::new(key);
        stt.start(&StartParams::default()).await.expect("start");
        for _ in 0..6 {
            let chunk = Arc::new(AudioFrame::mono(vec![0i16; 16_000], 16_000));
            let _ = stt.run_stt(chunk).await.expect("run_stt");
        }
        let _ = stt.flush().await.expect("flush");
    }
}
