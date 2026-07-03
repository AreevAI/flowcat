// SPDX-License-Identifier: Apache-2.0
//
//! **Sarvam** STT (segmented REST, Indian-language ASR).
//!
//! A **(D)istinct** segmented-HTTP client (cross-checked against the Sarvam STT
//! API used by pipecat `services/sarvam/stt.py`). It buffers PCM across `run_stt`
//! calls and, once a segment's worth has accumulated, wraps it in a WAV file and
//! POSTs it as `multipart/form-data` to `https://api.sarvam.ai/speech-to-text`
//! with an `api-subscription-key: <api-key>` header and the form fields `model`
//! (default `saarika:v2.5`) + `language_code`. The `saaras:*` models (transcribe +
//! translate) post to `/speech-to-text-translate` instead, without a
//! `language_code` (that route auto-detects and translates to English). The JSON
//! response is decoded by the **pure** [`decode_response`]:
//!
//! ```json
//! { "transcript": "नमस्ते", "language_code": "hi-IN" }
//! ```
//!
//! A non-empty `transcript` → one final [`Frame::Transcription`]; anything else →
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

/// Sarvam's fixed STT endpoint. The **host is fixed**; the API key travels only
/// in the `api-subscription-key` header, never in the URL.
pub const SARVAM_STT_URL: &str = "https://api.sarvam.ai/speech-to-text";

/// Sarvam's transcribe-and-translate endpoint — the `saaras:*` models are served
/// here, not at the plain STT route (same host-is-fixed rule).
pub const SARVAM_STT_TRANSLATE_URL: &str = "https://api.sarvam.ai/speech-to-text-translate";

/// Endpoint for a model id: `saaras:*` (transcribe + translate) posts to the
/// translate route; everything else (`saarika:*`) to plain speech-to-text.
fn endpoint_for(model: &str) -> &'static str {
    if model.starts_with("saaras") {
        SARVAM_STT_TRANSLATE_URL
    } else {
        SARVAM_STT_URL
    }
}

/// Seconds of audio buffered per POSTed segment.
const SEGMENT_SECS: f32 = 5.0;

/// Sarvam segmented-HTTP STT service.
pub struct SarvamStt {
    api_key: String,
    sample_rate: u32,
    model: String,
    language: String,
    buffer: SegmentBuffer,
    muted: bool,
}

impl SarvamStt {
    /// Construct bound to `api_key` (default 16 kHz, `saarika:v2.5`, auto-detect).
    pub fn new(api_key: impl Into<String>) -> Self {
        let sample_rate = 16_000;
        Self {
            api_key: api_key.into(),
            sample_rate,
            model: "saarika:v2.5".to_string(),
            language: "unknown".to_string(),
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

    /// Override the model (default `saarika:v2.5`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the language code (default `unknown` = auto-detect).
    pub fn language(mut self, lang: impl Into<String>) -> Self {
        self.language = lang.into();
        self
    }

    async fn transcribe_segment(&mut self) -> Result<Vec<Frame>> {
        let Some(wav) = self.buffer.take_wav() else {
            return Ok(vec![]);
        };
        let boundary = "----flowcatSarvamBoundary7MA4YWxkTrZu0gW";
        let url = endpoint_for(&self.model);
        let mut parts = vec![
            Part::file("file", "audio.wav", "audio/x-wav", wav),
            Part::text("model", self.model.clone()),
        ];
        // The translate route auto-detects the source language (output is
        // English); `language_code` is a plain-STT-only field.
        if url == SARVAM_STT_URL {
            parts.push(Part::text("language_code", self.language.clone()));
        }
        let (content_type, body) = multipart_body(boundary, &parts);
        let client = reqwest::Client::new();
        let resp = client
            .post(url)
            .header("api-subscription-key", &self.api_key)
            .header("Content-Type", content_type)
            .body(body)
            .send()
            .await
            .map_err(|e| FlowcatError::Network(format!("sarvam stt: {e}")))?;
        if !resp.status().is_success() {
            let code = resp.status();
            return Err(FlowcatError::Network(format!(
                "sarvam stt failed: HTTP {code}"
            )));
        }
        let body: Value = resp
            .json()
            .await
            .map_err(|e| FlowcatError::Network(format!("sarvam stt body: {e}")))?;
        Ok(decode_response(&body))
    }

    /// Transcribe any remaining buffered audio (end-of-turn / shutdown).
    pub async fn flush(&mut self) -> Result<Vec<Frame>> {
        self.transcribe_segment().await
    }
}

#[async_trait]
impl SttService for SarvamStt {
    fn name(&self) -> &str {
        "sarvam"
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

/// Decode the Sarvam STT JSON response. **Pure.** A non-empty `transcript`
/// becomes one final [`Frame::Transcription`]; anything else → nothing.
pub(crate) fn decode_response(body: &Value) -> Vec<Frame> {
    let text = body
        .get("transcript")
        .and_then(|t| t.as_str())
        .unwrap_or("")
        .trim();
    if text.is_empty() {
        return vec![];
    }
    let language = body
        .get("language_code")
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
    fn saaras_models_route_to_the_translate_endpoint() {
        assert_eq!(endpoint_for("saarika:v2.5"), SARVAM_STT_URL);
        assert_eq!(endpoint_for("saaras:v2"), SARVAM_STT_TRANSLATE_URL);
        assert_eq!(
            endpoint_for(""),
            SARVAM_STT_URL,
            "empty model = default saarika"
        );
    }

    #[test]
    fn decode_transcript_response() {
        let body = json!({ "transcript": "book a dentist", "language_code": "hi-IN" });
        match &decode_response(&body)[..] {
            [Frame::Transcription {
                text,
                final_,
                language,
                ..
            }] => {
                assert_eq!(text, "book a dentist");
                assert!(final_);
                assert_eq!(language.as_ref().map(|l| l.0.as_str()), Some("hi-IN"));
            }
            other => panic!("expected final, got {other:?}"),
        }
    }

    #[test]
    fn decode_handles_null_language_and_ignores_empty_malformed() {
        // null language_code → no language, still a transcript.
        let body = json!({ "transcript": "hello", "language_code": null });
        assert!(matches!(&decode_response(&body)[..],
            [Frame::Transcription { language, .. }] if language.is_none()));
        assert!(decode_response(&json!({ "transcript": "" })).is_empty());
        assert!(decode_response(&json!({ "language_code": "hi-IN" })).is_empty());
        assert!(decode_response(&json!({ "error": { "message": "bad" } })).is_empty());
        assert!(decode_response(&json!("nope")).is_empty());
    }

    /// Live smoke (requires `SARVAM_API_KEY`). Run:
    /// `SARVAM_API_KEY=… cargo test -p flowcat-services --features stt-sarvam -- --ignored sarvam_live`
    #[tokio::test]
    #[ignore = "requires SARVAM_API_KEY"]
    async fn sarvam_live_transcribes_a_segment() {
        let key = std::env::var("SARVAM_API_KEY").expect("SARVAM_API_KEY");
        let mut stt = SarvamStt::new(key);
        stt.start(&StartParams::default()).await.expect("start");
        for _ in 0..6 {
            let chunk = Arc::new(AudioFrame::mono(vec![0i16; 16_000], 16_000));
            let _ = stt.run_stt(chunk).await.expect("run_stt");
        }
        let _ = stt.flush().await.expect("flush");
    }
}
