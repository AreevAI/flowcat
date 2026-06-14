// SPDX-License-Identifier: Apache-2.0
//
//! **Sarvam** TTS — a **(D)istinct** HTTP-POST-audio client.
//!
//! Sarvam's Indian-language TTS synthesizes a whole utterance with one POST to
//! `{base}/text-to-speech` (cross-checked against pipecat
//! `services/sarvam/tts.py`): an `api-subscription-key: <key>` header and a JSON
//! body `{ text, target_language_code, speaker, sample_rate, model }`. The JSON
//! response carries base64 audio under `audios[0]`; that decodes to a WAV file
//! whose 44-byte header is stripped to raw PCM. The request encode
//! ([`build_body`]) + the response decode ([`decode_response`]) are pure,
//! unit-tested seams over the shared [`http`] helpers. Behind the `tts-sarvam`
//! feature.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use flowcat_core::error::{FlowcatError, Result};
use flowcat_core::processor::frame::{Frame, StartParams};
use flowcat_core::service::TtsService;

#[allow(clippy::duplicate_mod)] // each HTTP-TTS provider owns its own copy (feature-independent)
#[path = "http_tts_common.rs"]
pub mod http;

use http::{
    base64_decode, pcm_from_le_bytes, strip_wav_header, tts_frames, HttpTtsBody, HttpTtsClient,
    HttpTtsRequest,
};

/// Sarvam's default API base. The key rides the `api-subscription-key` header.
pub const SARVAM_API_BASE: &str = "https://api.sarvam.ai";

/// Sarvam HTTP TTS service (stateless request/response).
pub struct SarvamTts {
    client: HttpTtsClient,
    api_key: String,
    base_url: String,
    model: String,
    speaker: String,
    language: String,
    sample_rate: u32,
    ctx_counter: u64,
}

impl SarvamTts {
    /// Construct bound to `api_key` + `speaker` (default model `bulbul:v2`,
    /// 22050 Hz, language `en-IN`). Sarvam TTS REQUIRES a real
    /// `target_language_code` (`en-IN`, `hi-IN`, …) — `unknown` (an STT concept) is
    /// rejected with a 400; override via [`language`](Self::language) for other langs.
    pub fn new(api_key: impl Into<String>, speaker: impl Into<String>) -> Self {
        Self {
            client: HttpTtsClient::new("sarvam"),
            api_key: api_key.into(),
            base_url: SARVAM_API_BASE.to_string(),
            model: "bulbul:v2".to_string(),
            speaker: speaker.into(),
            language: "en-IN".to_string(),
            sample_rate: 22_050,
            ctx_counter: 0,
        }
    }

    /// Override the model (default `bulbul:v2`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the target language code (e.g. `hi-IN`).
    pub fn language(mut self, language: impl Into<String>) -> Self {
        self.language = language.into();
        self
    }

    /// Override the output sample rate (default 22050 Hz).
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    fn url(&self) -> String {
        format!("{}/text-to-speech", self.base_url)
    }
}

#[async_trait]
impl TtsService for SarvamTts {
    fn name(&self) -> &str {
        "sarvam"
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        if self.api_key.is_empty() {
            return Err(FlowcatError::Session("sarvam tts: empty api key".into()));
        }
        Ok(())
    }

    async fn run_tts(&mut self, text: &str) -> Result<Vec<Frame>> {
        self.ctx_counter += 1;
        let context_id: Arc<str> = Arc::from(format!("ctx-{}", self.ctx_counter));
        let body = build_body(
            text,
            &self.language,
            &self.speaker,
            &self.model,
            self.sample_rate,
        );
        let req = HttpTtsRequest {
            url: self.url(),
            headers: vec![("api-subscription-key".to_string(), self.api_key.clone())],
            body: HttpTtsBody::Json(body),
        };
        let raw = self.client.post(req).await?;
        let pcm = decode_response(&raw);
        Ok(tts_frames(pcm, self.sample_rate, context_id))
    }
}

/// Build the Sarvam `/text-to-speech` request body (pure seam).
pub fn build_body(
    text: &str,
    language: &str,
    speaker: &str,
    model: &str,
    sample_rate: u32,
) -> Value {
    json!({
        "text": text,
        "target_language_code": language,
        "speaker": speaker,
        "sample_rate": sample_rate,
        "model": model,
    })
}

/// Decode the Sarvam JSON response body into PCM samples (pure seam). The first
/// `audios[]` entry is base64 → WAV; the header is stripped to raw PCM. Any
/// missing/non-string/garbage field yields no samples — never panics.
pub fn decode_response(body: &[u8]) -> Vec<i16> {
    let Ok(value) = serde_json::from_slice::<Value>(body) else {
        return Vec::new();
    };
    let Some(b64) = value
        .get("audios")
        .and_then(|a| a.as_array())
        .and_then(|a| a.first())
        .and_then(|a| a.as_str())
    else {
        return Vec::new();
    };
    let audio = base64_decode(b64);
    pcm_from_le_bytes(strip_wav_header(&audio))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_matches_sarvam_schema() {
        let body = build_body("नमस्ते", "hi-IN", "anushka", "bulbul:v2", 22_050);
        assert_eq!(body["text"], "नमस्ते");
        assert_eq!(body["target_language_code"], "hi-IN");
        assert_eq!(body["speaker"], "anushka");
        assert_eq!(body["model"], "bulbul:v2");
        assert_eq!(body["sample_rate"], 22_050);
    }

    #[test]
    fn decode_response_base64_wav_to_pcm() {
        // Build a WAV: 44-byte RIFF/WAVE header + samples 1, -1.
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&[0u8; 4]);
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(&[0u8; 32]);
        wav.extend_from_slice(&[1, 0, 255, 255]);
        let b64 = http::base64_encode(&wav);
        let json = serde_json::to_vec(&json!({ "audios": [b64] })).unwrap();
        assert_eq!(decode_response(&json), vec![1, -1]);
    }

    #[test]
    fn decode_response_tolerates_missing_audio() {
        assert!(decode_response(b"not json").is_empty());
        let empty = serde_json::to_vec(&json!({ "audios": [] })).unwrap();
        assert!(decode_response(&empty).is_empty());
    }

    #[test]
    fn client_defaults() {
        let tts = SarvamTts::new("k", "anushka");
        assert_eq!(tts.name(), "sarvam");
        assert_eq!(tts.sample_rate(), 22_050);
        assert_eq!(tts.url(), "https://api.sarvam.ai/text-to-speech");
    }

    #[tokio::test]
    async fn start_rejects_empty_key() {
        let mut tts = SarvamTts::new("", "anushka");
        assert!(tts.start(&StartParams::default()).await.is_err());
    }

    /// Live smoke (requires `SARVAM_API_KEY`). Run:
    /// `SARVAM_API_KEY=… cargo test -p flowcat-services --features tts-sarvam -- --ignored sarvam_tts_live`
    #[tokio::test]
    #[ignore = "requires SARVAM_API_KEY"]
    async fn sarvam_tts_live_synthesizes_audio() {
        let key = std::env::var("SARVAM_API_KEY").expect("SARVAM_API_KEY");
        let mut tts = SarvamTts::new(key, "anushka").language("hi-IN");
        tts.start(&StartParams::default()).await.expect("start");
        let frames = tts.run_tts("नमस्ते").await.expect("run_tts");
        assert!(frames.iter().any(|f| matches!(f, Frame::TtsAudio { .. })));
    }
}
