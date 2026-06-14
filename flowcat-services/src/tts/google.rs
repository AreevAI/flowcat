// SPDX-License-Identifier: Apache-2.0
//
//! **Google Cloud TTS** — REST client (`/v1/text:synthesize`).
//!
//! Google's gRPC TTS needs OAuth2 + generated codecs (infra-gated); the **REST**
//! endpoint takes a plain **API key** (`?key=…`) and returns base64 audio, which
//! drops straight into the shared `http_tts` seam. POST
//! `{base}/v1/text:synthesize?key=<api_key>` with a JSON body
//! `{ input: { text }, voice: { languageCode, name }, audioConfig: { audioEncoding:
//! "LINEAR16", sampleRateHertz } }`; the response `audioContent` is base64 LINEAR16
//! (a WAV container — the 44-byte header is stripped to raw little-endian PCM). The
//! request encode ([`build_body`]) + response decode ([`decode_response`]) are pure,
//! unit-tested seams. Behind the `tts-google` feature.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use flowcat_core::error::{FlowcatError, Result};
use flowcat_core::processor::frame::{Frame, StartParams};
use flowcat_core::service::TtsService;

#[allow(clippy::duplicate_mod)] // each provider owns its own copy (feature-independent)
#[path = "http_tts_common.rs"]
pub mod http;

use http::{
    base64_decode, pcm_from_le_bytes, strip_wav_header, tts_frames, HttpTtsBody, HttpTtsClient,
    HttpTtsRequest,
};

/// Google Cloud TTS REST host. The API key rides the `?key=` query param (host fixed
/// → no SSRF surface).
pub const GOOGLE_TTS_BASE: &str = "https://texttospeech.googleapis.com";

/// Google Cloud TTS service (REST `text:synthesize`).
pub struct GoogleTts {
    client: HttpTtsClient,
    api_key: String,
    base_url: String,
    voice_id: String,
    lang: String,
    sample_rate: u32,
    ctx_counter: u64,
}

impl GoogleTts {
    /// Construct bound to `api_key` + `voice_id` (default `en-US`, 24000 Hz LINEAR16).
    pub fn new(api_key: impl Into<String>, voice_id: impl Into<String>) -> Self {
        Self {
            client: HttpTtsClient::new("google"),
            api_key: api_key.into(),
            base_url: GOOGLE_TTS_BASE.to_string(),
            voice_id: voice_id.into(),
            lang: "en-US".to_string(),
            sample_rate: 24_000,
            ctx_counter: 0,
        }
    }

    /// Override the output sample rate (default 24000 Hz).
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Override the BCP-47 language code (default `en-US`).
    pub fn language(mut self, lang: impl Into<String>) -> Self {
        self.lang = lang.into();
        self
    }

    fn url(&self) -> String {
        format!("{}/v1/text:synthesize?key={}", self.base_url, self.api_key)
    }
}

/// Build the Google Cloud TTS `text:synthesize` request body (pure seam).
pub fn build_body(text: &str, lang: &str, voice: &str, sample_rate: u32) -> Value {
    json!({
        "input": { "text": text },
        "voice": { "languageCode": lang, "name": voice },
        "audioConfig": { "audioEncoding": "LINEAR16", "sampleRateHertz": sample_rate },
    })
}

/// Decode the JSON response's `audioContent` (base64 LINEAR16, WAV-wrapped) into raw
/// PCM samples (pure seam). Untrusted: malformed JSON / missing field → empty.
pub fn decode_response(body: &[u8]) -> Vec<i16> {
    let Ok(value) = serde_json::from_slice::<Value>(body) else {
        return Vec::new();
    };
    let Some(b64) = value.get("audioContent").and_then(|a| a.as_str()) else {
        return Vec::new();
    };
    pcm_from_le_bytes(strip_wav_header(&base64_decode(b64)))
}

#[async_trait]
impl TtsService for GoogleTts {
    fn name(&self) -> &str {
        "google"
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        if self.api_key.is_empty() {
            return Err(FlowcatError::Session("google tts: empty api key".into()));
        }
        Ok(())
    }

    async fn run_tts(&mut self, text: &str) -> Result<Vec<Frame>> {
        self.ctx_counter += 1;
        let context_id: Arc<str> = Arc::from(format!("ctx-{}", self.ctx_counter));
        let req = HttpTtsRequest {
            url: self.url(),
            headers: vec![], // API key is in the `?key=` query param
            body: HttpTtsBody::Json(build_body(
                text,
                &self.lang,
                &self.voice_id,
                self.sample_rate,
            )),
        };
        let raw = self.client.post(req).await?;
        Ok(tts_frames(
            decode_response(&raw),
            self.sample_rate,
            context_id,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_matches_google_schema() {
        let body = build_body("hello", "en-US", "en-US-Standard-C", 24_000);
        assert_eq!(body["input"]["text"], "hello");
        assert_eq!(body["voice"]["languageCode"], "en-US");
        assert_eq!(body["voice"]["name"], "en-US-Standard-C");
        assert_eq!(body["audioConfig"]["audioEncoding"], "LINEAR16");
        assert_eq!(body["audioConfig"]["sampleRateHertz"], 24_000);
    }

    #[test]
    fn url_carries_key_in_query() {
        let tts = GoogleTts::new("secret-key", "en-US-Standard-C");
        assert_eq!(
            tts.url(),
            "https://texttospeech.googleapis.com/v1/text:synthesize?key=secret-key"
        );
        assert_eq!(tts.sample_rate(), 24_000);
        assert_eq!(tts.name(), "google");
    }

    #[test]
    fn decode_response_extracts_audio_content() {
        // base64 of two LE i16 samples [1, -1] = bytes 01 00 ff ff (no WAV header).
        let body = br#"{"audioContent":"AQD//w=="}"#;
        assert_eq!(decode_response(body), vec![1, -1]);
        // Malformed / missing field → empty, never panics.
        assert!(decode_response(b"not json").is_empty());
        assert!(decode_response(br#"{"other":"x"}"#).is_empty());
    }

    #[tokio::test]
    async fn start_rejects_empty_key() {
        let mut tts = GoogleTts::new("", "en-US-Standard-C");
        assert!(tts.start(&StartParams::default()).await.is_err());
    }
}
