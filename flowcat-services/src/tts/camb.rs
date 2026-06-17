// SPDX-License-Identifier: Apache-2.0
//
//! **Camb** TTS — a **(D)istinct** HTTP-POST-audio client.
//!
//! Camb.ai's streaming TTS POSTs the utterance to `{base}/apis/tts-stream`
//! (cross-checked against pipecat `services/camb/tts.py`, which drives the same
//! endpoint via the Camb SDK): an `x-api-key: <key>` header and a JSON body
//! `{ text, voice_id, language, speech_model, output_configuration: { format:
//! "wav" } }`. We request `wav` (not `pcm_s16le`): Camb's raw `pcm_s16le` stream
//! comes back **truncated** (~40 % of the utterance), while `wav` returns the full
//! audio as a canonical 48 kHz RIFF/WAVE body. The header is stripped with
//! [`http::strip_wav_header`] and the PCM decoded by [`http::pcm_from_le_bytes`].
//! The request encode ([`build_body`]) is a pure, unit-tested seam. Behind the
//! `tts-camb` feature.

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
    pcm_from_le_bytes, strip_wav_header, tts_frames, HttpTtsBody, HttpTtsClient, HttpTtsRequest,
};

/// Camb.ai's default API base. The key rides the `x-api-key` header.
pub const CAMB_API_BASE: &str = "https://client.camb.ai";

/// Camb HTTP TTS service (stateless request/response, raw PCM body).
pub struct CambTts {
    client: HttpTtsClient,
    api_key: String,
    base_url: String,
    voice_id: String,
    language: String,
    model: String,
    sample_rate: u32,
    ctx_counter: u64,
}

impl CambTts {
    /// Construct bound to `api_key` + `voice_id` (default model `mars-instruct`,
    /// language `en-us`, 48000 Hz `pcm_s16le`). Camb requires a region-coded language
    /// (`en-us`/`en-au`/… — bare `en` is rejected with a 422); override via
    /// [`language`](Self::language).
    ///
    /// Camb **ignores** the requested `sample_rate` and always streams its native
    /// 48 kHz, so we tag the PCM at 48 kHz (declaring anything else plays the audio
    /// at the wrong speed/pitch).
    pub fn new(api_key: impl Into<String>, voice_id: impl Into<String>) -> Self {
        Self {
            client: HttpTtsClient::new("camb"),
            api_key: api_key.into(),
            base_url: CAMB_API_BASE.to_string(),
            voice_id: voice_id.into(),
            language: "en-us".to_string(),
            model: "mars-instruct".to_string(),
            sample_rate: 48_000,
            ctx_counter: 0,
        }
    }

    /// Override the language code (default `en`).
    pub fn language(mut self, language: impl Into<String>) -> Self {
        self.language = language.into();
        self
    }

    /// Override the speech model (default `mars-instruct`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the output sample rate (default 48000 Hz). Note Camb ignores this
    /// over the wire and always returns 48 kHz, so changing it only mis-tags the
    /// PCM — kept for API uniformity.
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    fn url(&self) -> String {
        format!("{}/apis/tts-stream", self.base_url)
    }
}

#[async_trait]
impl TtsService for CambTts {
    fn name(&self) -> &str {
        "camb"
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        if self.api_key.is_empty() {
            return Err(FlowcatError::Session("camb tts: empty api key".into()));
        }
        Ok(())
    }

    async fn run_tts(&mut self, text: &str) -> Result<Vec<Frame>> {
        self.ctx_counter += 1;
        let context_id: Arc<str> = Arc::from(format!("ctx-{}", self.ctx_counter));
        let req = HttpTtsRequest {
            url: self.url(),
            headers: vec![("x-api-key".to_string(), self.api_key.clone())],
            body: HttpTtsBody::Json(build_body(
                text,
                &self.voice_id,
                &self.language,
                &self.model,
                self.sample_rate,
            )),
        };
        let raw = self.client.post(req).await?;
        Ok(tts_frames(
            pcm_from_le_bytes(strip_wav_header(&raw)),
            self.sample_rate,
            context_id,
        ))
    }
}

/// Build the Camb `/apis/tts-stream` request body (pure seam). The output
/// `sample_rate` is still sent for completeness, but Camb ignores it and always
/// streams its native 48 kHz `pcm_s16le` — the caller must tag the PCM at 48 kHz.
pub fn build_body(
    text: &str,
    voice_id: &str,
    language: &str,
    model: &str,
    sample_rate: u32,
) -> Value {
    // Camb's API requires `voice_id` as an INTEGER (per the OpenAPI schema). Send it as
    // a JSON number when it parses; fall back to the raw string otherwise.
    let voice = voice_id
        .parse::<i64>()
        .map(Value::from)
        .unwrap_or_else(|_| Value::from(voice_id));
    json!({
        "text": text,
        "voice_id": voice,
        "language": language,
        "speech_model": model,
        "output_configuration": { "format": "wav", "sample_rate": sample_rate },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_matches_camb_schema() {
        let body = build_body("hello", "v-42", "en", "mars-instruct", 24_000);
        assert_eq!(body["text"], "hello");
        assert_eq!(body["voice_id"], "v-42");
        assert_eq!(body["language"], "en");
        assert_eq!(body["output_configuration"]["sample_rate"], 24_000);
        assert_eq!(body["speech_model"], "mars-instruct");
        assert_eq!(body["output_configuration"]["format"], "wav");
    }

    #[test]
    fn voice_id_is_sent_as_an_integer() {
        // Camb's schema requires voice_id: integer. A numeric voice id → JSON number.
        let body = build_body("hi", "147324", "en-us", "mars-instruct", 24_000);
        assert_eq!(body["voice_id"], 147324);
        assert!(
            body["voice_id"].is_number(),
            "voice_id must be an integer, not a string"
        );
    }

    #[test]
    fn client_defaults() {
        let tts = CambTts::new("k", "v-42");
        assert_eq!(tts.name(), "camb");
        assert_eq!(tts.sample_rate(), 48_000); // Camb's fixed native rate

        assert_eq!(tts.url(), "https://client.camb.ai/apis/tts-stream");
    }

    #[tokio::test]
    async fn start_rejects_empty_key() {
        let mut tts = CambTts::new("", "v-42");
        assert!(tts.start(&StartParams::default()).await.is_err());
    }

    /// Live smoke (requires `CAMB_API_KEY` + `CAMB_VOICE_ID`). Run:
    /// `CAMB_API_KEY=… CAMB_VOICE_ID=… cargo test -p flowcat-services --features tts-camb -- --ignored camb_tts_live`
    #[tokio::test]
    #[ignore = "requires CAMB_API_KEY + CAMB_VOICE_ID"]
    async fn camb_tts_live_synthesizes_audio() {
        let key = std::env::var("CAMB_API_KEY").expect("CAMB_API_KEY");
        let voice = std::env::var("CAMB_VOICE_ID").expect("CAMB_VOICE_ID");
        let mut tts = CambTts::new(key, voice);
        tts.start(&StartParams::default()).await.expect("start");
        let frames = tts.run_tts("Hello from flowcat.").await.expect("run_tts");
        assert!(frames.iter().any(|f| matches!(f, Frame::TtsAudio { .. })));
    }
}
