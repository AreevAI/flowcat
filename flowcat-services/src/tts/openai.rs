// SPDX-License-Identifier: Apache-2.0
//
//! **OpenAI** TTS — the HTTP-POST-audio **(D)istinct** family client.
//!
//! Unlike the streaming-WebSocket providers (the Cartesia template), the OpenAI
//! TTS family is **request/response synthesis**: POST the whole utterance text to
//! an `/audio/speech`-shaped endpoint with `Authorization: Bearer <key>` and read
//! the audio body back in one shot. The OpenAI request asks for raw `pcm`
//! (24 kHz, 16-bit LE mono) so the decode is a straight
//! [`http::pcm_from_le_bytes`]; the response is framed into
//! [`Frame::TtsStarted`] / [`Frame::TtsAudio`] / [`Frame::TtsStopped`].
//!
//! This is the family client that [`GroqTts`](super::GroqTts) and
//! [`XaiTts`](super::XaiTts) wrap — each is the same buffered-POST client with a
//! different `base_url`, response container, or request body (the `(W)rapper`
//! triage, PROVIDERS.md §3). The three axes that vary are captured by
//! [`AudioContainer`] (raw PCM vs WAV) and [`BodyShape`] (the `/audio/speech`
//! `{input,model,voice,response_format}` body vs xAI's `{text,voice_id,
//! output_format}` body) so a wrapper is a few `with_*` calls (~20 lines).
//!
//! The request encode + audio decode are **pure functions** ([`build_request`],
//! and the [`http`] container decoders) so the wire shape is unit-tested without a
//! network call. Behind the `tts-openai` feature.

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

/// OpenAI's default API base. A `base_url` override (config, never request-derived
/// → no SSRF surface) points the same client at an `/audio/speech`-compatible
/// gateway — exactly how the `(W)` wrappers reuse this client.
pub const OPENAI_API_BASE: &str = "https://api.openai.com/v1";

/// Which audio container the endpoint returns, so the decode knows whether to
/// strip a WAV header before reading PCM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioContainer {
    /// Raw little-endian `pcm_s16le` bytes (OpenAI `response_format: "pcm"`, xAI).
    RawPcm,
    /// A WAV file (RIFF header + PCM); the 44-byte header is stripped (Groq).
    Wav,
}

/// Which request-body schema the endpoint expects. Both auth with
/// `Authorization: Bearer <key>`; only the JSON field names + the path differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyShape {
    /// OpenAI `/audio/speech`: `{input, model, voice, response_format}` (Groq adds
    /// nothing — same body, just `response_format: "wav"`).
    OpenAiSpeech,
    /// xAI `/tts`: `{text, voice_id, output_format: {codec, sample_rate}}`.
    XaiTts,
}

/// Builder for an OpenAI-family HTTP-TTS client: API key + base URL + endpoint
/// path + model + voice + output sample rate + body shape + container, with
/// OpenAI defaults. The `(W)` wrappers build through this.
#[derive(Debug, Clone)]
pub struct OpenAiTtsBuilder {
    name: &'static str,
    api_key: String,
    base_url: String,
    endpoint: String,
    model: String,
    voice: String,
    sample_rate: u32,
    body_shape: BodyShape,
    container: AudioContainer,
}

impl OpenAiTtsBuilder {
    /// Start a builder bound to `api_key` + `voice` (OpenAI base + `/audio/speech`
    /// + `gpt-4o-mini-tts`, 24 kHz raw PCM, the pipecat OpenAI defaults).
    pub fn new(api_key: impl Into<String>, voice: impl Into<String>) -> Self {
        Self {
            name: "openai",
            api_key: api_key.into(),
            base_url: OPENAI_API_BASE.to_string(),
            endpoint: "/audio/speech".to_string(),
            model: "gpt-4o-mini-tts".to_string(),
            voice: voice.into(),
            sample_rate: 24_000,
            body_shape: BodyShape::OpenAiSpeech,
            container: AudioContainer::RawPcm,
        }
    }

    /// Override the provider name (error messages + `TtsService::name`).
    pub fn name(mut self, name: &'static str) -> Self {
        self.name = name;
        self
    }

    /// Override the API base (a wrapper's provider endpoint). Trailing slashes are
    /// trimmed so `{base}{endpoint}` is always well-formed.
    pub fn base_url(mut self, base: impl Into<String>) -> Self {
        self.base_url = base.into().trim_end_matches('/').to_string();
        self
    }

    /// Override the endpoint path (default `/audio/speech`; xAI uses `/tts`).
    pub fn endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Override the model (default `gpt-4o-mini-tts`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the output sample rate (default 24 kHz).
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Select the request-body schema (default OpenAI `/audio/speech`).
    pub fn body_shape(mut self, shape: BodyShape) -> Self {
        self.body_shape = shape;
        self
    }

    /// Select the response container (default raw PCM; Groq returns WAV).
    pub fn container(mut self, container: AudioContainer) -> Self {
        self.container = container;
        self
    }

    /// Build the (stateless) client.
    pub fn build(self) -> OpenAiTts {
        OpenAiTts {
            client: HttpTtsClient::new(self.name),
            name: self.name,
            api_key: self.api_key,
            url: format!("{}{}", self.base_url, self.endpoint),
            model: self.model,
            voice: self.voice,
            sample_rate: self.sample_rate,
            body_shape: self.body_shape,
            container: self.container,
            ctx_counter: 0,
        }
    }
}

/// An OpenAI-family HTTP-TTS service (stateless request/response).
pub struct OpenAiTts {
    client: HttpTtsClient,
    name: &'static str,
    api_key: String,
    url: String,
    model: String,
    voice: String,
    sample_rate: u32,
    body_shape: BodyShape,
    container: AudioContainer,
    ctx_counter: u64,
}

impl OpenAiTts {
    /// Construct with OpenAI defaults. Use [`OpenAiTtsBuilder`] for non-defaults.
    pub fn new(api_key: impl Into<String>, voice: impl Into<String>) -> Self {
        OpenAiTtsBuilder::new(api_key, voice).build()
    }

    /// The resolved POST URL (the wrapper tests assert on this).
    pub fn url(&self) -> &str {
        &self.url
    }

    /// The synthesis request body for `text` (pure — the wire-fixture seam).
    fn request_body(&self, text: &str) -> Value {
        build_request(
            text,
            &self.model,
            &self.voice,
            self.sample_rate,
            self.body_shape,
            self.container,
        )
    }

    /// Decode the response body into PCM per the configured container.
    fn decode_audio(&self, body: &[u8]) -> Vec<i16> {
        match self.container {
            AudioContainer::RawPcm => pcm_from_le_bytes(body),
            AudioContainer::Wav => pcm_from_le_bytes(strip_wav_header(body)),
        }
    }
}

#[async_trait]
impl TtsService for OpenAiTts {
    fn name(&self) -> &str {
        self.name
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        // Stateless: no socket to open. Validate we have a key (fail fast).
        if self.api_key.is_empty() {
            return Err(FlowcatError::Session(format!(
                "{} tts: empty api key",
                self.name
            )));
        }
        Ok(())
    }

    async fn run_tts(&mut self, text: &str) -> Result<Vec<Frame>> {
        self.ctx_counter += 1;
        let context_id: Arc<str> = Arc::from(format!("ctx-{}", self.ctx_counter));
        let req = HttpTtsRequest {
            url: self.url.clone(),
            headers: vec![(
                "Authorization".to_string(),
                format!("Bearer {}", self.api_key),
            )],
            body: HttpTtsBody::Json(self.request_body(text)),
        };
        let body = self.client.post(req).await?;
        let pcm = self.decode_audio(&body);
        Ok(tts_frames(pcm, self.sample_rate, context_id))
    }
}

/// Build the synthesis request body for `text` (pure — the wire-fixture seam).
/// Both shapes ask for raw/WAV PCM at `sample_rate`.
pub fn build_request(
    text: &str,
    model: &str,
    voice: &str,
    sample_rate: u32,
    body_shape: BodyShape,
    container: AudioContainer,
) -> Value {
    match body_shape {
        BodyShape::OpenAiSpeech => {
            let response_format = match container {
                AudioContainer::RawPcm => "pcm",
                AudioContainer::Wav => "wav",
            };
            json!({
                "input": text,
                "model": model,
                "voice": voice,
                "response_format": response_format,
            })
        }
        BodyShape::XaiTts => json!({
            "text": text,
            "voice_id": voice,
            "output_format": { "codec": "pcm", "sample_rate": sample_rate },
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_request_body_matches_audio_speech_schema() {
        let body = build_request(
            "hello there",
            "gpt-4o-mini-tts",
            "alloy",
            24_000,
            BodyShape::OpenAiSpeech,
            AudioContainer::RawPcm,
        );
        assert_eq!(body["input"], "hello there");
        assert_eq!(body["model"], "gpt-4o-mini-tts");
        assert_eq!(body["voice"], "alloy");
        assert_eq!(body["response_format"], "pcm");
    }

    #[test]
    fn wav_container_requests_wav_response_format() {
        let body = build_request(
            "hi",
            "m",
            "v",
            48_000,
            BodyShape::OpenAiSpeech,
            AudioContainer::Wav,
        );
        assert_eq!(body["response_format"], "wav");
    }

    #[test]
    fn xai_body_shape_uses_text_and_output_format() {
        let body = build_request(
            "hi",
            "m",
            "eve",
            16_000,
            BodyShape::XaiTts,
            AudioContainer::RawPcm,
        );
        assert_eq!(body["text"], "hi");
        assert_eq!(body["voice_id"], "eve");
        assert_eq!(body["output_format"]["codec"], "pcm");
        assert_eq!(body["output_format"]["sample_rate"], 16_000);
    }

    #[test]
    fn default_client_targets_openai_audio_speech() {
        let tts = OpenAiTts::new("k", "alloy");
        assert_eq!(tts.url(), "https://api.openai.com/v1/audio/speech");
        assert_eq!(tts.name(), "openai");
        assert_eq!(tts.sample_rate(), 24_000);
    }

    #[test]
    fn raw_pcm_decode_reads_le_i16() {
        let tts = OpenAiTts::new("k", "alloy");
        // 1 and -1 as LE i16.
        assert_eq!(tts.decode_audio(&[1, 0, 255, 255]), vec![1, -1]);
    }

    #[test]
    fn wav_decode_strips_header_then_reads_pcm() {
        let tts = OpenAiTtsBuilder::new("k", "v")
            .container(AudioContainer::Wav)
            .build();
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&[0u8; 4]);
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(&[0u8; 32]); // pad to 44
        wav.extend_from_slice(&[1, 0, 255, 255]); // 1, -1
        assert_eq!(tts.decode_audio(&wav), vec![1, -1]);
    }

    #[tokio::test]
    async fn start_rejects_empty_key() {
        let mut tts = OpenAiTts::new("", "alloy");
        assert!(tts.start(&StartParams::default()).await.is_err());
    }

    /// Live smoke (requires `OPENAI_API_KEY`): synthesize one short utterance and
    /// confirm audio came back. Run:
    /// `OPENAI_API_KEY=… cargo test -p flowcat-services --features tts-openai -- --ignored openai_tts_live`
    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY"]
    async fn openai_tts_live_synthesizes_audio() {
        let key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY");
        let mut tts = OpenAiTts::new(key, "alloy");
        tts.start(&StartParams::default()).await.expect("start");
        let frames = tts.run_tts("Hello from flowcat.").await.expect("run_tts");
        let audio = frames
            .iter()
            .filter(|f| matches!(f, Frame::TtsAudio { .. }))
            .count();
        assert!(audio > 0, "expected at least one TtsAudio chunk");
    }
}
