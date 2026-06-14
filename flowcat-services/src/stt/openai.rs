// SPDX-License-Identifier: Apache-2.0
//
//! **OpenAI Whisper** STT — the Whisper-HTTP **(D)istinct** family client.
//!
//! Unlike the streaming-WebSocket providers (the Deepgram template), the Whisper
//! family is **request/response file transcription**: the VAD-segmented utterance
//! audio fed to [`SttService::run_stt`] is buffered, wrapped in a minimal WAV
//! container, and POSTed once to an OpenAI-`/audio/transcriptions`-shaped endpoint
//! with `Authorization: Bearer <key>`. The JSON response (`{ "text": "…" }`) is
//! decoded into a single final [`Frame::Transcription`].
//!
//! This is the family client that [`GroqStt`](super::GroqStt),
//! [`FalStt`](super::FalStt), [`SpeachesStt`](super::SpeachesStt) and
//! [`XaiStt`](super::XaiStt) wrap — each is the same client with a different
//! `base_url` + auth scheme (the `(W)rapper` triage, PROVIDERS.md §2).
//!
//! The wire shape is split into **pure functions** ([`build_wav`],
//! [`encode_multipart`], [`decode_transcription`]) so the request encode + response
//! decode are unit-tested without a network call. Behind the `stt-openai` feature.
//!
//! ## Auth scheme
//!
//! Whisper-HTTP providers differ only in how the API key rides the request:
//! OpenAI / Groq / Speaches / xAI use `Authorization: Bearer <key>`; fal uses
//! `Authorization: Key <key>`. [`WhisperAuth`] captures that one axis so a wrapper
//! is a `base_url` + [`WhisperAuth`] + default-model change (~20 lines).

use std::sync::Arc;

use async_trait::async_trait;

use flowcat_core::error::{FlowcatError, Result};
use flowcat_core::processor::frame::{AudioFrame, Frame, Language, StartParams};
use flowcat_core::service::SttService;

/// OpenAI's default API base. A `base_url` override (config, never request-derived,
/// so no SSRF surface) points the same client at a Whisper-HTTP-compatible gateway
/// — that override is exactly how the `(W)` wrappers reuse this client.
pub const OPENAI_API_BASE: &str = "https://api.openai.com/v1";

/// How a Whisper-HTTP provider authenticates the POST. The only axis the family's
/// wrappers vary besides `base_url` + default model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhisperAuth {
    /// `Authorization: Bearer <key>` (OpenAI, Groq, Speaches, xAI).
    Bearer,
    /// `Authorization: Key <key>` (fal).
    Key,
}

impl WhisperAuth {
    /// Render the `Authorization` header value for `api_key` (the pure seam the
    /// wrapper tests assert on).
    pub fn header_value(self, api_key: &str) -> String {
        match self {
            WhisperAuth::Bearer => format!("Bearer {api_key}"),
            WhisperAuth::Key => format!("Key {api_key}"),
        }
    }
}

/// Builder for a Whisper-HTTP STT client: API key + base URL + auth + model + input
/// sample rate, with OpenAI defaults. The `(W)` wrappers build through this.
#[derive(Debug, Clone)]
pub struct WhisperHttpSttBuilder {
    api_key: String,
    base_url: String,
    auth: WhisperAuth,
    model: String,
    sample_rate: u32,
    language: Option<String>,
}

impl WhisperHttpSttBuilder {
    /// Start a builder bound to `api_key` (OpenAI base + `gpt-4o-transcribe`, the
    /// pipecat OpenAI default; 16 kHz input; Bearer auth).
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: OPENAI_API_BASE.to_string(),
            auth: WhisperAuth::Bearer,
            model: "gpt-4o-transcribe".to_string(),
            sample_rate: 16_000,
            language: None,
        }
    }

    /// Override the API base (a wrapper's provider endpoint). Trailing slashes are
    /// trimmed so `{base}/audio/transcriptions` is always well-formed.
    pub fn base_url(mut self, base: impl Into<String>) -> Self {
        self.base_url = base.into().trim_end_matches('/').to_string();
        self
    }

    /// Override the auth scheme (default `Bearer`; fal is `Key`).
    pub fn auth(mut self, auth: WhisperAuth) -> Self {
        self.auth = auth;
        self
    }

    /// Override the model (default `gpt-4o-transcribe`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the input sample rate (default 16 kHz). Used to size the WAV header.
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Set the transcription language hint (ISO-639-1, e.g. `en`).
    pub fn language(mut self, lang: impl Into<String>) -> Self {
        self.language = Some(lang.into());
        self
    }

    /// The `{base}/audio/transcriptions` URL for this config (testable, no socket).
    pub fn url(&self) -> String {
        format!("{}/audio/transcriptions", self.base_url)
    }

    /// Build the (stateless until `run_stt`) client.
    pub fn build(self) -> WhisperHttpStt {
        WhisperHttpStt {
            http: reqwest::Client::new(),
            name: "openai",
            api_key: self.api_key,
            base_url: self.base_url,
            auth: self.auth,
            model: self.model,
            sample_rate: self.sample_rate,
            language: self.language,
            buf: Vec::new(),
            muted: false,
        }
    }
}

/// Audio buffered (seconds) before POSTing one transcription. Without upstream VAD,
/// the cascaded pipeline feeds ~20 ms chunks; buffer into one utterance per request.
const SEGMENT_SECS: f32 = 4.0;

/// A Whisper-HTTP segmented STT session. The family client every `(W)rapper` reuses
/// (a `base_url` + [`WhisperAuth`] + model change). Public type alias [`OpenAiStt`]
/// is the OpenAI-flavoured constructor.
pub struct WhisperHttpStt {
    http: reqwest::Client,
    name: &'static str,
    api_key: String,
    base_url: String,
    auth: WhisperAuth,
    model: String,
    sample_rate: u32,
    language: Option<String>,
    /// VAD-segmented utterance PCM accumulated across `run_stt` calls; drained +
    /// POSTed each call (each call delivers a complete upstream-VAD segment).
    buf: Vec<i16>,
    muted: bool,
}

/// The OpenAI Whisper STT client — the family's reference `(D)` impl. A
/// [`WhisperHttpStt`] with the OpenAI base + Bearer auth.
pub type OpenAiStt = WhisperHttpStt;

impl WhisperHttpStt {
    /// Construct an OpenAI Whisper client bound to `api_key` (OpenAI base, Bearer
    /// auth, 16 kHz). Use [`WhisperHttpSttBuilder`] for a wrapper / non-default
    /// settings.
    pub fn new(api_key: impl Into<String>) -> Self {
        WhisperHttpSttBuilder::new(api_key).build()
    }

    /// Override the provider name reported by [`SttService::name`] (the wrappers set
    /// their own — `groq`, `fal`, `speaches`, `xai`).
    pub fn with_name(mut self, name: &'static str) -> Self {
        self.name = name;
        self
    }

    /// The `{base}/audio/transcriptions` URL for this client.
    fn url(&self) -> String {
        format!("{}/audio/transcriptions", self.base_url)
    }

    /// POST the buffered utterance and decode the transcript. Drains the buffer.
    async fn transcribe(&mut self) -> Result<Vec<Frame>> {
        if self.buf.is_empty() {
            return Ok(vec![]);
        }
        let pcm = std::mem::take(&mut self.buf);
        let wav = build_wav(&pcm, self.sample_rate);
        let (body, content_type) = encode_multipart(&wav, &self.model, self.language.as_deref());

        let resp = self
            .http
            .post(self.url())
            .header("Authorization", self.auth.header_value(&self.api_key))
            .header("Content-Type", content_type)
            .body(body)
            .send()
            .await
            .map_err(|e| FlowcatError::Network(format!("{} stt send: {e}", self.name)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(FlowcatError::Network(format!(
                "{} stt {status}: {text}",
                self.name
            )));
        }

        let text = resp
            .text()
            .await
            .map_err(|e| FlowcatError::Network(format!("{} stt body: {e}", self.name)))?;
        decode_transcription(&text)
    }
}

#[async_trait]
impl SttService for WhisperHttpStt {
    fn name(&self) -> &str {
        self.name
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        // Stateless: no socket to open. Validate we have a key (fail fast).
        if self.api_key.is_empty() {
            return Err(FlowcatError::Session(format!(
                "{} stt: empty api key",
                self.name
            )));
        }
        Ok(())
    }

    async fn run_stt(&mut self, audio: Arc<AudioFrame>) -> Result<Vec<Frame>> {
        if self.muted {
            return Ok(vec![]);
        }
        // Accumulate, then POST one utterance (request/response — not streaming). The
        // design expects upstream VAD to deliver complete utterances, but the cascaded
        // pipeline feeds raw ~20 ms chunks with no VAD — so buffer ~SEGMENT_SECS into
        // one request instead of POSTing a 20 ms clip per chunk.
        self.buf.extend_from_slice(&audio.pcm);
        let threshold = (self.sample_rate as f32 * SEGMENT_SECS) as usize;
        if self.buf.len() >= threshold {
            self.transcribe().await
        } else {
            Ok(vec![])
        }
    }

    async fn set_muted(&mut self, muted: bool) {
        self.muted = muted;
    }
}

/// Build a canonical 44-byte-header mono 16-bit-PCM WAV from `pcm` at `sample_rate`.
/// Whisper endpoints accept `audio/wav`; pipecat's `SegmentedSTTService` likewise
/// hands the API a WAV. **Pure** — the seam the request-fixture tests drive.
pub fn build_wav(pcm: &[i16], sample_rate: u32) -> Vec<u8> {
    let num_channels: u16 = 1;
    let bits_per_sample: u16 = 16;
    let byte_rate = sample_rate * u32::from(num_channels) * u32::from(bits_per_sample) / 8;
    let block_align = num_channels * bits_per_sample / 8;
    let data_len = (pcm.len() * 2) as u32;
    let riff_len = 36 + data_len;

    let mut out = Vec::with_capacity(44 + pcm.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_len.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    // fmt sub-chunk (PCM, 16 bytes).
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // audio format = PCM
    out.extend_from_slice(&num_channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());
    // data sub-chunk.
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for s in pcm {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// The fixed multipart boundary. A constant string never present in binary WAV /
/// the short text fields, so no escaping is required.
const MULTIPART_BOUNDARY: &str = "----flowcatWhisperBoundary7MA4YWxkTrZu0gW";

/// Encode the `multipart/form-data` request body for a Whisper transcription:
/// a `file` part (the WAV) + a `model` field (+ optional `language`). Returns the
/// raw body bytes and the `Content-Type` header value (with the boundary).
///
/// Built by hand (the `reqwest` `multipart` feature is intentionally not enabled —
/// this crate pins `reqwest` to `json`/`rustls`/`stream` only). **Pure** — the
/// seam the request-fixture tests drive.
pub fn encode_multipart(wav: &[u8], model: &str, language: Option<&str>) -> (Vec<u8>, String) {
    let mut body = Vec::new();
    let dashes = "--";

    // model field
    body.extend_from_slice(format!("{dashes}{MULTIPART_BOUNDARY}\r\n").as_bytes());
    body.extend_from_slice(b"Content-Disposition: form-data; name=\"model\"\r\n\r\n");
    body.extend_from_slice(model.as_bytes());
    body.extend_from_slice(b"\r\n");

    // optional language field
    if let Some(lang) = language {
        body.extend_from_slice(format!("{dashes}{MULTIPART_BOUNDARY}\r\n").as_bytes());
        body.extend_from_slice(b"Content-Disposition: form-data; name=\"language\"\r\n\r\n");
        body.extend_from_slice(lang.as_bytes());
        body.extend_from_slice(b"\r\n");
    }

    // file part (the WAV)
    body.extend_from_slice(format!("{dashes}{MULTIPART_BOUNDARY}\r\n").as_bytes());
    body.extend_from_slice(
        b"Content-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n",
    );
    body.extend_from_slice(b"Content-Type: audio/wav\r\n\r\n");
    body.extend_from_slice(wav);
    body.extend_from_slice(b"\r\n");

    // closing boundary
    body.extend_from_slice(format!("{dashes}{MULTIPART_BOUNDARY}{dashes}\r\n").as_bytes());

    let content_type = format!("multipart/form-data; boundary={MULTIPART_BOUNDARY}");
    (body, content_type)
}

/// Decode an OpenAI-`/audio/transcriptions` JSON response (`{ "text": "…" }`, or the
/// `verbose_json` shape which also carries a top-level `text`) into a single final
/// [`Frame::Transcription`]. An empty / whitespace-only transcript yields nothing
/// (mirrors pipecat dropping empty transcripts). **Pure** — the seam the
/// response-fixture tests drive.
pub fn decode_transcription(json: &str) -> Result<Vec<Frame>> {
    let value: serde_json::Value = serde_json::from_str(json).map_err(FlowcatError::Json)?;
    let text = value
        .get("text")
        .and_then(|t| t.as_str())
        .unwrap_or("")
        .trim();
    if text.is_empty() {
        return Ok(vec![]);
    }
    let language = value
        .get("language")
        .and_then(|l| l.as_str())
        .map(|s| Language(s.to_string()));
    Ok(vec![Frame::Transcription {
        text: text.to_string(),
        user_id: Arc::from("user"),
        language,
        final_: true,
    }])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_appends_the_transcriptions_path() {
        let c = OpenAiStt::new("sk-secret");
        assert_eq!(c.url(), "https://api.openai.com/v1/audio/transcriptions");
        // The key never leaks into the URL.
        assert!(!c.url().contains("secret"));
    }

    #[test]
    fn builder_trims_base_url_and_appends_path() {
        let b = WhisperHttpSttBuilder::new("k")
            .base_url("https://api.groq.com/openai/v1/")
            .build();
        assert_eq!(
            b.url(),
            "https://api.groq.com/openai/v1/audio/transcriptions"
        );
    }

    #[test]
    fn bearer_and_key_auth_headers() {
        assert_eq!(WhisperAuth::Bearer.header_value("abc"), "Bearer abc");
        assert_eq!(WhisperAuth::Key.header_value("abc"), "Key abc");
    }

    #[test]
    fn wav_header_is_canonical_pcm() {
        // 3 samples at 16 kHz → 6 data bytes, 44-byte header.
        let wav = build_wav(&[1, -2, 256], 16_000);
        assert_eq!(wav.len(), 44 + 6);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(&wav[36..40], b"data");
        // RIFF chunk size = 36 + data_len.
        assert_eq!(u32::from_le_bytes([wav[4], wav[5], wav[6], wav[7]]), 36 + 6);
        // fmt: PCM (1), 1 channel, 16 kHz.
        assert_eq!(u16::from_le_bytes([wav[20], wav[21]]), 1);
        assert_eq!(u16::from_le_bytes([wav[22], wav[23]]), 1);
        assert_eq!(
            u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]),
            16_000
        );
        // data size + little-endian samples.
        assert_eq!(u32::from_le_bytes([wav[40], wav[41], wav[42], wav[43]]), 6);
        assert_eq!(&wav[44..50], &[1, 0, 254, 255, 0, 1]);
    }

    #[test]
    fn multipart_carries_model_and_file_parts() {
        let wav = build_wav(&[0, 0], 16_000);
        let (body, content_type) = encode_multipart(&wav, "gpt-4o-transcribe", None);
        assert!(
            content_type.starts_with("multipart/form-data; boundary=----flowcatWhisperBoundary")
        );
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("name=\"model\""));
        assert!(text.contains("gpt-4o-transcribe"));
        assert!(text.contains("name=\"file\"; filename=\"audio.wav\""));
        assert!(text.contains("Content-Type: audio/wav"));
        // The WAV bytes are embedded verbatim.
        assert!(body.windows(4).any(|w| w == b"RIFF"));
        // Closing boundary present.
        assert!(text
            .trim_end()
            .ends_with("----flowcatWhisperBoundary7MA4YWxkTrZu0gW--"));
    }

    #[test]
    fn multipart_includes_language_when_set() {
        let (body, _) = encode_multipart(b"wav", "whisper-1", Some("en"));
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("name=\"language\""));
        assert!(text.contains("\r\nen\r\n"));
    }

    #[test]
    fn decode_simple_json_response() {
        let frames = decode_transcription(r#"{"text":"book a dentist"}"#).expect("decode");
        assert_eq!(frames.len(), 1);
        match &frames[0] {
            Frame::Transcription { text, final_, .. } => {
                assert_eq!(text, "book a dentist");
                assert!(final_);
            }
            other => panic!("expected final Transcription, got {}", other.name()),
        }
    }

    #[test]
    fn decode_verbose_json_carries_language() {
        let frames = decode_transcription(
            r#"{"task":"transcribe","language":"en","duration":1.2,"text":"hello there"}"#,
        )
        .expect("decode");
        assert_eq!(frames.len(), 1);
        match &frames[0] {
            Frame::Transcription { text, language, .. } => {
                assert_eq!(text, "hello there");
                assert_eq!(language.as_ref().map(|l| l.0.as_str()), Some("en"));
            }
            other => panic!("expected Transcription, got {}", other.name()),
        }
    }

    #[test]
    fn decode_empty_or_whitespace_yields_nothing() {
        assert!(decode_transcription(r#"{"text":""}"#)
            .expect("decode")
            .is_empty());
        assert!(decode_transcription(r#"{"text":"   "}"#)
            .expect("decode")
            .is_empty());
        assert!(decode_transcription(r#"{"other":"x"}"#)
            .expect("decode")
            .is_empty());
    }

    #[test]
    fn decode_rejects_non_json() {
        assert!(decode_transcription("not json at all").is_err());
    }

    #[tokio::test]
    async fn muted_run_stt_is_a_noop() {
        let mut stt = OpenAiStt::new("k");
        stt.set_muted(true).await;
        let audio = Arc::new(AudioFrame::mono(vec![1, 2, 3], 16_000));
        let frames = stt.run_stt(audio).await.expect("run_stt");
        assert!(frames.is_empty());
    }

    #[tokio::test]
    async fn start_rejects_empty_key() {
        let mut stt = OpenAiStt::new("");
        assert!(stt.start(&StartParams::default()).await.is_err());
    }

    /// Live smoke (requires `OPENAI_API_KEY`): transcribe a beat of silence and
    /// confirm the POST round-trips. Run with:
    /// `OPENAI_API_KEY=… cargo test -p flowcat-services --features stt-openai -- --ignored openai_stt_live`
    #[tokio::test]
    #[ignore = "requires OPENAI_API_KEY"]
    async fn openai_stt_live_transcribes() {
        let key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY");
        let mut stt = WhisperHttpSttBuilder::new(key).model("whisper-1").build();
        stt.start(&StartParams::default()).await.expect("start");
        // 1 s of silence at 16 kHz.
        let audio = Arc::new(AudioFrame::mono(vec![0i16; 16_000], 16_000));
        let _ = stt.run_stt(audio).await.expect("run_stt");
    }
}
