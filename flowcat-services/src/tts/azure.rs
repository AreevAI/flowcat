// SPDX-License-Identifier: Apache-2.0
//
//! **Azure Speech** TTS — a **(D)istinct** SSML HTTP client.
//!
//! Azure Cognitive Services TTS over its REST endpoint
//! `https://{region}.tts.speech.microsoft.com/cognitiveservices/v1`
//! (cross-checked against pipecat `services/azure/tts.py`, whose
//! `AzureHttpTTSService` drives the same service via the Speech SDK). Unlike the
//! JSON-body providers, **Azure's body is SSML** (`Content-Type:
//! application/ssml+xml`); auth is the `Ocp-Apim-Subscription-Key: <key>` header
//! and the output container is selected by the `X-Microsoft-OutputFormat:
//! raw-{rate}-16bit-mono-pcm` header — so the response is **raw little-endian
//! PCM** (no WAV header on the `raw-*` formats), decoded straight by
//! [`http::pcm_from_le_bytes`].
//!
//! ## SSML / auth seam (reviewer note)
//!
//! The one bespoke piece is [`build_ssml`]: it composes the
//! `<speak><voice>…</voice></speak>` document and **XML-escapes the synthesis
//! text** ([`escape_xml`]) so the caller's text can never break out of the SSML
//! (the five reserved chars `& < > " '` are entity-encoded). The auth header
//! ([`Ocp-Apim-Subscription-Key`](AZURE_KEY_HEADER)) + the output-format header
//! ([`output_format`]) are the other two seams. All three are pure, unit-tested
//! functions. Behind the `tts-azure` feature.

use std::sync::Arc;

use async_trait::async_trait;

use flowcat_core::error::{FlowcatError, Result};
use flowcat_core::processor::frame::{Frame, StartParams};
use flowcat_core::service::TtsService;

#[allow(clippy::duplicate_mod)] // each HTTP-TTS provider owns its own copy (feature-independent)
#[path = "http_tts_common.rs"]
pub mod http;

use http::{pcm_from_le_bytes, tts_frames, HttpTtsBody, HttpTtsClient, HttpTtsRequest};

/// Azure's subscription-key auth header.
pub const AZURE_KEY_HEADER: &str = "Ocp-Apim-Subscription-Key";
/// The output-format selector header.
pub const AZURE_FORMAT_HEADER: &str = "X-Microsoft-OutputFormat";

/// Azure HTTP TTS service (stateless request/response, SSML body, raw PCM out).
pub struct AzureTts {
    client: HttpTtsClient,
    api_key: String,
    region: String,
    voice: String,
    language: String,
    sample_rate: u32,
    ctx_counter: u64,
}

impl AzureTts {
    /// Construct bound to `api_key` + `region` + `voice` (default language
    /// `en-US`, 24 kHz raw PCM — the pipecat Azure defaults).
    pub fn new(
        api_key: impl Into<String>,
        region: impl Into<String>,
        voice: impl Into<String>,
    ) -> Self {
        Self {
            client: HttpTtsClient::new("azure"),
            api_key: api_key.into(),
            region: region.into(),
            voice: voice.into(),
            language: "en-US".to_string(),
            sample_rate: 24_000,
            ctx_counter: 0,
        }
    }

    /// Override the synthesis language (default `en-US`).
    pub fn language(mut self, language: impl Into<String>) -> Self {
        self.language = language.into();
        self
    }

    /// Override the output sample rate (default 24 kHz). Azure's `raw-*` PCM
    /// formats support 8000/16000/24000/48000; [`output_format`] falls back to
    /// 24 kHz for any other value.
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    fn url(&self) -> String {
        build_url(&self.region)
    }
}

#[async_trait]
impl TtsService for AzureTts {
    fn name(&self) -> &str {
        "azure"
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        if self.api_key.is_empty() {
            return Err(FlowcatError::Session("azure tts: empty api key".into()));
        }
        if self.region.is_empty() {
            return Err(FlowcatError::Session("azure tts: empty region".into()));
        }
        Ok(())
    }

    async fn run_tts(&mut self, text: &str) -> Result<Vec<Frame>> {
        self.ctx_counter += 1;
        let context_id: Arc<str> = Arc::from(format!("ctx-{}", self.ctx_counter));
        let ssml = build_ssml(text, &self.voice, &self.language);
        let req = HttpTtsRequest {
            url: self.url(),
            headers: vec![
                (AZURE_KEY_HEADER.to_string(), self.api_key.clone()),
                (
                    AZURE_FORMAT_HEADER.to_string(),
                    output_format(self.sample_rate).to_string(),
                ),
            ],
            body: HttpTtsBody::Raw {
                content_type: "application/ssml+xml",
                body: ssml,
            },
        };
        let raw = self.client.post(req).await?;
        Ok(tts_frames(
            pcm_from_le_bytes(&raw),
            self.sample_rate,
            context_id,
        ))
    }
}

/// The per-region synthesis endpoint (pure seam — the region is validated config,
/// never request-derived).
pub fn build_url(region: &str) -> String {
    format!("https://{region}.tts.speech.microsoft.com/cognitiveservices/v1")
}

/// The `X-Microsoft-OutputFormat` value for a raw-PCM rate. Azure only offers
/// `raw-*-16bit-mono-pcm` at a fixed set of rates; an unsupported rate falls back
/// to 24 kHz (the service default).
pub fn output_format(sample_rate: u32) -> &'static str {
    match sample_rate {
        8000 => "raw-8khz-16bit-mono-pcm",
        16000 => "raw-16khz-16bit-mono-pcm",
        22050 => "raw-22050hz-16bit-mono-pcm",
        48000 => "raw-48khz-16bit-mono-pcm",
        _ => "raw-24khz-16bit-mono-pcm",
    }
}

/// Build the SSML synthesis document for `text` in `voice` + `language` (pure
/// seam). The text is XML-escaped so it cannot break out of the document.
pub fn build_ssml(text: &str, voice: &str, language: &str) -> String {
    format!(
        "<speak version='1.0' xml:lang='{lang}' \
         xmlns='http://www.w3.org/2001/10/synthesis'>\
         <voice name='{voice}'>{body}</voice></speak>",
        lang = escape_xml(language),
        voice = escape_xml(voice),
        body = escape_xml(text),
    )
}

/// XML-escape the five SSML reserved characters (`& < > " '`). Order matters — `&`
/// is replaced first so the `&amp;` it introduces is not re-escaped.
pub fn escape_xml(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_embeds_region() {
        assert_eq!(
            build_url("eastus"),
            "https://eastus.tts.speech.microsoft.com/cognitiveservices/v1"
        );
    }

    #[test]
    fn output_format_maps_supported_rates_and_falls_back() {
        assert_eq!(output_format(16000), "raw-16khz-16bit-mono-pcm");
        assert_eq!(output_format(48000), "raw-48khz-16bit-mono-pcm");
        assert_eq!(output_format(24000), "raw-24khz-16bit-mono-pcm");
        // Unsupported rate → 24 kHz default.
        assert_eq!(output_format(12345), "raw-24khz-16bit-mono-pcm");
    }

    #[test]
    fn escape_xml_encodes_all_reserved_chars_amp_first() {
        assert_eq!(
            escape_xml("a & b < c > d \" e ' f"),
            "a &amp; b &lt; c &gt; d &quot; e &apos; f"
        );
        // `&` must be escaped first (no double-escaping of the entities it adds).
        assert_eq!(escape_xml("<&>"), "&lt;&amp;&gt;");
    }

    #[test]
    fn ssml_wraps_escaped_text_in_voice() {
        let ssml = build_ssml("Tom & Jerry <3", "en-US-SaraNeural", "en-US");
        assert!(ssml.starts_with("<speak version='1.0' xml:lang='en-US'"));
        assert!(ssml.contains("<voice name='en-US-SaraNeural'>"));
        // The text is escaped — no raw `&` or `<` from the input survives.
        assert!(ssml.contains("Tom &amp; Jerry &lt;3"));
        assert!(ssml.ends_with("</voice></speak>"));
        // No unescaped reserved char from the body leaks into the document.
        assert!(!ssml.contains("Tom & Jerry"));
    }

    #[test]
    fn client_defaults() {
        let tts = AzureTts::new("k", "eastus", "en-US-SaraNeural");
        assert_eq!(tts.name(), "azure");
        assert_eq!(tts.sample_rate(), 24_000);
        assert_eq!(
            tts.url(),
            "https://eastus.tts.speech.microsoft.com/cognitiveservices/v1"
        );
    }

    #[tokio::test]
    async fn start_rejects_empty_key_or_region() {
        let mut no_key = AzureTts::new("", "eastus", "v");
        assert!(no_key.start(&StartParams::default()).await.is_err());
        let mut no_region = AzureTts::new("k", "", "v");
        assert!(no_region.start(&StartParams::default()).await.is_err());
    }

    /// Live smoke (requires `AZURE_SPEECH_KEY` + `AZURE_SPEECH_REGION`). Run:
    /// `AZURE_SPEECH_KEY=… AZURE_SPEECH_REGION=… cargo test -p flowcat-services --features tts-azure -- --ignored azure_tts_live`
    #[tokio::test]
    #[ignore = "requires AZURE_SPEECH_KEY + AZURE_SPEECH_REGION"]
    async fn azure_tts_live_synthesizes_audio() {
        let key = std::env::var("AZURE_SPEECH_KEY").expect("AZURE_SPEECH_KEY");
        let region = std::env::var("AZURE_SPEECH_REGION").expect("AZURE_SPEECH_REGION");
        let mut tts = AzureTts::new(key, region, "en-US-SaraNeural");
        tts.start(&StartParams::default()).await.expect("start");
        let frames = tts.run_tts("Hello from flowcat.").await.expect("run_tts");
        assert!(frames.iter().any(|f| matches!(f, Frame::TtsAudio { .. })));
    }
}
