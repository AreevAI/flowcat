// SPDX-License-Identifier: Apache-2.0
//
//! **Neuphonic** TTS — interruptible HTTP/SSE client (Group H).
//!
//! Neuphonic exposes both a streaming WebSocket and an HTTP **Server-Sent-Events**
//! synthesis endpoint (pipecat `services/neuphonic/tts.py` → `NeuphonicHttpTTSService`).
//! The Group-H `tts-neuphonic` feature enables only `reqwest`+`tokio`, so this client
//! takes the SSE path:
//!
//! ```text
//! POST https://api.neuphonic.com/sse/speak/{lang}
//!   X-API-KEY: <key>
//!   { "text": "...", "voice_id": "...", "encoding": "pcm_linear",
//!     "sampling_rate": 24000, "lang_code": "en" }
//! ```
//!
//! The response is an SSE stream whose `data:` events carry JSON
//! `{ "data": { "audio": "<base64 pcm_s16le>" } }`. Request-encode
//! ([`build_payload`]) and response-decode ([`audio_from_sse_body`]) are **pure
//! functions** unit-tested without a network.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use flowcat_core::error::{FlowcatError, Result};
use flowcat_core::processor::frame::{Frame, StartParams};
use flowcat_core::service::TtsService;

#[path = "tail_tts_common.rs"]
#[allow(clippy::duplicate_mod)] // shared header pattern: included into each Group-H module
mod tail;

/// Neuphonic HTTP/SSE base URL.
pub const NEUPHONIC_HTTP_BASE: &str = "https://api.neuphonic.com";
/// Default audio encoding (linear PCM s16le) — the only format we decode.
pub const NEUPHONIC_ENCODING: &str = "pcm_linear";

/// Neuphonic TTS service (HTTP/SSE).
pub struct NeuphonicTts {
    api_key: String,
    voice_id: String,
    sample_rate: u32,
    lang: String,
    base_url: String,
    http: reqwest::Client,
    ctx_counter: u64,
}

impl NeuphonicTts {
    /// Construct bound to `api_key` + `voice_id` (default 22050 Hz, English).
    pub fn new(api_key: impl Into<String>, voice_id: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            voice_id: voice_id.into(),
            sample_rate: 22_050,
            lang: "en".to_string(),
            base_url: NEUPHONIC_HTTP_BASE.to_string(),
            http: reqwest::Client::new(),
            ctx_counter: 0,
        }
    }

    /// Override the output sample rate (default 22050 Hz).
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Override the language code (default `en`).
    pub fn language(mut self, lang: impl Into<String>) -> Self {
        self.lang = lang.into();
        self
    }

    fn url(&self) -> String {
        format!("{}/sse/speak/{}", self.base_url, self.lang)
    }
}

/// Build the Neuphonic synthesis JSON body (pure — the request seam).
fn build_payload(text: &str, voice_id: &str, sample_rate: u32, lang: &str) -> Value {
    json!({
        "text": text,
        "voice_id": voice_id,
        "lang_code": lang,
        "encoding": NEUPHONIC_ENCODING,
        "sampling_rate": sample_rate,
    })
}

/// Decode an SSE body into raw concatenated PCM bytes (pure — the response seam).
/// Each `data:` event is JSON `{ "data": { "audio": "<base64>" } }`; non-audio
/// events (status pings, etc.) are skipped.
fn audio_from_sse_body(body: &str) -> Result<Vec<u8>> {
    let mut pcm = Vec::new();
    for ev in tail::sse_data_events(body) {
        let Ok(value) = serde_json::from_str::<Value>(&ev) else {
            continue;
        };
        if let Some(b64) = value
            .get("data")
            .and_then(|d| d.get("audio"))
            .and_then(|a| a.as_str())
        {
            pcm.extend_from_slice(&tail::b64_decode(b64)?);
        }
    }
    Ok(pcm)
}

#[async_trait]
impl TtsService for NeuphonicTts {
    fn name(&self) -> &str {
        "neuphonic"
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        // Stateless HTTP/SSE — nothing to open. The synthesis POST is per-utterance.
        Ok(())
    }

    async fn run_tts(&mut self, text: &str) -> Result<Vec<Frame>> {
        self.ctx_counter += 1;
        let context_id: Arc<str> = Arc::from(format!("ctx-{}", self.ctx_counter));
        let payload = build_payload(text, &self.voice_id, self.sample_rate, &self.lang);

        let resp = self
            .http
            .post(self.url())
            .header("X-API-KEY", &self.api_key)
            .json(&payload)
            .send()
            .await
            .map_err(|e| FlowcatError::Network(format!("neuphonic send: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(FlowcatError::Network(format!(
                "neuphonic http {status}: {body}"
            )));
        }
        let body = resp
            .text()
            .await
            .map_err(|e| FlowcatError::Network(format!("neuphonic body: {e}")))?;
        let pcm = audio_from_sse_body(&body)?;
        Ok(tail::one_shot_frames(&pcm, self.sample_rate, context_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_matches_neuphonic_schema() {
        let p = build_payload("hi there", "voice-x", 24_000, "en");
        assert_eq!(p["text"], "hi there");
        assert_eq!(p["voice_id"], "voice-x");
        assert_eq!(p["encoding"], "pcm_linear");
        assert_eq!(p["sampling_rate"], 24_000);
        assert_eq!(p["lang_code"], "en");
    }

    #[test]
    fn url_uses_the_fixed_host_and_lang() {
        let t = NeuphonicTts::new("k", "v").language("es");
        assert_eq!(t.url(), "https://api.neuphonic.com/sse/speak/es");
    }

    #[test]
    fn decode_sse_body_concatenates_audio_events() {
        // base64 of [1,0] and [255,255] → two LE i16 samples 1 and -1.
        let a = "AQA="; // [1,0]
        let b = "//8="; // [255,255]
        let body = format!(
            "data: {{\"data\":{{\"audio\":\"{a}\"}}}}\n\ndata: {{\"status\":\"ok\"}}\n\ndata: {{\"data\":{{\"audio\":\"{b}\"}}}}\n\n"
        );
        let pcm = audio_from_sse_body(&body).unwrap();
        assert_eq!(tail::pcm_s16le(&pcm), vec![1, -1]);
    }

    /// Live smoke (requires `NEUPHONIC_API_KEY` + `NEUPHONIC_VOICE_ID`).
    #[tokio::test]
    #[ignore = "requires NEUPHONIC_API_KEY + NEUPHONIC_VOICE_ID"]
    async fn neuphonic_live_synthesizes_audio() {
        let key = std::env::var("NEUPHONIC_API_KEY").expect("NEUPHONIC_API_KEY");
        let voice = std::env::var("NEUPHONIC_VOICE_ID").expect("NEUPHONIC_VOICE_ID");
        let mut tts = NeuphonicTts::new(key, voice);
        tts.start(&StartParams::default()).await.expect("start");
        let frames = tts.run_tts("Hello from flowcat.").await.expect("run_tts");
        assert!(frames.iter().any(|f| matches!(f, Frame::TtsAudio { .. })));
    }
}
