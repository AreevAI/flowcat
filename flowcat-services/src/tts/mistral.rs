// SPDX-License-Identifier: Apache-2.0
//
//! **Mistral** TTS — a **(D)istinct** HTTP-POST-audio client.
//!
//! Mistral's Voxtral TTS POSTs the utterance to `{base}/v1/audio/speech`
//! (cross-checked against pipecat `services/mistral/tts.py`, which drives the same
//! endpoint via the Mistral SDK): `Authorization: Bearer <key>` and a JSON body
//! `{ input, model, voice_id, response_format: "pcm", stream: true }`. The
//! response is **SSE**: `data:`-prefixed JSON lines carrying
//! `speech.audio.delta` events whose `data.audio_data` is base64 **float32** PCM
//! at 24 kHz (terminated by a `speech.audio.done` event). The request encode
//! ([`build_body`]) + the SSE/float32 decode ([`decode_sse`]) are pure,
//! unit-tested seams over the shared [`http`] helpers. Behind the `tts-mistral`
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
    base64_decode, float32_le_to_i16, tts_frames, HttpTtsBody, HttpTtsClient, HttpTtsRequest,
};

/// Mistral's default API base. The key rides the `Authorization` header.
pub const MISTRAL_API_BASE: &str = "https://api.mistral.ai";
/// Mistral TTS streams float32 PCM at a fixed 24 kHz.
const MISTRAL_SAMPLE_RATE: u32 = 24_000;

/// Mistral HTTP TTS service (stateless request/response over SSE).
pub struct MistralTts {
    client: HttpTtsClient,
    api_key: String,
    base_url: String,
    model: String,
    voice: String,
    ctx_counter: u64,
}

impl MistralTts {
    /// Construct bound to `api_key` + `voice` (default model
    /// `voxtral-mini-tts-2603`, 24 kHz).
    pub fn new(api_key: impl Into<String>, voice: impl Into<String>) -> Self {
        Self {
            client: HttpTtsClient::new("mistral"),
            api_key: api_key.into(),
            base_url: MISTRAL_API_BASE.to_string(),
            model: "voxtral-mini-tts-2603".to_string(),
            voice: voice.into(),
            ctx_counter: 0,
        }
    }

    /// Override the model (default `voxtral-mini-tts-2603`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    fn url(&self) -> String {
        format!("{}/v1/audio/speech", self.base_url)
    }
}

#[async_trait]
impl TtsService for MistralTts {
    fn name(&self) -> &str {
        "mistral"
    }

    fn sample_rate(&self) -> u32 {
        MISTRAL_SAMPLE_RATE
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        if self.api_key.is_empty() {
            return Err(FlowcatError::Session("mistral tts: empty api key".into()));
        }
        Ok(())
    }

    async fn run_tts(&mut self, text: &str) -> Result<Vec<Frame>> {
        self.ctx_counter += 1;
        let context_id: Arc<str> = Arc::from(format!("ctx-{}", self.ctx_counter));
        let req = HttpTtsRequest {
            url: self.url(),
            headers: vec![(
                "Authorization".to_string(),
                format!("Bearer {}", self.api_key),
            )],
            body: HttpTtsBody::Json(build_body(text, &self.model, &self.voice)),
        };
        let raw = self.client.post(req).await?;
        let pcm = decode_sse(&raw);
        Ok(tts_frames(pcm, MISTRAL_SAMPLE_RATE, context_id))
    }
}

/// Build the Mistral `/v1/audio/speech` request body (pure seam).
pub fn build_body(text: &str, model: &str, voice: &str) -> Value {
    json!({
        "input": text,
        "model": model,
        "voice_id": voice,
        "response_format": "pcm",
        "stream": true,
    })
}

/// Decode the Mistral SSE response body into PCM samples (pure seam). Each
/// `data:` line is JSON; a `speech.audio.delta` event's `data.audio_data` is
/// base64 float32 PCM, concatenated in order and converted to i16. Non-`data`
/// lines, the `[DONE]` sentinel, malformed JSON, or other event types are skipped
/// — never panics.
pub fn decode_sse(body: &[u8]) -> Vec<i16> {
    let text = String::from_utf8_lossy(body);
    let mut pcm = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        let Some(payload) = line.strip_prefix("data:") else {
            continue;
        };
        let payload = payload.trim();
        if payload.is_empty() || payload == "[DONE]" {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(payload) else {
            continue;
        };
        if value.get("event").and_then(|e| e.as_str()) != Some("speech.audio.delta") {
            continue;
        }
        if let Some(b64) = value
            .get("data")
            .and_then(|d| d.get("audio_data"))
            .and_then(|a| a.as_str())
        {
            let f32_bytes = base64_decode(b64);
            pcm.extend(float32_le_to_i16(&f32_bytes));
        }
    }
    pcm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_matches_mistral_schema() {
        let body = build_body("hello", "voxtral-mini-tts-2603", "nova");
        assert_eq!(body["input"], "hello");
        assert_eq!(body["model"], "voxtral-mini-tts-2603");
        assert_eq!(body["voice_id"], "nova");
        assert_eq!(body["response_format"], "pcm");
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn decode_sse_collects_float32_deltas() {
        // Two deltas: [1.0] then [-1.0, 0.0] (float32 LE → base64).
        let d1 = http::base64_encode(&1.0f32.to_le_bytes());
        let mut d2_bytes = Vec::new();
        d2_bytes.extend_from_slice(&(-1.0f32).to_le_bytes());
        d2_bytes.extend_from_slice(&0.0f32.to_le_bytes());
        let d2 = http::base64_encode(&d2_bytes);
        let sse = format!(
            "data: {}\n\ndata: {}\n\ndata: {}\n\ndata: [DONE]\n",
            json!({ "event": "speech.audio.delta", "data": { "audio_data": d1 } }),
            json!({ "event": "speech.audio.delta", "data": { "audio_data": d2 } }),
            json!({ "event": "speech.audio.done" }),
        );
        assert_eq!(decode_sse(sse.as_bytes()), vec![32767, -32767, 0]);
    }

    #[test]
    fn decode_sse_tolerates_garbage() {
        assert!(decode_sse(b"not sse at all").is_empty());
        assert!(decode_sse(b"data: {bad json}\n").is_empty());
    }

    #[test]
    fn client_defaults() {
        let tts = MistralTts::new("k", "nova");
        assert_eq!(tts.name(), "mistral");
        assert_eq!(tts.sample_rate(), 24_000);
        assert_eq!(tts.url(), "https://api.mistral.ai/v1/audio/speech");
    }

    #[tokio::test]
    async fn start_rejects_empty_key() {
        let mut tts = MistralTts::new("", "nova");
        assert!(tts.start(&StartParams::default()).await.is_err());
    }

    /// Live smoke (requires `MISTRAL_API_KEY`). Run:
    /// `MISTRAL_API_KEY=… cargo test -p flowcat-services --features tts-mistral -- --ignored mistral_tts_live`
    #[tokio::test]
    #[ignore = "requires MISTRAL_API_KEY"]
    async fn mistral_tts_live_synthesizes_audio() {
        let key = std::env::var("MISTRAL_API_KEY").expect("MISTRAL_API_KEY");
        let voice = std::env::var("MISTRAL_VOICE").unwrap_or_else(|_| "nova".into());
        let mut tts = MistralTts::new(key, voice);
        tts.start(&StartParams::default()).await.expect("start");
        let frames = tts.run_tts("Hello from flowcat.").await.expect("run_tts");
        assert!(frames.iter().any(|f| matches!(f, Frame::TtsAudio { .. })));
    }
}
