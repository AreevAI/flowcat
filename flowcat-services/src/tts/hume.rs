// SPDX-License-Identifier: Apache-2.0
//
//! **Hume** TTS — a **(D)istinct** HTTP-POST-audio client.
//!
//! Hume's Octave TTS streams JSON chunks from `{base}/v0/tts/stream/json`
//! (cross-checked against pipecat `services/hume/tts.py`, which drives the same
//! endpoint via the Hume SDK): an `X-Hume-Api-Key: <key>` header and a JSON body
//! `{ utterances: [{ text, voice: { id } }], format: { type: "pcm" },
//! instant_mode: true, version }`. The response is **JSONL** — one JSON object per
//! line, each with a base64 PCM `audio` field — at 48 kHz mono. The request encode
//! ([`build_body`]) + the JSONL/base64 decode ([`decode_jsonl`]) are pure,
//! unit-tested seams over the shared [`http`] helpers. Behind the `tts-hume`
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
    base64_decode, pcm_from_le_bytes, tts_frames, HttpTtsBody, HttpTtsClient, HttpTtsRequest,
};

/// Hume's default API base. The key rides the `X-Hume-Api-Key` header.
pub const HUME_API_BASE: &str = "https://api.hume.ai";
/// Hume TTS streams 48 kHz PCM.
const HUME_SAMPLE_RATE: u32 = 48_000;

/// Hume HTTP TTS service (stateless request/response over JSONL).
pub struct HumeTts {
    client: HttpTtsClient,
    api_key: String,
    base_url: String,
    voice_id: String,
    ctx_counter: u64,
}

impl HumeTts {
    /// Construct bound to `api_key` + `voice_id` (48 kHz PCM).
    pub fn new(api_key: impl Into<String>, voice_id: impl Into<String>) -> Self {
        Self {
            client: HttpTtsClient::new("hume"),
            api_key: api_key.into(),
            base_url: HUME_API_BASE.to_string(),
            voice_id: voice_id.into(),
            ctx_counter: 0,
        }
    }

    fn url(&self) -> String {
        format!("{}/v0/tts/stream/json", self.base_url)
    }
}

#[async_trait]
impl TtsService for HumeTts {
    fn name(&self) -> &str {
        "hume"
    }

    fn sample_rate(&self) -> u32 {
        HUME_SAMPLE_RATE
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        if self.api_key.is_empty() {
            return Err(FlowcatError::Session("hume tts: empty api key".into()));
        }
        Ok(())
    }

    async fn run_tts(&mut self, text: &str) -> Result<Vec<Frame>> {
        self.ctx_counter += 1;
        let context_id: Arc<str> = Arc::from(format!("ctx-{}", self.ctx_counter));
        let req = HttpTtsRequest {
            url: self.url(),
            headers: vec![("X-Hume-Api-Key".to_string(), self.api_key.clone())],
            body: HttpTtsBody::Json(build_body(text, &self.voice_id)),
        };
        let raw = self.client.post(req).await?;
        let pcm = decode_jsonl(&raw);
        Ok(tts_frames(pcm, HUME_SAMPLE_RATE, context_id))
    }
}

/// Build the Hume `/v0/tts/stream/json` request body (pure seam). Asks for raw
/// PCM in instant mode (version 2 — the no-description default).
pub fn build_body(text: &str, voice_id: &str) -> Value {
    json!({
        "utterances": [{ "text": text, "voice": { "id": voice_id } }],
        "format": { "type": "pcm" },
        "instant_mode": true,
        "version": "2",
    })
}

/// Decode the Hume JSONL response body into PCM samples (pure seam). Each line is
/// a JSON object; a string `audio` field is base64 PCM, concatenated in order.
/// Blank lines, malformed JSON, or objects without `audio` (timestamp messages)
/// are skipped — never panics.
pub fn decode_jsonl(body: &[u8]) -> Vec<i16> {
    let text = String::from_utf8_lossy(body);
    let mut pcm = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if let Some(b64) = value.get("audio").and_then(|a| a.as_str()) {
            pcm.extend(pcm_from_le_bytes(&base64_decode(b64)));
        }
    }
    pcm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_matches_hume_schema() {
        let body = build_body("hello there", "voice-x");
        assert_eq!(body["utterances"][0]["text"], "hello there");
        assert_eq!(body["utterances"][0]["voice"]["id"], "voice-x");
        assert_eq!(body["format"]["type"], "pcm");
        assert_eq!(body["instant_mode"], true);
    }

    #[test]
    fn decode_jsonl_collects_base64_audio_lines() {
        // Two audio lines: [1, -1] then [2] (LE i16 → base64), plus a timestamp.
        let a1 = http::base64_encode(&[1u8, 0, 255, 255]); // 1, -1
        let a2 = http::base64_encode(&[2u8, 0]); // 2
        let jsonl = format!(
            "{}\n{}\n{}\n",
            json!({ "audio": a1 }),
            json!({ "timestamp": { "type": "word" } }),
            json!({ "audio": a2 }),
        );
        assert_eq!(decode_jsonl(jsonl.as_bytes()), vec![1, -1, 2]);
    }

    #[test]
    fn decode_jsonl_tolerates_garbage() {
        assert!(decode_jsonl(b"\n\n").is_empty());
        assert!(decode_jsonl(b"{not json}\n").is_empty());
    }

    #[test]
    fn client_defaults() {
        let tts = HumeTts::new("k", "voice-x");
        assert_eq!(tts.name(), "hume");
        assert_eq!(tts.sample_rate(), 48_000);
        assert_eq!(tts.url(), "https://api.hume.ai/v0/tts/stream/json");
    }

    #[tokio::test]
    async fn start_rejects_empty_key() {
        let mut tts = HumeTts::new("", "voice-x");
        assert!(tts.start(&StartParams::default()).await.is_err());
    }

    /// Live smoke (requires `HUME_API_KEY` + `HUME_VOICE_ID`). Run:
    /// `HUME_API_KEY=… HUME_VOICE_ID=… cargo test -p flowcat-services --features tts-hume -- --ignored hume_tts_live`
    #[tokio::test]
    #[ignore = "requires HUME_API_KEY + HUME_VOICE_ID"]
    async fn hume_tts_live_synthesizes_audio() {
        let key = std::env::var("HUME_API_KEY").expect("HUME_API_KEY");
        let voice = std::env::var("HUME_VOICE_ID").expect("HUME_VOICE_ID");
        let mut tts = HumeTts::new(key, voice);
        tts.start(&StartParams::default()).await.expect("start");
        let frames = tts.run_tts("Hello from flowcat.").await.expect("run_tts");
        assert!(frames.iter().any(|f| matches!(f, Frame::TtsAudio { .. })));
    }
}
