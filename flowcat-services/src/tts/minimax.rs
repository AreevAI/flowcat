// SPDX-License-Identifier: Apache-2.0
//
//! **MiniMax** TTS — a **(D)istinct** HTTP-POST-audio client.
//!
//! MiniMax's T2A v2 endpoint POSTs the utterance to `{base}?GroupId=<group>`
//! (cross-checked against pipecat `services/minimax/tts.py`): `Authorization:
//! Bearer <key>` and a JSON body `{ model, text, stream: true,
//! voice_setting: { voice_id, speed, vol, pitch },
//! audio_setting: { bitrate, format: "pcm", channel, sample_rate } }`. The
//! response is **SSE** (`data:`-prefixed JSON blocks); each block carries audio at
//! `data.audio` as a **hex** string (the final block has `extra_info` and no
//! audio). The request encode ([`build_body`] / [`build_url`]) + the SSE/hex
//! decode ([`decode_sse`]) are pure, unit-tested seams over the shared [`http`]
//! helpers. Behind the `tts-minimax` feature.

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
    hex_to_bytes, pcm_from_le_bytes, tts_frames, HttpTtsBody, HttpTtsClient, HttpTtsRequest,
};

/// MiniMax's default T2A v2 endpoint (the global host). The key rides the
/// `Authorization` header; only the validated `GroupId` is a query param.
pub const MINIMAX_API_BASE: &str = "https://api.minimax.io/v1/t2a_v2";

/// MiniMax HTTP TTS service (stateless request/response over SSE).
pub struct MiniMaxTts {
    client: HttpTtsClient,
    api_key: String,
    base_url: String,
    group_id: String,
    model: String,
    voice_id: String,
    sample_rate: u32,
    ctx_counter: u64,
}

impl MiniMaxTts {
    /// Construct bound to `api_key` + `group_id` + `voice_id` (default model
    /// `speech-02-turbo`, 24 kHz PCM).
    pub fn new(
        api_key: impl Into<String>,
        group_id: impl Into<String>,
        voice_id: impl Into<String>,
    ) -> Self {
        Self {
            client: HttpTtsClient::new("minimax"),
            api_key: api_key.into(),
            base_url: MINIMAX_API_BASE.to_string(),
            group_id: group_id.into(),
            model: "speech-02-turbo".to_string(),
            voice_id: voice_id.into(),
            sample_rate: 24_000,
            ctx_counter: 0,
        }
    }

    /// Override the API base (e.g. the China or US-west host).
    pub fn base_url(mut self, base: impl Into<String>) -> Self {
        self.base_url = base.into();
        self
    }

    /// Override the model (default `speech-02-turbo`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the output sample rate (default 24 kHz).
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    fn url(&self) -> String {
        build_url(&self.base_url, &self.group_id)
    }
}

#[async_trait]
impl TtsService for MiniMaxTts {
    fn name(&self) -> &str {
        "minimax"
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        if self.api_key.is_empty() {
            return Err(FlowcatError::Session("minimax tts: empty api key".into()));
        }
        // MiniMax T2A requires the GroupId query param; an empty one yields a
        // malformed `?GroupId=` URL → an opaque MiniMax 400. Fail with a clear
        // message instead (set FLOWCAT_MINIMAX_GROUP_ID).
        if self.group_id.trim().is_empty() {
            return Err(FlowcatError::Session(
                "minimax tts: empty group_id (set FLOWCAT_MINIMAX_GROUP_ID)".into(),
            ));
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
            body: HttpTtsBody::Json(build_body(
                text,
                &self.model,
                &self.voice_id,
                self.sample_rate,
            )),
        };
        let raw = self.client.post(req).await?;
        let pcm = decode_sse(&raw);
        Ok(tts_frames(pcm, self.sample_rate, context_id))
    }
}

/// Build the `?GroupId=` URL (pure seam — the GroupId is validated config).
pub fn build_url(base_url: &str, group_id: &str) -> String {
    format!("{base_url}?GroupId={group_id}")
}

/// Build the MiniMax T2A v2 request body (pure seam).
pub fn build_body(text: &str, model: &str, voice_id: &str, sample_rate: u32) -> Value {
    json!({
        "model": model,
        "text": text,
        "stream": true,
        "voice_setting": {
            "voice_id": voice_id,
            "speed": 1.0,
            "vol": 1.0,
            "pitch": 0,
        },
        "audio_setting": {
            "bitrate": 128000,
            "format": "pcm",
            "channel": 1,
            "sample_rate": sample_rate,
        },
    })
}

/// Decode the MiniMax SSE response body into PCM samples (pure seam). Each
/// `data:` line is JSON; `data.audio` is a hex PCM string, hex-decoded and
/// concatenated in order. The final block (with `extra_info`) and any malformed /
/// audio-less block are skipped — never panics.
pub fn decode_sse(body: &[u8]) -> Vec<i16> {
    let text = String::from_utf8_lossy(body);
    let mut bytes = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        let Some(payload) = line.strip_prefix("data:") else {
            continue;
        };
        let payload = payload.trim();
        if payload.is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(payload) else {
            continue;
        };
        // The terminal block carries extra_info and no usable audio.
        if value.get("extra_info").is_some() {
            continue;
        }
        if let Some(hex) = value
            .get("data")
            .and_then(|d| d.get("audio"))
            .and_then(|a| a.as_str())
        {
            bytes.extend(hex_to_bytes(hex));
        }
    }
    pcm_from_le_bytes(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_carries_group_id() {
        assert_eq!(
            build_url(MINIMAX_API_BASE, "grp-7"),
            "https://api.minimax.io/v1/t2a_v2?GroupId=grp-7"
        );
    }

    #[test]
    fn body_matches_minimax_schema() {
        let body = build_body("hello", "speech-02-turbo", "voice-x", 24_000);
        assert_eq!(body["model"], "speech-02-turbo");
        assert_eq!(body["text"], "hello");
        assert_eq!(body["stream"], true);
        assert_eq!(body["voice_setting"]["voice_id"], "voice-x");
        assert_eq!(body["audio_setting"]["format"], "pcm");
        assert_eq!(body["audio_setting"]["sample_rate"], 24_000);
    }

    #[test]
    fn decode_sse_concatenates_hex_audio() {
        // Two audio blocks: hex "0100" (→ 1) then "ffff" (→ -1), plus a final
        // extra_info block (skipped).
        let sse = format!(
            "data: {}\n\ndata: {}\n\ndata: {}\n",
            json!({ "data": { "audio": "0100" } }),
            json!({ "data": { "audio": "ffff" } }),
            json!({ "data": { "audio": "dead" }, "extra_info": { "audio_length": 1 } }),
        );
        assert_eq!(decode_sse(sse.as_bytes()), vec![1, -1]);
    }

    #[test]
    fn decode_sse_tolerates_garbage() {
        assert!(decode_sse(b"no data here").is_empty());
        assert!(decode_sse(b"data: {bad}\n").is_empty());
    }

    #[test]
    fn client_defaults() {
        let tts = MiniMaxTts::new("k", "grp", "voice-x");
        assert_eq!(tts.name(), "minimax");
        assert_eq!(tts.sample_rate(), 24_000);
        assert!(tts.url().contains("GroupId=grp"));
    }

    #[tokio::test]
    async fn start_rejects_empty_key() {
        let mut tts = MiniMaxTts::new("", "grp", "voice-x");
        assert!(tts.start(&StartParams::default()).await.is_err());
    }

    #[tokio::test]
    async fn start_rejects_empty_group_id() {
        // An empty group_id would build a malformed `?GroupId=` URL → opaque 400;
        // start() must reject it with a clear message instead.
        let mut tts = MiniMaxTts::new("k", "", "voice-x");
        let err = tts.start(&StartParams::default()).await.unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("group_id"),
            "got: {err}"
        );
    }

    /// Live smoke (requires `MINIMAX_API_KEY` + `MINIMAX_GROUP_ID` +
    /// `MINIMAX_VOICE_ID`). Run:
    /// `MINIMAX_API_KEY=… MINIMAX_GROUP_ID=… MINIMAX_VOICE_ID=… cargo test -p flowcat-services --features tts-minimax -- --ignored minimax_tts_live`
    #[tokio::test]
    #[ignore = "requires MINIMAX_API_KEY + MINIMAX_GROUP_ID + MINIMAX_VOICE_ID"]
    async fn minimax_tts_live_synthesizes_audio() {
        let key = std::env::var("MINIMAX_API_KEY").expect("MINIMAX_API_KEY");
        let group = std::env::var("MINIMAX_GROUP_ID").expect("MINIMAX_GROUP_ID");
        let voice = std::env::var("MINIMAX_VOICE_ID").expect("MINIMAX_VOICE_ID");
        let mut tts = MiniMaxTts::new(key, group, voice);
        tts.start(&StartParams::default()).await.expect("start");
        let frames = tts.run_tts("Hello from flowcat.").await.expect("run_tts");
        assert!(frames.iter().any(|f| matches!(f, Frame::TtsAudio { .. })));
    }
}
