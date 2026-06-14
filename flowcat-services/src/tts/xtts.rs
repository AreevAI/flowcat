// SPDX-License-Identifier: Apache-2.0
//
//! **XTTS** TTS — local-model HTTP-server client (Group H).
//!
//! Coqui XTTS runs as a local streaming server (Docker:
//! `ghcr.io/coqui-ai/xtts-streaming-server`). pipecat's `XTTSService` (a) `GET`s the
//! studio-speaker embeddings, then (b) `POST /tts_stream` with the chosen speaker's
//! `speaker_embedding` + `gpt_cond_latent`, receiving raw 24 kHz s16 PCM. The
//! `tts-xtts` feature enables only `reqwest`+`tokio`, which is exactly this HTTP
//! client; a clear *"local model server not wired"* seam (no panic) covers the case
//! where no server URL is configured.
//!
//! [`build_stream_payload`] is the **pure** request seam.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use flowcat_core::error::{FlowcatError, Result};
use flowcat_core::processor::frame::{Frame, StartParams};
use flowcat_core::service::TtsService;

#[path = "tail_tts_common.rs"]
#[allow(clippy::duplicate_mod)] // shared header pattern: included into each Group-H module
mod tail;

/// XTTS server output sample rate (fixed at 24 kHz).
pub const XTTS_SAMPLE_RATE: u32 = 24_000;

/// XTTS TTS service (local streaming server).
pub struct XttsTts {
    voice_id: String,
    lang: String,
    base_url: Option<String>,
    /// Speaker embeddings fetched in [`TtsService::start`]; keyed by speaker name.
    studio_speakers: Option<HashMap<String, Value>>,
    http: reqwest::Client,
    ctx_counter: u64,
}

impl XttsTts {
    /// Construct bound to `voice_id` (the studio-speaker name). XTTS takes no API
    /// key; `_api_key` is accepted for a uniform constructor and ignored.
    pub fn new(_api_key: impl Into<String>, voice_id: impl Into<String>) -> Self {
        Self {
            voice_id: voice_id.into(),
            lang: "en".to_string(),
            base_url: None,
            studio_speakers: None,
            http: reqwest::Client::new(),
            ctx_counter: 0,
        }
    }

    /// Point the client at a running XTTS streaming server (e.g.
    /// `http://localhost:8000`). Required.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        let mut url = base_url.into();
        while url.ends_with('/') {
            url.pop();
        }
        self.base_url = Some(url);
        self
    }

    /// Override the language code (default `en`).
    pub fn language(mut self, lang: impl Into<String>) -> Self {
        self.lang = lang.into();
        self
    }
}

/// Build the XTTS `/tts_stream` request body from the resolved speaker embedding
/// (pure — the request seam). `add_wav_header:false` → headerless PCM.
fn build_stream_payload(text: &str, lang: &str, embedding: &Value) -> Value {
    json!({
        "text": text,
        "language": lang,
        "speaker_embedding": embedding.get("speaker_embedding").cloned().unwrap_or(Value::Null),
        "gpt_cond_latent": embedding.get("gpt_cond_latent").cloned().unwrap_or(Value::Null),
        "add_wav_header": false,
        "stream_chunk_size": 20,
    })
}

#[async_trait]
impl TtsService for XttsTts {
    fn name(&self) -> &str {
        "xtts"
    }

    fn sample_rate(&self) -> u32 {
        XTTS_SAMPLE_RATE
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        let Some(base) = self.base_url.clone() else {
            return Err(FlowcatError::Other(
                "xtts TTS: local model not wired — set the XTTS streaming-server URL \
                 via XttsTts::with_base_url"
                    .into(),
            ));
        };
        let resp = self
            .http
            .get(format!("{base}/studio_speakers"))
            .send()
            .await
            .map_err(|e| FlowcatError::Network(format!("xtts studio_speakers: {e}")))?;
        if !resp.status().is_success() {
            return Err(FlowcatError::Network(format!(
                "xtts studio_speakers http {}",
                resp.status()
            )));
        }
        let speakers: HashMap<String, Value> = resp
            .json()
            .await
            .map_err(|e| FlowcatError::Protocol(format!("xtts studio_speakers json: {e}")))?;
        self.studio_speakers = Some(speakers);
        Ok(())
    }

    async fn run_tts(&mut self, text: &str) -> Result<Vec<Frame>> {
        let Some(base) = self.base_url.clone() else {
            return Err(FlowcatError::Other(
                "xtts TTS: local model not wired — set XttsTts::with_base_url".into(),
            ));
        };
        let speakers = self.studio_speakers.as_ref().ok_or_else(|| {
            FlowcatError::Other("xtts TTS: run_tts before start (no studio speakers)".into())
        })?;
        let embedding = speakers.get(&self.voice_id).ok_or_else(|| {
            FlowcatError::Other(format!(
                "xtts TTS: unknown studio speaker '{}'",
                self.voice_id
            ))
        })?;
        let payload = build_stream_payload(text, &self.lang, embedding);

        self.ctx_counter += 1;
        let context_id: Arc<str> = Arc::from(format!("ctx-{}", self.ctx_counter));
        let resp = self
            .http
            .post(format!("{base}/tts_stream"))
            .json(&payload)
            .send()
            .await
            .map_err(|e| FlowcatError::Network(format!("xtts tts_stream: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(FlowcatError::Network(format!("xtts http {status}: {body}")));
        }
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| FlowcatError::Network(format!("xtts body: {e}")))?;
        // add_wav_header:false → headerless PCM, but strip defensively.
        let pcm = tail::strip_wav_header(&bytes);
        Ok(tail::one_shot_frames(pcm, XTTS_SAMPLE_RATE, context_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_payload_carries_embedding_and_flags() {
        let emb = json!({ "speaker_embedding": [0.1, 0.2], "gpt_cond_latent": [[1.0]] });
        let p = build_stream_payload("hi", "en", &emb);
        assert_eq!(p["text"], "hi");
        assert_eq!(p["language"], "en");
        assert_eq!(p["speaker_embedding"], json!([0.1, 0.2]));
        assert_eq!(p["gpt_cond_latent"], json!([[1.0]]));
        assert_eq!(p["add_wav_header"], false);
        assert_eq!(p["stream_chunk_size"], 20);
    }

    #[tokio::test]
    async fn without_base_url_start_returns_not_wired_seam() {
        let mut tts = XttsTts::new("", "Claribel Dervla");
        let err = tts.start(&StartParams::default()).await.unwrap_err();
        assert!(err.to_string().contains("local model not wired"));
    }

    #[tokio::test]
    async fn run_before_start_errors_cleanly() {
        let mut tts = XttsTts::new("", "spk").with_base_url("http://localhost:8000");
        let err = tts.run_tts("hi").await.unwrap_err();
        assert!(err.to_string().contains("run_tts before start"));
    }

    /// Live smoke (requires a running XTTS server at `XTTS_BASE_URL`).
    #[tokio::test]
    #[ignore = "requires XTTS_BASE_URL (xtts-streaming-server) + XTTS_SPEAKER"]
    async fn xtts_live_synthesizes_audio() {
        let base = std::env::var("XTTS_BASE_URL").expect("XTTS_BASE_URL");
        let spk = std::env::var("XTTS_SPEAKER").expect("XTTS_SPEAKER");
        let mut tts = XttsTts::new("", spk).with_base_url(base);
        tts.start(&StartParams::default()).await.expect("start");
        let frames = tts.run_tts("Hello from flowcat.").await.expect("run_tts");
        assert!(frames.iter().any(|f| matches!(f, Frame::TtsAudio { .. })));
    }
}
