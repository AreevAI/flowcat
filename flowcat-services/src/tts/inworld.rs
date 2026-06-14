// SPDX-License-Identifier: Apache-2.0
//
//! **Inworld** TTS — a **(D)istinct** HTTP-POST-audio client.
//!
//! Inworld's HTTP TTS streams JSON chunks from
//! `{base}/tts/v1/voice:stream` (cross-checked against pipecat
//! `services/inworld/tts.py`): an `Authorization: Basic <key>` header (Inworld
//! issues a base64 key used verbatim) and a JSON body `{ text, voiceId, modelId,
//! audioConfig: { audioEncoding: "PCM", sampleRateHertz } }`. The response is
//! **JSONL** — one JSON object per line, each carrying base64 audio at
//! `result.audioContent`; that audio may itself be a WAV file, whose 44-byte
//! header is stripped to raw PCM. The request encode ([`build_body`]) + the
//! JSONL/base64 decode ([`decode_jsonl`]) are pure, unit-tested seams over the
//! shared [`http`] helpers. Behind the `tts-inworld` feature.

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
    base64_decode, pcm_from_le_bytes, strip_wav_header, tts_frames, HttpTtsBody, HttpTtsClient,
    HttpTtsRequest,
};

/// Inworld's default API base. The key rides the `Authorization: Basic` header.
pub const INWORLD_API_BASE: &str = "https://api.inworld.ai";

/// Inworld HTTP TTS service (stateless request/response over JSONL).
pub struct InworldTts {
    client: HttpTtsClient,
    api_key: String,
    base_url: String,
    voice_id: String,
    model: String,
    sample_rate: u32,
    ctx_counter: u64,
}

impl InworldTts {
    /// Construct bound to `api_key` + `voice_id` (default model `inworld-tts-2`,
    /// 24 kHz PCM).
    pub fn new(api_key: impl Into<String>, voice_id: impl Into<String>) -> Self {
        Self {
            client: HttpTtsClient::new("inworld"),
            api_key: api_key.into(),
            base_url: INWORLD_API_BASE.to_string(),
            voice_id: voice_id.into(),
            model: "inworld-tts-2".to_string(),
            sample_rate: 24_000,
            ctx_counter: 0,
        }
    }

    /// Override the model (default `inworld-tts-2`).
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
        format!("{}/tts/v1/voice:stream", self.base_url)
    }
}

#[async_trait]
impl TtsService for InworldTts {
    fn name(&self) -> &str {
        "inworld"
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        if self.api_key.is_empty() {
            return Err(FlowcatError::Session("inworld tts: empty api key".into()));
        }
        Ok(())
    }

    async fn run_tts(&mut self, text: &str) -> Result<Vec<Frame>> {
        self.ctx_counter += 1;
        let context_id: Arc<str> = Arc::from(format!("ctx-{}", self.ctx_counter));
        // Inworld caps one request at 2000 chars (400 otherwise). Split a long reply
        // into ≤MAX-char chunks at word boundaries, synthesize each, and concatenate
        // the PCM into one framed utterance. Short replies → a single chunk (unchanged).
        let mut pcm: Vec<i16> = Vec::new();
        for chunk in split_text(text, INWORLD_MAX_CHARS) {
            if chunk.trim().is_empty() {
                continue;
            }
            let req = HttpTtsRequest {
                url: self.url(),
                headers: vec![(
                    "Authorization".to_string(),
                    format!("Basic {}", self.api_key),
                )],
                body: HttpTtsBody::Json(build_body(
                    &chunk,
                    &self.voice_id,
                    &self.model,
                    self.sample_rate,
                )),
            };
            let raw = self.client.post(req).await?;
            pcm.extend(decode_jsonl(&raw));
        }
        Ok(tts_frames(pcm, self.sample_rate, context_id))
    }
}

/// Inworld rejects a `text` longer than 2000 characters (400). Keep a small margin.
const INWORLD_MAX_CHARS: usize = 1900;

/// Split `text` into chunks of at most `max` characters, breaking at word
/// boundaries (never mid-word, unless a single word itself exceeds `max`, which is
/// then hard-split). Pure + dependency-free; whitespace is normalised to single
/// spaces (inaudible in TTS). `text` ≤ `max` returns a single chunk unchanged.
pub fn split_text(text: &str, max: usize) -> Vec<String> {
    if text.chars().count() <= max {
        return vec![text.to_string()];
    }
    let mut chunks = Vec::new();
    let mut cur = String::new();
    let push_cur = |cur: &mut String, chunks: &mut Vec<String>| {
        if !cur.is_empty() {
            chunks.push(std::mem::take(cur));
        }
    };
    for word in text.split_whitespace() {
        let wlen = word.chars().count();
        if wlen > max {
            // A single oversized word: flush, then hard-split it by chars.
            push_cur(&mut cur, &mut chunks);
            let mut buf = String::new();
            for ch in word.chars() {
                if buf.chars().count() == max {
                    chunks.push(std::mem::take(&mut buf));
                }
                buf.push(ch);
            }
            cur = buf;
            continue;
        }
        let extra = wlen + if cur.is_empty() { 0 } else { 1 };
        if !cur.is_empty() && cur.chars().count() + extra > max {
            push_cur(&mut cur, &mut chunks);
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(word);
    }
    push_cur(&mut cur, &mut chunks);
    chunks
}

/// Build the Inworld `voice:stream` request body (pure seam).
pub fn build_body(text: &str, voice_id: &str, model: &str, sample_rate: u32) -> Value {
    json!({
        "text": text,
        "voiceId": voice_id,
        "modelId": model,
        "audioConfig": {
            "audioEncoding": "PCM",
            "sampleRateHertz": sample_rate,
        },
    })
}

/// Decode the Inworld JSONL response body into PCM samples (pure seam). Each line
/// is a JSON object; a string `result.audioContent` is base64 audio (possibly
/// WAV-wrapped — the header is stripped), concatenated in order. Blank lines,
/// malformed JSON, or lines without audio content are skipped — never panics.
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
        if let Some(b64) = value
            .get("result")
            .and_then(|r| r.get("audioContent"))
            .and_then(|a| a.as_str())
        {
            let audio = base64_decode(b64);
            pcm.extend(pcm_from_le_bytes(strip_wav_header(&audio)));
        }
    }
    pcm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_text_chunks_long_text_at_word_boundaries() {
        assert_eq!(split_text("hi there", 1900), vec!["hi there".to_string()]);
        let long = "word ".repeat(1000); // ~5000 chars
        let chunks = split_text(long.trim(), 1900);
        assert!(chunks.len() > 1, "should split");
        for c in &chunks {
            assert!(
                c.chars().count() <= 1900,
                "chunk over limit: {}",
                c.chars().count()
            );
            assert!(!c.starts_with(' ') && !c.ends_with(' '));
        }
        assert_eq!(chunks.join(" ").split_whitespace().count(), 1000);
    }

    #[test]
    fn split_text_hard_splits_an_oversized_word() {
        let huge = "a".repeat(2500);
        let chunks = split_text(&huge, 1900);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].chars().count() <= 1900);
    }

    #[test]
    fn body_matches_inworld_schema() {
        let body = build_body("hello", "Ashley", "inworld-tts-2", 24_000);
        assert_eq!(body["text"], "hello");
        assert_eq!(body["voiceId"], "Ashley");
        assert_eq!(body["modelId"], "inworld-tts-2");
        assert_eq!(body["audioConfig"]["audioEncoding"], "PCM");
        assert_eq!(body["audioConfig"]["sampleRateHertz"], 24_000);
    }

    #[test]
    fn decode_jsonl_collects_result_audio_content() {
        // Raw PCM line then a WAV-wrapped line.
        let raw = http::base64_encode(&[1u8, 0, 255, 255]); // 1, -1
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&[0u8; 4]);
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(&[0u8; 32]);
        wav.extend_from_slice(&[2, 0]); // 2
        let wav_b64 = http::base64_encode(&wav);
        let jsonl = format!(
            "{}\n{}\n",
            json!({ "result": { "audioContent": raw } }),
            json!({ "result": { "audioContent": wav_b64 } }),
        );
        assert_eq!(decode_jsonl(jsonl.as_bytes()), vec![1, -1, 2]);
    }

    #[test]
    fn decode_jsonl_tolerates_garbage() {
        assert!(decode_jsonl(b"\n").is_empty());
        assert!(decode_jsonl(b"{nope}\n").is_empty());
        let no_audio = serde_json::to_vec(&json!({ "result": {} })).unwrap();
        assert!(decode_jsonl(&no_audio).is_empty());
    }

    #[test]
    fn client_defaults() {
        let tts = InworldTts::new("k", "Ashley");
        assert_eq!(tts.name(), "inworld");
        assert_eq!(tts.sample_rate(), 24_000);
        assert_eq!(tts.url(), "https://api.inworld.ai/tts/v1/voice:stream");
    }

    #[tokio::test]
    async fn start_rejects_empty_key() {
        let mut tts = InworldTts::new("", "Ashley");
        assert!(tts.start(&StartParams::default()).await.is_err());
    }

    /// Live smoke (requires `INWORLD_API_KEY`). Run:
    /// `INWORLD_API_KEY=… cargo test -p flowcat-services --features tts-inworld -- --ignored inworld_tts_live`
    #[tokio::test]
    #[ignore = "requires INWORLD_API_KEY"]
    async fn inworld_tts_live_synthesizes_audio() {
        let key = std::env::var("INWORLD_API_KEY").expect("INWORLD_API_KEY");
        let voice = std::env::var("INWORLD_VOICE").unwrap_or_else(|_| "Ashley".into());
        let mut tts = InworldTts::new(key, voice);
        tts.start(&StartParams::default()).await.expect("start");
        let frames = tts.run_tts("Hello from flowcat.").await.expect("run_tts");
        assert!(frames.iter().any(|f| matches!(f, Frame::TtsAudio { .. })));
    }
}
