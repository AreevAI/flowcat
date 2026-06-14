// SPDX-License-Identifier: Apache-2.0
//
//! **Resemble AI** streaming TTS (`/stream` WebSocket).
//!
//! A **(D)istinct** streaming-WebSocket client (cross-checked against pipecat
//! `services/resembleai/tts.py`). Connects to the fixed
//! `wss://websocket.cluster.resemble.ai/stream` with the API key in the
//! `Authorization: Bearer <key>` header, then sends one request per utterance:
//!
//! ```json
//! { "voice_uuid": "<voice>", "data": "hello there", "binary_response": false,
//!   "request_id": 0, "output_format": "wav", "sample_rate": 24000,
//!   "precision": "PCM_16", "no_audio_header": true }
//! ```
//!
//! `binary_response: false` + `no_audio_header: true` makes the server stream
//! base64 raw 16-bit PCM in JSON frames, keyed by the echoed `request_id`:
//!
//! ```json
//! { "type": "audio", "audio_content": "<base64 pcm>", "request_id": 0,
//!   "audio_timestamps": { "graph_chars": ["b","o"], "graph_times": [[0.0,0.1],[0.1,0.2]] } }
//! { "type": "audio_end", "request_id": 0 }
//! { "type": "error", "error_name": "InvalidVoice", "message": "…", "status_code": 400 }
//! ```
//!
//! `audio` base64 → [`Frame::TtsAudio`]; `audio_timestamps` graphemes → word
//! timings → [`Frame::TtsText`] (split on spaces, each word stamped at its first
//! grapheme's start); `audio_end` ends the run; `error` surfaces the message.
//! Since [`TtsService::run_tts`] is one utterance, the inline read needs no
//! `request_id` demux. All decode is pure.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use flowcat_core::error::Result;
use flowcat_core::processor::frame::{Frame, StartParams};
use flowcat_core::service::TtsService;

#[allow(clippy::duplicate_mod)] // each WS provider owns its own copy (feature-independent)
#[path = "ws_tts_common.rs"]
pub mod ws_tts;

use ws_tts::{Decoded, OutMsg, WsTtsConfig, WsTtsSession};

/// Resemble AI's TTS WebSocket host (the host is fixed).
pub const RESEMBLE_WSS: &str = "wss://websocket.cluster.resemble.ai/stream";

/// Resemble AI streaming-TTS session.
pub struct ResembleTts {
    api_key: String,
    voice_id: String,
    sample_rate: u32,
    session: Option<WsTtsSession>,
    request_id: u64,
}

impl ResembleTts {
    /// Construct bound to `api_key` + `voice_id` (the voice UUID) — default
    /// 22.05 kHz raw PCM_16 output (Resemble's default).
    pub fn new(api_key: impl Into<String>, voice_id: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            voice_id: voice_id.into(),
            sample_rate: 22_050,
            session: None,
            request_id: 0,
        }
    }

    /// Override the output sample rate (default 22.05 kHz; Resemble accepts
    /// 8000/16000/22050/32000/44100).
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }
}

#[async_trait]
impl TtsService for ResembleTts {
    fn name(&self) -> &str {
        "resemble"
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        let cfg = WsTtsConfig {
            url: RESEMBLE_WSS.to_string(),
            headers: vec![(
                "Authorization".to_string(),
                format!("Bearer {}", self.api_key),
            )],
            init_message: None,
            decode: decode_message,
        };
        self.session = Some(WsTtsSession::connect(cfg).await?);
        Ok(())
    }

    async fn run_tts(&mut self, text: &str) -> Result<Vec<Frame>> {
        let request_id = self.request_id;
        self.request_id += 1;
        let context_id: Arc<str> = Arc::from(format!("ctx-{request_id}"));
        let rate = self.sample_rate;
        let msg = build_request(text, request_id, &self.voice_id, rate);
        let session = ws_tts::require(&mut self.session, "resemble")?;
        session
            .synthesize(vec![OutMsg::Text(msg.to_string())], context_id, rate)
            .await
    }
}

/// Build the Resemble synthesis request for one utterance (pure — the wire-fixture
/// seam). `request_id` is a numeric id the server echoes; `no_audio_header:true`
/// gives raw PCM so the bytes are directly playable.
fn build_request(text: &str, request_id: u64, voice_uuid: &str, sample_rate: u32) -> Value {
    json!({
        "voice_uuid": voice_uuid,
        "data": text,
        "binary_response": false,
        "request_id": request_id,
        "output_format": "wav",
        "sample_rate": sample_rate,
        "precision": "PCM_16",
        "no_audio_header": true,
    })
}

/// Fold Resemble grapheme timestamps into `(word, start_seconds)` pairs (pure).
/// `graph_chars[i]` pairs with `graph_times[i] = [start, end]`; words are split on
/// space graphemes and stamped at the first grapheme's start. A length mismatch
/// yields no words (never panics).
fn words_from_timestamps(ts: &Value) -> Vec<(String, f32)> {
    let chars = ts.get("graph_chars").and_then(|c| c.as_array());
    let times = ts.get("graph_times").and_then(|t| t.as_array());
    let (Some(chars), Some(times)) = (chars, times) else {
        return vec![];
    };
    if chars.len() != times.len() {
        return vec![];
    }
    let mut out = Vec::new();
    let mut word = String::new();
    let mut word_start: Option<f32> = None;
    for (c, t) in chars.iter().zip(times.iter()) {
        let ch = c.as_str().unwrap_or("");
        let start = t
            .as_array()
            .and_then(|pair| pair.first())
            .and_then(|s| s.as_f64())
            .unwrap_or(0.0) as f32;
        if ch == " " {
            if !word.is_empty() {
                out.push((std::mem::take(&mut word), word_start.unwrap_or(0.0)));
                word_start = None;
            }
        } else {
            if word_start.is_none() {
                word_start = Some(start);
            }
            word.push_str(ch);
        }
    }
    if !word.is_empty() {
        out.push((word, word_start.unwrap_or(0.0)));
    }
    out
}

/// Decode one Resemble server message (pure — the wire-fixture seam). `audio` →
/// PCM (and `audio_timestamps` → word timings); `audio_end` ends the run; `error`
/// surfaces the message; anything else is ignored.
pub(crate) fn decode_message(json: Option<&Value>, _binary: Option<&[u8]>) -> Decoded {
    let Some(value) = json else {
        return Decoded::Ignore;
    };
    match value.get("type").and_then(|t| t.as_str()) {
        Some("audio") => {
            // Prefer word timings when present, else the PCM chunk. (One message
            // carries one or the other in practice; if both, audio wins so the
            // chunk is never dropped — the helper interleaves them across messages.)
            if let Some(ts) = value.get("audio_timestamps").filter(|v| !v.is_null()) {
                let words = words_from_timestamps(ts);
                if !words.is_empty()
                    && value
                        .get("audio_content")
                        .and_then(|a| a.as_str())
                        .map(|s| s.is_empty())
                        .unwrap_or(true)
                {
                    return Decoded::Words(words);
                }
            }
            Decoded::Audio(ws_tts::pcm_from_b64_field(value, "audio_content"))
        }
        Some("audio_end") => Decoded::Done,
        Some("error") => {
            let name = value
                .get("error_name")
                .and_then(|n| n.as_str())
                .unwrap_or("Unknown");
            let msg = value
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown error");
            Decoded::Error(format!("{name}: {msg}"))
        }
        _ => Decoded::Ignore,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine as _;

    #[test]
    fn host_is_fixed_and_keyless() {
        assert_eq!(RESEMBLE_WSS, "wss://websocket.cluster.resemble.ai/stream");
        // The key never appears in the URL (it travels in the Authorization header).
        assert!(!RESEMBLE_WSS.contains('?'));
    }

    #[test]
    fn request_body_matches_resemble_schema() {
        let body = build_request("hello there", 3, "voice-uuid", 16_000);
        assert_eq!(body["voice_uuid"], "voice-uuid");
        assert_eq!(body["data"], "hello there");
        assert_eq!(body["binary_response"], false);
        assert_eq!(body["request_id"], 3);
        assert_eq!(body["sample_rate"], 16_000);
        assert_eq!(body["precision"], "PCM_16");
        assert_eq!(body["no_audio_header"], true);
    }

    #[test]
    fn decode_audio_into_pcm() {
        let b64 = base64::engine::general_purpose::STANDARD.encode([1u8, 0, 255, 255]);
        let msg = json!({ "type": "audio", "audio_content": b64, "request_id": 0 });
        match decode_message(Some(&msg), None) {
            Decoded::Audio(pcm) => assert_eq!(pcm, vec![1, -1]),
            _ => panic!("expected Audio"),
        }
    }

    #[test]
    fn decode_timestamps_only_message_into_words() {
        // A timestamps-carrying message with no audio_content → word timings.
        let msg = json!({
            "type": "audio",
            "audio_content": "",
            "request_id": 0,
            "audio_timestamps": {
                "graph_chars": ["b", "o", "o", "k", " ", "i", "t"],
                "graph_times": [[0.0, 0.03], [0.03, 0.06], [0.06, 0.09], [0.09, 0.12],
                                [0.12, 0.15], [0.15, 0.18], [0.18, 0.21]]
            }
        });
        match decode_message(Some(&msg), None) {
            Decoded::Words(words) => {
                assert_eq!(words.len(), 2);
                assert_eq!(words[0].0, "book");
                assert!((words[0].1 - 0.0).abs() < 1e-6);
                assert_eq!(words[1].0, "it");
                assert!((words[1].1 - 0.15).abs() < 1e-6);
            }
            _ => panic!("expected Words"),
        }
    }

    #[test]
    fn decode_audio_end_error_and_ignore() {
        assert!(matches!(
            decode_message(Some(&json!({ "type": "audio_end", "request_id": 0 })), None),
            Decoded::Done
        ));
        match decode_message(
            Some(
                &json!({ "type": "error", "error_name": "InvalidVoice", "message": "no such voice", "status_code": 400 }),
            ),
            None,
        ) {
            Decoded::Error(e) => assert_eq!(e, "InvalidVoice: no such voice"),
            _ => panic!("expected Error"),
        }
        // Unknown type / no type → ignore (no panic).
        assert!(matches!(
            decode_message(Some(&json!({ "type": "phonemes" })), None),
            Decoded::Ignore
        ));
        assert!(matches!(
            decode_message(Some(&json!("nope")), None),
            Decoded::Ignore
        ));
        // Mismatched timestamp lengths → no words; falls through to audio (empty) → Audio([]).
        let bad = json!({
            "type": "audio", "audio_content": "",
            "audio_timestamps": { "graph_chars": ["a"], "graph_times": [[0.0, 0.1], [0.1, 0.2]] }
        });
        assert!(matches!(
            decode_message(Some(&bad), None),
            Decoded::Audio(_)
        ));
    }

    /// Live smoke (requires `RESEMBLE_API_KEY` + `RESEMBLE_VOICE_ID`). Run:
    /// `RESEMBLE_API_KEY=… RESEMBLE_VOICE_ID=… cargo test -p flowcat-services --features tts-resemble -- --ignored resemble_live`
    #[tokio::test]
    #[ignore = "requires RESEMBLE_API_KEY + RESEMBLE_VOICE_ID"]
    async fn resemble_live_synthesizes_audio() {
        let key = std::env::var("RESEMBLE_API_KEY").expect("RESEMBLE_API_KEY");
        let voice = std::env::var("RESEMBLE_VOICE_ID").expect("RESEMBLE_VOICE_ID");
        let mut tts = ResembleTts::new(key, voice);
        tts.start(&StartParams::default()).await.expect("connect");
        let frames = tts.run_tts("Hello from flowcat.").await.expect("run_tts");
        let audio_chunks = frames
            .iter()
            .filter(|f| matches!(f, Frame::TtsAudio { .. }))
            .count();
        assert!(audio_chunks > 0, "expected at least one TtsAudio chunk");
    }
}
