// SPDX-License-Identifier: Apache-2.0
//
//! **Rime** streaming TTS (`/ws3` WebSocket).
//!
//! A **(D)istinct** streaming-WebSocket client (cross-checked against pipecat
//! `services/rime/tts.py`). Connects to the fixed
//! `wss://users-ws.rime.ai/ws3?speaker=<voice>&modelId=<model>&audioFormat=pcm&samplingRate=<rate>`
//! with the API key in the `Authorization: Bearer <key>` header, then per utterance
//! sends a text message then an `eos` control:
//!
//! ```json
//! { "text": "hello there", "contextId": "ctx-1" }
//! { "operation": "eos" }
//! ```
//!
//! Server messages are JSON (cross-checked against pipecat):
//!
//! ```json
//! { "type": "chunk", "data": "<base64 pcm>", "contextId": "ctx-1" }
//! { "type": "timestamps", "word_timestamps": { "words": ["book"], "start": [0.0], "end": [0.3] } }
//! { "type": "done", "contextId": "ctx-1" }
//! { "type": "error", "message": "bad voice" }
//! ```
//!
//! `chunk` base64 → [`Frame::TtsAudio`]; `timestamps` → [`Frame::TtsText`] per
//! word; `done` ends the run; `error` surfaces the message. All decode is pure.

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

/// Rime's TTS WebSocket host. The query is appended at connect time.
pub const RIME_WSS_BASE: &str = "wss://users-ws.rime.ai/ws3";

/// Rime streaming-TTS session.
pub struct RimeTts {
    api_key: String,
    voice_id: String,
    model: String,
    sample_rate: u32,
    ctx_counter: u64,
}

impl RimeTts {
    /// Construct bound to `api_key` + `voice_id` (default `mistv2` model, 24 kHz
    /// raw PCM output).
    pub fn new(api_key: impl Into<String>, voice_id: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            voice_id: voice_id.into(),
            model: "mistv2".to_string(),
            sample_rate: 24_000,
            ctx_counter: 0,
        }
    }

    /// Override the model (default `mistv2`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the output sample rate (default 24 kHz).
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// The `/ws3` connect URL (host fixed; key in the header, not the URL).
    fn url(&self) -> String {
        format!(
            "{RIME_WSS_BASE}?speaker={}&modelId={}&audioFormat=pcm&samplingRate={}",
            self.voice_id, self.model, self.sample_rate
        )
    }
}

#[async_trait]
impl TtsService for RimeTts {
    fn name(&self) -> &str {
        "rime"
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        // Per-utterance reconnect: Rime's `/ws3` closes the socket after each
        // utterance's `eos`, so a reused socket goes stale (alternating silence +
        // off-by-one audio). Open a FRESH socket in each `run_tts` instead — nothing
        // to connect up-front here. (Isolated to this provider; the shared WS-TTS
        // helper still reuses one socket for the providers that support it.)
        Ok(())
    }

    async fn run_tts(&mut self, text: &str) -> Result<Vec<Frame>> {
        self.ctx_counter += 1;
        let context_id: Arc<str> = Arc::from(format!("ctx-{}", self.ctx_counter));
        let rate = self.sample_rate;
        let msgs = build_messages(text, &context_id);
        let cfg = WsTtsConfig {
            url: self.url(),
            headers: vec![(
                "Authorization".to_string(),
                format!("Bearer {}", self.api_key),
            )],
            init_message: None,
            decode: decode_message,
        };
        // Fresh socket for this utterance (dropped when it returns).
        let mut session = WsTtsSession::connect(cfg).await?;
        session.synthesize(msgs, context_id, rate).await
    }
}

/// The text + `eos` message sequence for one utterance (pure — the wire-fixture
/// seam).
fn build_messages(text: &str, context_id: &str) -> Vec<OutMsg> {
    let speak = json!({ "text": text, "contextId": context_id });
    let eos = json!({ "operation": "eos" });
    vec![
        OutMsg::Text(speak.to_string()),
        OutMsg::Text(eos.to_string()),
    ]
}

/// Pull `(word, start_seconds)` pairs from a Rime `word_timestamps` object (pure).
fn words_from_timestamps(ts: &Value) -> Vec<(String, f32)> {
    let words = ts.get("words").and_then(|w| w.as_array());
    let starts = ts.get("start").and_then(|s| s.as_array());
    let (Some(words), Some(starts)) = (words, starts) else {
        return vec![];
    };
    words
        .iter()
        .zip(starts.iter())
        .filter_map(|(w, s)| {
            let word = w.as_str()?.to_string();
            let start = s.as_f64().unwrap_or(0.0) as f32;
            Some((word, start))
        })
        .collect()
}

/// Decode one Rime server message (pure — the wire-fixture seam). `chunk` →
/// audio; `timestamps` → word timings; `done` ends the run; `error` surfaces the
/// message; binary / anything else is ignored.
pub(crate) fn decode_message(json: Option<&Value>, _binary: Option<&[u8]>) -> Decoded {
    let Some(value) = json else {
        return Decoded::Ignore;
    };
    match value.get("type").and_then(|t| t.as_str()) {
        Some("chunk") => Decoded::Audio(ws_tts::pcm_from_b64_field(value, "data")),
        Some("timestamps") => {
            let words = value
                .get("word_timestamps")
                .map(words_from_timestamps)
                .unwrap_or_default();
            if words.is_empty() {
                Decoded::Ignore
            } else {
                Decoded::Words(words)
            }
        }
        Some("done") => Decoded::Done,
        Some("error") => Decoded::Error(
            value
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown")
                .to_string(),
        ),
        _ => Decoded::Ignore,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine as _;

    #[test]
    fn url_uses_the_fixed_host_with_speaker_and_format() {
        let t = RimeTts::new("secret-key-xyz", "cove").sample_rate(16_000);
        let url = t.url();
        assert!(url.starts_with("wss://users-ws.rime.ai/ws3?"));
        assert!(url.contains("speaker=cove"));
        assert!(url.contains("modelId=mistv2"));
        assert!(url.contains("audioFormat=pcm"));
        assert!(url.contains("samplingRate=16000"));
        // The key never appears in the URL (it travels in the Authorization header).
        assert!(!url.contains("secret-key-xyz"));
    }

    #[test]
    fn build_messages_is_text_then_eos() {
        let msgs = build_messages("hello there", "ctx-1");
        assert_eq!(msgs.len(), 2);
        let texts: Vec<Value> = msgs
            .iter()
            .map(|m| match m {
                OutMsg::Text(t) => serde_json::from_str(t).unwrap(),
                OutMsg::Binary(_) => panic!("expected text"),
            })
            .collect();
        assert_eq!(texts[0]["text"], "hello there");
        assert_eq!(texts[0]["contextId"], "ctx-1");
        assert_eq!(texts[1]["operation"], "eos");
    }

    #[test]
    fn decode_chunk_into_pcm() {
        let b64 = base64::engine::general_purpose::STANDARD.encode([1u8, 0, 255, 255]);
        let msg = json!({ "type": "chunk", "data": b64, "contextId": "ctx-1" });
        match decode_message(Some(&msg), None) {
            Decoded::Audio(pcm) => assert_eq!(pcm, vec![1, -1]),
            _ => panic!("expected Audio"),
        }
    }

    #[test]
    fn decode_timestamps_into_word_timestamps() {
        let msg = json!({
            "type": "timestamps",
            "word_timestamps": { "words": ["book", "it"], "start": [0.0, 0.3], "end": [0.3, 0.5] }
        });
        match decode_message(Some(&msg), None) {
            Decoded::Words(words) => {
                assert_eq!(words.len(), 2);
                assert_eq!(words[0].0, "book");
                assert!((words[1].1 - 0.3).abs() < 1e-6);
            }
            _ => panic!("expected Words"),
        }
    }

    #[test]
    fn decode_done_error_and_ignore() {
        assert!(matches!(
            decode_message(Some(&json!({ "type": "done", "contextId": "ctx-1" })), None),
            Decoded::Done
        ));
        match decode_message(
            Some(&json!({ "type": "error", "message": "bad voice" })),
            None,
        ) {
            Decoded::Error(e) => assert_eq!(e, "bad voice"),
            _ => panic!("expected Error"),
        }
        // Unknown / no type → ignore (no panic).
        assert!(matches!(
            decode_message(Some(&json!({ "type": "open" })), None),
            Decoded::Ignore
        ));
        assert!(matches!(
            decode_message(Some(&json!("nope")), None),
            Decoded::Ignore
        ));
        // Empty timestamps → ignore.
        assert!(matches!(
            decode_message(
                Some(&json!({ "type": "timestamps", "word_timestamps": {} })),
                None
            ),
            Decoded::Ignore
        ));
    }

    /// Live smoke (requires `RIME_API_KEY` + `RIME_VOICE_ID`). Run:
    /// `RIME_API_KEY=… RIME_VOICE_ID=… cargo test -p flowcat-services --features tts-rime -- --ignored rime_live`
    #[tokio::test]
    #[ignore = "requires RIME_API_KEY + RIME_VOICE_ID"]
    async fn rime_live_synthesizes_audio() {
        let key = std::env::var("RIME_API_KEY").expect("RIME_API_KEY");
        let voice = std::env::var("RIME_VOICE_ID").expect("RIME_VOICE_ID");
        let mut tts = RimeTts::new(key, voice);
        tts.start(&StartParams::default()).await.expect("connect");
        let frames = tts.run_tts("Hello from flowcat.").await.expect("run_tts");
        let audio_chunks = frames
            .iter()
            .filter(|f| matches!(f, Frame::TtsAudio { .. }))
            .count();
        assert!(audio_chunks > 0, "expected at least one TtsAudio chunk");
    }
}
