// SPDX-License-Identifier: Apache-2.0
//
//! **Gradium** streaming TTS (`/api/speech/tts` WebSocket).
//!
//! A **(D)istinct** streaming-WebSocket client (cross-checked against pipecat
//! `services/gradium/tts.py`). Connects to the fixed
//! `wss://eu.api.gradium.ai/api/speech/tts` with the API key in the `x-api-key`
//! header, then drives one utterance as a setup → text → end-of-stream sequence
//! keyed by `client_req_id`:
//!
//! ```json
//! { "type": "setup", "output_format": "pcm", "voice_id": "<voice>",
//!   "close_ws_on_eos": false, "client_req_id": "ctx-1" }
//! { "text": "hello there", "type": "text", "client_req_id": "ctx-1" }
//! { "type": "end_of_stream", "client_req_id": "ctx-1" }
//! ```
//!
//! Server messages are JSON:
//!
//! ```json
//! { "type": "audio", "audio": "<base64 pcm>", "client_req_id": "ctx-1" }
//! { "type": "text", "text": "book", "start_s": 0.0, "client_req_id": "ctx-1" }
//! { "type": "end_of_stream", "client_req_id": "ctx-1" }
//! { "type": "error", "message": "bad voice" }
//! ```
//!
//! `audio` base64 → [`Frame::TtsAudio`]; `text` → [`Frame::TtsText`] (a word + its
//! `start_s`); `end_of_stream` ends the run; `error` surfaces the message; `ready`
//! is ignored. All decode is pure.

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

/// Gradium's TTS WebSocket host (the host is fixed).
pub const GRADIUM_WSS: &str = "wss://eu.api.gradium.ai/api/speech/tts";

/// Gradium streaming-TTS session.
pub struct GradiumTts {
    api_key: String,
    voice_id: String,
    sample_rate: u32,
    session: Option<WsTtsSession>,
    ctx_counter: u64,
}

impl GradiumTts {
    /// Construct bound to `api_key` + `voice_id` (default 48 kHz PCM output —
    /// Gradium's native rate).
    pub fn new(api_key: impl Into<String>, voice_id: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            voice_id: voice_id.into(),
            sample_rate: 48_000,
            session: None,
            ctx_counter: 0,
        }
    }

    /// Override the output sample rate (default 48 kHz).
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }
}

#[async_trait]
impl TtsService for GradiumTts {
    fn name(&self) -> &str {
        "gradium"
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        let cfg = WsTtsConfig {
            url: GRADIUM_WSS.to_string(),
            headers: vec![
                ("x-api-key".to_string(), self.api_key.clone()),
                ("x-api-source".to_string(), "flowcat".to_string()),
            ],
            init_message: None,
            decode: decode_message,
        };
        self.session = Some(WsTtsSession::connect(cfg).await?);
        Ok(())
    }

    async fn run_tts(&mut self, text: &str) -> Result<Vec<Frame>> {
        self.ctx_counter += 1;
        let context_id: Arc<str> = Arc::from(format!("ctx-{}", self.ctx_counter));
        let rate = self.sample_rate;
        let msgs = build_messages(text, &context_id, &self.voice_id);
        let session = ws_tts::require(&mut self.session, "gradium")?;
        session.synthesize(msgs, context_id, rate).await
    }
}

/// The setup + text + end-of-stream message sequence for one utterance (pure —
/// the wire-fixture seam).
fn build_messages(text: &str, context_id: &str, voice_id: &str) -> Vec<OutMsg> {
    let setup = json!({
        "type": "setup",
        "output_format": "pcm",
        "voice_id": voice_id,
        "close_ws_on_eos": false,
        "client_req_id": context_id,
    });
    let text_msg = json!({ "text": text, "type": "text", "client_req_id": context_id });
    let eos = json!({ "type": "end_of_stream", "client_req_id": context_id });
    vec![
        OutMsg::Text(setup.to_string()),
        OutMsg::Text(text_msg.to_string()),
        OutMsg::Text(eos.to_string()),
    ]
}

/// Decode one Gradium server message (pure — the wire-fixture seam). `audio` →
/// PCM; `text` → a word timing; `end_of_stream` ends the run; `error` surfaces the
/// message; `ready` / binary / anything else is ignored.
pub(crate) fn decode_message(json: Option<&Value>, _binary: Option<&[u8]>) -> Decoded {
    let Some(value) = json else {
        return Decoded::Ignore;
    };
    match value.get("type").and_then(|t| t.as_str()) {
        Some("audio") => Decoded::Audio(ws_tts::pcm_from_b64_field(value, "audio")),
        Some("text") => {
            let word = value
                .get("text")
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_string();
            let start = value.get("start_s").and_then(|s| s.as_f64()).unwrap_or(0.0) as f32;
            if word.is_empty() {
                Decoded::Ignore
            } else {
                Decoded::Words(vec![(word, start)])
            }
        }
        Some("end_of_stream") => Decoded::Done,
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
    fn host_is_fixed_and_keyless() {
        assert_eq!(GRADIUM_WSS, "wss://eu.api.gradium.ai/api/speech/tts");
        // The key never appears in the URL (it travels in the x-api-key header).
        assert!(!GRADIUM_WSS.contains('?'));
    }

    #[test]
    fn build_messages_is_setup_text_eos() {
        let msgs = build_messages("hello there", "ctx-1", "voice-x");
        assert_eq!(msgs.len(), 3);
        let texts: Vec<Value> = msgs
            .iter()
            .map(|m| match m {
                OutMsg::Text(t) => serde_json::from_str(t).unwrap(),
                OutMsg::Binary(_) => panic!("expected text"),
            })
            .collect();
        assert_eq!(texts[0]["type"], "setup");
        assert_eq!(texts[0]["voice_id"], "voice-x");
        assert_eq!(texts[0]["client_req_id"], "ctx-1");
        assert_eq!(texts[1]["type"], "text");
        assert_eq!(texts[1]["text"], "hello there");
        assert_eq!(texts[2]["type"], "end_of_stream");
    }

    #[test]
    fn decode_audio_text_eos_error_ignore() {
        let b64 = base64::engine::general_purpose::STANDARD.encode([1u8, 0, 255, 255]);
        match decode_message(Some(&json!({ "type": "audio", "audio": b64 })), None) {
            Decoded::Audio(pcm) => assert_eq!(pcm, vec![1, -1]),
            _ => panic!("expected Audio"),
        }
        match decode_message(
            Some(&json!({ "type": "text", "text": "book", "start_s": 0.25 })),
            None,
        ) {
            Decoded::Words(words) => {
                assert_eq!(words[0].0, "book");
                assert!((words[0].1 - 0.25).abs() < 1e-6);
            }
            _ => panic!("expected Words"),
        }
        assert!(matches!(
            decode_message(Some(&json!({ "type": "end_of_stream" })), None),
            Decoded::Done
        ));
        match decode_message(
            Some(&json!({ "type": "error", "message": "bad voice" })),
            None,
        ) {
            Decoded::Error(e) => assert_eq!(e, "bad voice"),
            _ => panic!("expected Error"),
        }
        // ready / unknown → ignore (no panic).
        assert!(matches!(
            decode_message(Some(&json!({ "type": "ready" })), None),
            Decoded::Ignore
        ));
        assert!(matches!(
            decode_message(Some(&json!("nope")), None),
            Decoded::Ignore
        ));
    }

    /// Live smoke (requires `GRADIUM_API_KEY` + `GRADIUM_VOICE_ID`). Run:
    /// `GRADIUM_API_KEY=… GRADIUM_VOICE_ID=… cargo test -p flowcat-services --features tts-gradium -- --ignored gradium_live`
    #[tokio::test]
    #[ignore = "requires GRADIUM_API_KEY + GRADIUM_VOICE_ID"]
    async fn gradium_live_synthesizes_audio() {
        let key = std::env::var("GRADIUM_API_KEY").expect("GRADIUM_API_KEY");
        let voice = std::env::var("GRADIUM_VOICE_ID").expect("GRADIUM_VOICE_ID");
        let mut tts = GradiumTts::new(key, voice);
        tts.start(&StartParams::default()).await.expect("connect");
        let frames = tts.run_tts("Hello from flowcat.").await.expect("run_tts");
        let audio_chunks = frames
            .iter()
            .filter(|f| matches!(f, Frame::TtsAudio { .. }))
            .count();
        assert!(audio_chunks > 0, "expected at least one TtsAudio chunk");
    }
}
