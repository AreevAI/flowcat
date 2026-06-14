// SPDX-License-Identifier: Apache-2.0
//
//! **Deepgram** streaming TTS (`/v1/speak` WebSocket).
//!
//! A **(D)istinct** streaming-WebSocket client (cross-checked against pipecat
//! `services/deepgram/tts.py`). Connects to the fixed
//! `wss://api.deepgram.com/v1/speak?model=<voice>&encoding=linear16&sample_rate=<rate>`
//! with the API key in the `Authorization: Token <key>` header, then per utterance
//! sends a `Speak` then a `Flush` control message:
//!
//! ```json
//! { "type": "Speak", "text": "hello there" }
//! { "type": "Flush" }
//! ```
//!
//! Unlike the JSON-base64 providers, Deepgram returns audio as **binary** WS
//! frames (raw linear16 PCM); JSON text frames are control/metadata, of which
//! `{ "type": "Flushed" }` marks the end of the utterance. Binary frames →
//! [`Frame::TtsAudio`]; `Flushed` ends the run; `{ "type": "Warning" }` is logged
//! (ignored here). Deepgram WS TTS does not emit word timestamps.

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

/// Deepgram's TTS WebSocket host. The query is appended at connect time.
pub const DEEPGRAM_WSS_BASE: &str = "wss://api.deepgram.com";

/// Deepgram streaming-TTS session.
pub struct DeepgramTts {
    api_key: String,
    voice_id: String,
    sample_rate: u32,
    session: Option<WsTtsSession>,
    ctx_counter: u64,
}

impl DeepgramTts {
    /// Construct bound to `api_key` + `voice_id` (default `aura-2-thalia-en`
    /// model, 24 kHz linear16 PCM output).
    pub fn new(api_key: impl Into<String>, voice_id: impl Into<String>) -> Self {
        let voice_id = voice_id.into();
        let voice_id = if voice_id.is_empty() {
            "aura-2-thalia-en".to_string()
        } else {
            voice_id
        };
        Self {
            api_key: api_key.into(),
            voice_id,
            sample_rate: 24_000,
            session: None,
            ctx_counter: 0,
        }
    }

    /// Override the output sample rate (default 24 kHz).
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// The `/v1/speak` connect URL (host fixed; key in the header, not the URL).
    fn url(&self) -> String {
        format!(
            "{DEEPGRAM_WSS_BASE}/v1/speak?model={}&encoding=linear16&sample_rate={}",
            self.voice_id, self.sample_rate
        )
    }
}

#[async_trait]
impl TtsService for DeepgramTts {
    fn name(&self) -> &str {
        "deepgram"
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        let cfg = WsTtsConfig {
            url: self.url(),
            headers: vec![(
                "Authorization".to_string(),
                format!("Token {}", self.api_key),
            )],
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
        let msgs = build_messages(text);
        let session = ws_tts::require(&mut self.session, "deepgram")?;
        session.synthesize(msgs, context_id, rate).await
    }
}

/// The `Speak` + `Flush` control message sequence for one utterance (pure — the
/// wire-fixture seam).
fn build_messages(text: &str) -> Vec<OutMsg> {
    let speak = json!({ "type": "Speak", "text": text });
    let flush = json!({ "type": "Flush" });
    vec![
        OutMsg::Text(speak.to_string()),
        OutMsg::Text(flush.to_string()),
    ]
}

/// Decode one Deepgram server message (pure — the wire-fixture seam). A **binary**
/// frame is raw linear16 PCM → audio; `{ "type": "Flushed" }` ends the run;
/// `Metadata` / `Warning` / anything else is ignored.
pub(crate) fn decode_message(json: Option<&Value>, binary: Option<&[u8]>) -> Decoded {
    if let Some(bytes) = binary {
        return Decoded::Audio(ws_tts::pcm_from_le_bytes(bytes));
    }
    let Some(value) = json else {
        return Decoded::Ignore;
    };
    match value.get("type").and_then(|t| t.as_str()) {
        Some("Flushed") => Decoded::Done,
        _ => Decoded::Ignore,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_uses_the_fixed_host_with_model_and_format() {
        let t = DeepgramTts::new("secret-key-xyz", "aura-2-asteria-en").sample_rate(16_000);
        let url = t.url();
        assert!(url.starts_with("wss://api.deepgram.com/v1/speak?"));
        assert!(url.contains("model=aura-2-asteria-en"));
        assert!(url.contains("encoding=linear16"));
        assert!(url.contains("sample_rate=16000"));
        // The key never appears in the URL (it travels in the Authorization header).
        assert!(!url.contains("secret-key-xyz"));
    }

    #[test]
    fn empty_voice_id_defaults_to_a_model() {
        let t = DeepgramTts::new("k", "");
        assert!(t.url().contains("model=aura-2-thalia-en"));
    }

    #[test]
    fn build_messages_is_speak_then_flush() {
        let msgs = build_messages("hello there");
        assert_eq!(msgs.len(), 2);
        let texts: Vec<Value> = msgs
            .iter()
            .map(|m| match m {
                OutMsg::Text(t) => serde_json::from_str(t).unwrap(),
                OutMsg::Binary(_) => panic!("expected text"),
            })
            .collect();
        assert_eq!(texts[0]["type"], "Speak");
        assert_eq!(texts[0]["text"], "hello there");
        assert_eq!(texts[1]["type"], "Flush");
    }

    #[test]
    fn decode_binary_into_pcm() {
        // Two LE i16 samples: 1 and -1 → bytes [1,0, 255,255].
        match decode_message(None, Some(&[1u8, 0, 255, 255])) {
            Decoded::Audio(pcm) => assert_eq!(pcm, vec![1, -1]),
            _ => panic!("expected Audio"),
        }
    }

    #[test]
    fn decode_flushed_and_ignore_and_malformed() {
        assert!(matches!(
            decode_message(Some(&json!({ "type": "Flushed" })), None),
            Decoded::Done
        ));
        // Metadata / Warning / unknown control → ignore.
        assert!(matches!(
            decode_message(Some(&json!({ "type": "Metadata" })), None),
            Decoded::Ignore
        ));
        assert!(matches!(
            decode_message(
                Some(&json!({ "type": "Warning", "description": "x" })),
                None
            ),
            Decoded::Ignore
        ));
        // No type field at all → ignore (no panic).
        assert!(matches!(
            decode_message(Some(&json!({ "foo": 1 })), None),
            Decoded::Ignore
        ));
    }

    /// Live smoke (requires `DEEPGRAM_API_KEY`). Run:
    /// `DEEPGRAM_API_KEY=… cargo test -p flowcat-services --features tts-deepgram -- --ignored deepgram_live`
    #[tokio::test]
    #[ignore = "requires DEEPGRAM_API_KEY"]
    async fn deepgram_live_synthesizes_audio() {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let key = std::env::var("DEEPGRAM_API_KEY").expect("DEEPGRAM_API_KEY");
        let mut tts = DeepgramTts::new(key, "aura-2-thalia-en");
        tts.start(&StartParams::default()).await.expect("connect");
        let frames = tts.run_tts("Hello from flowcat.").await.expect("run_tts");
        let audio_chunks = frames
            .iter()
            .filter(|f| matches!(f, Frame::TtsAudio { .. }))
            .count();
        assert!(audio_chunks > 0, "expected at least one TtsAudio chunk");
    }
}
