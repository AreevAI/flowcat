// SPDX-License-Identifier: Apache-2.0
//
//! **Soniox** streaming TTS (`/tts-websocket`).
//!
//! A **(D)istinct** streaming-WebSocket client (cross-checked against pipecat
//! `services/soniox/tts.py`). Connects bare to the fixed
//! `wss://tts-rt.soniox.com/tts-websocket` (no header/query auth — the API key
//! travels in the per-stream config message), then per utterance opens a stream,
//! pushes text, and ends it:
//!
//! ```json
//! { "api_key": "<key>", "stream_id": "ctx-1", "model": "tts-rt-v1",
//!   "voice": "Adrian", "audio_format": "pcm_s16le", "sample_rate": 24000, "language": "en" }
//! { "text": "hello there", "text_end": false, "stream_id": "ctx-1" }
//! { "text": "", "text_end": true, "stream_id": "ctx-1" }    // flush + end
//! ```
//!
//! Server messages: `{ "audio": "<base64 pcm>", "stream_id": "ctx-1" }` chunks,
//! the terminal `{ "terminated": true, "stream_id": "ctx-1" }`, and errors
//! `{ "error_code": 401, "error_message": "…", "stream_id": "ctx-1" }`. Base64 PCM
//! → [`Frame::TtsAudio`]; `terminated` ends the run; an `error_code` surfaces the
//! message. Soniox TTS does not emit word timestamps. All decode is pure.

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

/// Soniox's fixed real-time TTS WSS. The **host is fixed**; the API key is sent
/// only in the per-stream config message, never in the URL.
pub const SONIOX_TTS_WSS: &str = "wss://tts-rt.soniox.com/tts-websocket";

/// Soniox streaming-TTS session.
pub struct SonioxTts {
    api_key: String,
    voice_id: String,
    model: String,
    sample_rate: u32,
    session: Option<WsTtsSession>,
    ctx_counter: u64,
}

impl SonioxTts {
    /// Construct bound to `api_key` + `voice_id` (default `tts-rt-v1` model,
    /// 24 kHz raw PCM output). If `voice_id` is empty, defaults to `Adrian`.
    pub fn new(api_key: impl Into<String>, voice_id: impl Into<String>) -> Self {
        let voice_id = voice_id.into();
        let voice_id = if voice_id.is_empty() {
            "Adrian".to_string()
        } else {
            voice_id
        };
        Self {
            api_key: api_key.into(),
            voice_id,
            model: "tts-rt-v1".to_string(),
            sample_rate: 24_000,
            session: None,
            ctx_counter: 0,
        }
    }

    /// Override the model (default `tts-rt-v1`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the output sample rate (default 24 kHz; Soniox accepts
    /// 8000/16000/24000/44100/48000 for raw PCM).
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// The per-stream config message that opens a stream (pure — the wire-fixture
    /// seam). Carries the API key + voice + audio shape.
    fn config_message(&self, stream_id: &str) -> Value {
        json!({
            "api_key": self.api_key,
            "stream_id": stream_id,
            "model": self.model,
            "voice": self.voice_id,
            "audio_format": "pcm_s16le",
            "sample_rate": self.sample_rate,
            "language": "en",
        })
    }
}

#[async_trait]
impl TtsService for SonioxTts {
    fn name(&self) -> &str {
        "soniox"
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        let cfg = WsTtsConfig {
            url: SONIOX_TTS_WSS.to_string(),
            headers: vec![],
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
        let config = self.config_message(&context_id);
        let msgs = build_messages(config, text, &context_id);
        let session = ws_tts::require(&mut self.session, "soniox")?;
        session.synthesize(msgs, context_id, rate).await
    }
}

/// The config + text + text-end message sequence for one utterance (pure — the
/// wire-fixture seam). Each utterance opens a fresh `stream_id`.
fn build_messages(config: Value, text: &str, stream_id: &str) -> Vec<OutMsg> {
    let speak = json!({ "text": text, "text_end": false, "stream_id": stream_id });
    let end = json!({ "text": "", "text_end": true, "stream_id": stream_id });
    vec![
        OutMsg::Text(config.to_string()),
        OutMsg::Text(speak.to_string()),
        OutMsg::Text(end.to_string()),
    ]
}

/// Decode one Soniox server message (pure — the wire-fixture seam). An
/// `error_code` surfaces the error; `terminated: true` ends the run; an `audio`
/// field → PCM; anything else is ignored.
pub(crate) fn decode_message(json: Option<&Value>, _binary: Option<&[u8]>) -> Decoded {
    let Some(value) = json else {
        return Decoded::Ignore;
    };
    if let Some(code) = value.get("error_code") {
        let msg = value
            .get("error_message")
            .and_then(|m| m.as_str())
            .unwrap_or("");
        return Decoded::Error(format!("{code} {msg}").trim().to_string());
    }
    if value.get("terminated").and_then(|t| t.as_bool()) == Some(true) {
        return Decoded::Done;
    }
    if value.get("audio").and_then(|a| a.as_str()).is_some() {
        return Decoded::Audio(ws_tts::pcm_from_b64_field(value, "audio"));
    }
    Decoded::Ignore
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine as _;

    #[test]
    fn config_carries_key_and_audio_shape() {
        let c = SonioxTts::new("secret", "Daniel")
            .sample_rate(16_000)
            .config_message("ctx-1");
        assert_eq!(c["api_key"], "secret");
        assert_eq!(c["stream_id"], "ctx-1");
        assert_eq!(c["voice"], "Daniel");
        assert_eq!(c["audio_format"], "pcm_s16le");
        assert_eq!(c["sample_rate"], 16_000);
        // The connect URL is the fixed host (no key in it).
        assert_eq!(SONIOX_TTS_WSS, "wss://tts-rt.soniox.com/tts-websocket");
    }

    #[test]
    fn empty_voice_defaults_to_adrian() {
        let c = SonioxTts::new("k", "").config_message("ctx-1");
        assert_eq!(c["voice"], "Adrian");
    }

    #[test]
    fn build_messages_is_config_text_then_end() {
        let config = json!({ "stream_id": "ctx-1" });
        let msgs = build_messages(config, "hello there", "ctx-1");
        assert_eq!(msgs.len(), 3);
        let texts: Vec<Value> = msgs
            .iter()
            .map(|m| match m {
                OutMsg::Text(t) => serde_json::from_str(t).unwrap(),
                OutMsg::Binary(_) => panic!("expected text"),
            })
            .collect();
        assert_eq!(texts[1]["text"], "hello there");
        assert_eq!(texts[1]["text_end"], false);
        assert_eq!(texts[2]["text"], "");
        assert_eq!(texts[2]["text_end"], true);
    }

    #[test]
    fn decode_audio_terminated_error_ignore() {
        let b64 = base64::engine::general_purpose::STANDARD.encode([1u8, 0, 255, 255]);
        match decode_message(Some(&json!({ "audio": b64, "stream_id": "ctx-1" })), None) {
            Decoded::Audio(pcm) => assert_eq!(pcm, vec![1, -1]),
            _ => panic!("expected Audio"),
        }
        assert!(matches!(
            decode_message(
                Some(&json!({ "terminated": true, "stream_id": "ctx-1" })),
                None
            ),
            Decoded::Done
        ));
        match decode_message(
            Some(&json!({ "error_code": 401, "error_message": "bad key" })),
            None,
        ) {
            Decoded::Error(e) => assert_eq!(e, "401 bad key"),
            _ => panic!("expected Error"),
        }
        // audio_end (informational) / unknown → ignore (no panic).
        assert!(matches!(
            decode_message(Some(&json!({ "audio_end": true })), None),
            Decoded::Ignore
        ));
        assert!(matches!(
            decode_message(Some(&json!("nope")), None),
            Decoded::Ignore
        ));
    }

    /// Live smoke (requires `SONIOX_API_KEY`). Run:
    /// `SONIOX_API_KEY=… cargo test -p flowcat-services --features tts-soniox -- --ignored soniox_tts_live`
    #[tokio::test]
    #[ignore = "requires SONIOX_API_KEY"]
    async fn soniox_tts_live_synthesizes_audio() {
        let key = std::env::var("SONIOX_API_KEY").expect("SONIOX_API_KEY");
        let mut tts = SonioxTts::new(key, "Adrian");
        tts.start(&StartParams::default()).await.expect("connect");
        let frames = tts.run_tts("Hello from flowcat.").await.expect("run_tts");
        let audio_chunks = frames
            .iter()
            .filter(|f| matches!(f, Frame::TtsAudio { .. }))
            .count();
        assert!(audio_chunks > 0, "expected at least one TtsAudio chunk");
    }
}
