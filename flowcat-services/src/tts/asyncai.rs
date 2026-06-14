// SPDX-License-Identifier: Apache-2.0
//
//! **AsyncAI** streaming TTS (Async `/text_to_speech/websocket/ws`).
//!
//! A **(D)istinct** streaming-WebSocket client (cross-checked against pipecat
//! `services/asyncai/tts.py`). A Cartesia-shaped sibling: connect to the fixed
//! `wss://api.async.com/text_to_speech/websocket/ws?api_key=<key>&version=v1`,
//! send a JSON **init** message once at connect (voice + output format), then per
//! utterance a `transcript` message followed by a `force` flush:
//!
//! ```json
//! // init (sent once at connect):
//! { "model_id": "asyncflow_v2.0", "voice": { "mode": "id", "id": "<voice>" },
//!   "output_format": { "container": "raw", "encoding": "pcm_s16le", "sample_rate": 24000 } }
//! // per utterance:
//! { "transcript": "hello there", "context_id": "ctx-1", "force": false }
//! { "transcript": " ", "context_id": "ctx-1", "force": true }   // flush + end
//! ```
//!
//! Server messages are `{ "audio": "<base64 pcm>", "context_id": "ctx-1" }` chunks
//! followed by a terminal `{ "final": true, "context_id": "ctx-1" }`. Base64 PCM
//! → [`Frame::TtsAudio`]; `final` ends the run. AsyncAI WS does not emit word
//! timestamps. Note the API key is a **required query param** for this provider's
//! handshake (it has no header-auth form) — the host is still fixed, so there is
//! no request-derived URL.

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

/// AsyncAI's TTS WebSocket host. The `api_key` + `version` query is appended at
/// connect time (the provider requires the key as a query param).
pub const ASYNCAI_WSS_BASE: &str = "wss://api.async.com/text_to_speech/websocket/ws";
/// The Async API version this client speaks.
pub const ASYNCAI_VERSION: &str = "v1";

/// AsyncAI streaming-TTS session.
pub struct AsyncAiTts {
    api_key: String,
    voice_id: String,
    model: String,
    sample_rate: u32,
    session: Option<WsTtsSession>,
    ctx_counter: u64,
}

impl AsyncAiTts {
    /// Construct bound to `api_key` + `voice_id` (default `asyncflow_v2.0` model,
    /// 24 kHz raw PCM output).
    pub fn new(api_key: impl Into<String>, voice_id: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            voice_id: voice_id.into(),
            model: "asyncflow_v2.0".to_string(),
            sample_rate: 24_000,
            session: None,
            ctx_counter: 0,
        }
    }

    /// Override the model (default `asyncflow_v2.0`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the output sample rate (default 24 kHz).
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    fn url(&self) -> String {
        format!(
            "{ASYNCAI_WSS_BASE}?api_key={}&version={ASYNCAI_VERSION}",
            self.api_key
        )
    }

    /// The init message sent once at connect (pure — the wire-fixture seam).
    fn init_message(&self) -> Value {
        json!({
            "model_id": self.model,
            "voice": { "mode": "id", "id": self.voice_id },
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": self.sample_rate,
            },
        })
    }
}

#[async_trait]
impl TtsService for AsyncAiTts {
    fn name(&self) -> &str {
        "asyncai"
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        let cfg = WsTtsConfig {
            url: self.url(),
            headers: vec![],
            init_message: Some(self.init_message().to_string()),
            decode: decode_message,
        };
        self.session = Some(WsTtsSession::connect(cfg).await?);
        Ok(())
    }

    async fn run_tts(&mut self, text: &str) -> Result<Vec<Frame>> {
        self.ctx_counter += 1;
        let context_id: Arc<str> = Arc::from(format!("ctx-{}", self.ctx_counter));
        let rate = self.sample_rate;
        let msgs = build_messages(text, &context_id);
        let session = ws_tts::require(&mut self.session, "asyncai")?;
        session.synthesize(msgs, context_id, rate).await
    }
}

/// The transcript + force-flush message sequence for one utterance (pure — the
/// wire-fixture seam).
fn build_messages(text: &str, context_id: &str) -> Vec<OutMsg> {
    let speak = json!({ "transcript": text, "context_id": context_id, "force": false });
    let flush = json!({ "transcript": " ", "context_id": context_id, "force": true });
    vec![
        OutMsg::Text(speak.to_string()),
        OutMsg::Text(flush.to_string()),
    ]
}

/// Decode one AsyncAI server message (pure — the wire-fixture seam). An `audio`
/// field → PCM; `final: true` ends the run; binary / anything else is ignored.
pub(crate) fn decode_message(json: Option<&Value>, _binary: Option<&[u8]>) -> Decoded {
    let Some(value) = json else {
        return Decoded::Ignore;
    };
    if value.get("final").and_then(|f| f.as_bool()) == Some(true) {
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
    fn url_uses_the_fixed_host_with_version() {
        let t = AsyncAiTts::new("k", "voice-x");
        let url = t.url();
        assert!(url.starts_with("wss://api.async.com/text_to_speech/websocket/ws?"));
        assert!(url.contains("version=v1"));
        // This provider requires the key as a query param.
        assert!(url.contains("api_key=k"));
    }

    #[test]
    fn init_carries_voice_and_output_format() {
        let init = AsyncAiTts::new("k", "voice-x")
            .sample_rate(16_000)
            .init_message();
        assert_eq!(init["model_id"], "asyncflow_v2.0");
        assert_eq!(init["voice"]["mode"], "id");
        assert_eq!(init["voice"]["id"], "voice-x");
        assert_eq!(init["output_format"]["container"], "raw");
        assert_eq!(init["output_format"]["encoding"], "pcm_s16le");
        assert_eq!(init["output_format"]["sample_rate"], 16_000);
    }

    #[test]
    fn build_messages_is_transcript_then_force_flush() {
        let msgs = build_messages("hello there", "ctx-1");
        assert_eq!(msgs.len(), 2);
        let texts: Vec<Value> = msgs
            .iter()
            .map(|m| match m {
                OutMsg::Text(t) => serde_json::from_str(t).unwrap(),
                OutMsg::Binary(_) => panic!("expected text"),
            })
            .collect();
        assert_eq!(texts[0]["transcript"], "hello there");
        assert_eq!(texts[0]["context_id"], "ctx-1");
        assert_eq!(texts[0]["force"], false);
        assert_eq!(texts[1]["force"], true);
    }

    #[test]
    fn decode_audio_into_pcm() {
        let b64 = base64::engine::general_purpose::STANDARD.encode([1u8, 0, 255, 255]);
        let msg = json!({ "audio": b64, "context_id": "ctx-1" });
        match decode_message(Some(&msg), None) {
            Decoded::Audio(pcm) => assert_eq!(pcm, vec![1, -1]),
            _ => panic!("expected Audio"),
        }
    }

    #[test]
    fn decode_final_and_ignore() {
        assert!(matches!(
            decode_message(Some(&json!({ "final": true, "context_id": "ctx-1" })), None),
            Decoded::Done
        ));
        // No audio / final → ignore (no panic).
        assert!(matches!(
            decode_message(Some(&json!({ "status": "ok" })), None),
            Decoded::Ignore
        ));
        assert!(matches!(
            decode_message(Some(&json!("nope")), None),
            Decoded::Ignore
        ));
    }

    /// Live smoke (requires `ASYNCAI_API_KEY` + `ASYNCAI_VOICE_ID`). Run:
    /// `ASYNCAI_API_KEY=… ASYNCAI_VOICE_ID=… cargo test -p flowcat-services --features tts-asyncai -- --ignored asyncai_live`
    #[tokio::test]
    #[ignore = "requires ASYNCAI_API_KEY + ASYNCAI_VOICE_ID"]
    async fn asyncai_live_synthesizes_audio() {
        let key = std::env::var("ASYNCAI_API_KEY").expect("ASYNCAI_API_KEY");
        let voice = std::env::var("ASYNCAI_VOICE_ID").expect("ASYNCAI_VOICE_ID");
        let mut tts = AsyncAiTts::new(key, voice);
        tts.start(&StartParams::default()).await.expect("connect");
        let frames = tts.run_tts("Hello from flowcat.").await.expect("run_tts");
        let audio_chunks = frames
            .iter()
            .filter(|f| matches!(f, Frame::TtsAudio { .. }))
            .count();
        assert!(audio_chunks > 0, "expected at least one TtsAudio chunk");
    }
}
