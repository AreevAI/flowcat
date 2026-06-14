// SPDX-License-Identifier: Apache-2.0
//
//! **Soniox** streaming STT (token-stream protocol).
//!
//! A **(D)istinct** streaming-WebSocket client (cross-checked against pipecat
//! `services/soniox/stt.py`). Connects to the fixed
//! `wss://stt-rt.soniox.com/transcribe-websocket`, sends a JSON **config** message
//! first (the API key travels in this body, not a header), then streams raw
//! little-endian PCM as **binary** frames. Server messages carry a token stream:
//!
//! ```json
//! { "tokens": [ {"text":"book","is_final":true}, {"text":" a","is_final":true},
//!               {"text":"<end>","is_final":true} ] }
//! ```
//!
//! Each token has `text` + `is_final`. A special `<end>` token marks an endpoint:
//! the buffered final tokens up to it form one final [`Frame::Transcription`].
//! Non-endpoint final + non-final tokens render an [`Frame::InterimTranscription`]
//! of the running text. `<fin>` (a keep-alive finalize marker) and empty token
//! lists yield nothing. The decode is **pure per message** (final text up to the
//! `<end>` → Transcription; otherwise the joined text → interim).

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use flowcat_core::error::Result;
use flowcat_core::processor::frame::{AudioFrame, Frame, StartParams};
use flowcat_core::service::SttService;

#[allow(clippy::duplicate_mod)] // each WS provider owns its own copy (feature-independent)
#[path = "ws_stt_common.rs"]
pub mod ws_stt;

use ws_stt::{WsSttConfig, WsSttSession};

/// Soniox's fixed real-time transcription WSS. The **host is fixed**; the API key
/// is sent only in the first JSON config message, never in the URL.
pub const SONIOX_WSS: &str = "wss://stt-rt.soniox.com/transcribe-websocket";

/// Soniox's endpoint (turn-boundary) control token.
const END_TOKEN: &str = "<end>";
/// Soniox's finalize keep-alive marker token (ignored).
const FINALIZED_TOKEN: &str = "<fin>";

/// Soniox streaming-STT session.
pub struct SonioxStt {
    api_key: String,
    sample_rate: u32,
    model: String,
    session: Option<WsSttSession>,
    muted: bool,
}

impl SonioxStt {
    /// Construct bound to `api_key` (default 16 kHz input, `stt-rt-v4` model).
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            sample_rate: 16_000,
            model: "stt-rt-v4".to_string(),
            muted: false,
            session: None,
        }
    }

    /// Override the input sample rate (default 16 kHz).
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Override the transcription model (default `stt-rt-v4`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// The first-message config (carries the API key + audio shape). Testable
    /// without a socket.
    pub(crate) fn config_message(&self) -> Value {
        json!({
            "api_key": self.api_key,
            "model": self.model,
            "audio_format": "pcm_s16le",
            "num_channels": 1,
            "sample_rate": self.sample_rate,
            "enable_endpoint_detection": true,
        })
    }
}

#[async_trait]
impl SttService for SonioxStt {
    fn name(&self) -> &str {
        "soniox"
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        let cfg = WsSttConfig {
            url: SONIOX_WSS.to_string(),
            headers: vec![],
            init_message: Some(self.config_message().to_string()),
            decode: decode_message,
        };
        // Lazy connect: open the WS (sending the config init message) on first audio,
        // NOT here — an eager connect stalls the pipeline Start handshake.
        self.session = Some(WsSttSession::lazy(cfg));
        Ok(())
    }

    async fn run_stt(&mut self, audio: Arc<AudioFrame>) -> Result<Vec<Frame>> {
        let muted = self.muted;
        let session = ws_stt::require(&mut self.session, "soniox")?;
        // While muted, feed SILENCE (not the mic) to keep the socket warm without
        // transcribing the bot's echo; decoded frames are dropped while muted.
        if muted {
            let silence = AudioFrame::mono(vec![0i16; audio.pcm.len()], audio.sample_rate);
            let _ = session.send_pcm_binary(&silence).await;
            session.drain();
            return Ok(vec![]);
        }
        // Soniox streams raw little-endian PCM as binary frames.
        session.send_pcm_binary(&audio).await?;
        Ok(session.drain())
    }

    async fn set_muted(&mut self, muted: bool) {
        self.muted = muted;
    }
}

fn token_text(t: &Value) -> &str {
    t.get("text").and_then(|x| x.as_str()).unwrap_or("")
}
fn token_is_final(t: &Value) -> bool {
    t.get("is_final").and_then(|x| x.as_bool()).unwrap_or(false)
}

/// Decode one Soniox token message into transcription frames. **Pure.** Final
/// tokens up to an `<end>` token become one final [`Frame::Transcription`]; the
/// remaining joined text (final-not-yet-ended + non-final) becomes an interim.
pub(crate) fn decode_message(value: &Value) -> Vec<Frame> {
    let Some(tokens) = value.get("tokens").and_then(|t| t.as_array()) else {
        return vec![];
    };
    // A lone <fin> keep-alive token carries no transcript.
    if tokens.len() == 1 && token_text(&tokens[0]) == FINALIZED_TOKEN {
        return vec![];
    }
    let user_id: Arc<str> = Arc::from("user");
    let mut out = Vec::new();
    let mut final_buf = String::new();
    let mut interim_buf = String::new();
    for t in tokens {
        let text = token_text(t);
        if token_is_final(t) {
            if text == END_TOKEN {
                // Endpoint: flush the buffered final text as one Transcription.
                if !final_buf.is_empty() {
                    out.push(Frame::Transcription {
                        text: std::mem::take(&mut final_buf),
                        user_id: user_id.clone(),
                        language: None,
                        final_: true,
                    });
                }
            } else if text != FINALIZED_TOKEN {
                final_buf.push_str(text);
            }
        } else {
            interim_buf.push_str(text);
        }
    }
    // Anything still un-ended (final-but-no-endpoint + non-final) is interim.
    let running = format!("{final_buf}{interim_buf}");
    if !running.is_empty() {
        out.push(Frame::InterimTranscription {
            text: running,
            user_id,
            language: None,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn config_carries_key_and_audio_shape() {
        let c = SonioxStt::new("secret").sample_rate(8000).config_message();
        assert_eq!(c["api_key"], "secret");
        assert_eq!(c["audio_format"], "pcm_s16le");
        assert_eq!(c["sample_rate"], 8000);
        assert_eq!(c["enable_endpoint_detection"], true);
        // The connect URL is the fixed host (no key in it).
        assert_eq!(SONIOX_WSS, "wss://stt-rt.soniox.com/transcribe-websocket");
    }

    #[test]
    fn decode_endpoint_emits_final_then_interim_of_rest() {
        let msg = json!({ "tokens": [
            { "text": "book", "is_final": true },
            { "text": " a dentist", "is_final": true },
            { "text": "<end>", "is_final": true },
            { "text": " tomorrow", "is_final": false }
        ]});
        let frames = decode_message(&msg);
        // First: final "book a dentist"; then interim " tomorrow".
        assert!(
            matches!(&frames[0], Frame::Transcription { text, final_, .. }
            if text == "book a dentist" && *final_)
        );
        assert!(
            matches!(&frames[1], Frame::InterimTranscription { text, .. }
            if text == " tomorrow")
        );
    }

    #[test]
    fn decode_no_endpoint_is_interim_only() {
        let msg = json!({ "tokens": [
            { "text": "book a", "is_final": true },
            { "text": " den", "is_final": false }
        ]});
        match &decode_message(&msg)[..] {
            [Frame::InterimTranscription { text, .. }] => assert_eq!(text, "book a den"),
            other => panic!("expected one interim, got {other:?}"),
        }
    }

    #[test]
    fn decode_ignores_fin_keepalive_empty_and_malformed() {
        // Lone <fin> keep-alive → nothing.
        assert!(
            decode_message(&json!({ "tokens": [{ "text": "<fin>", "is_final": true }] }))
                .is_empty()
        );
        // Empty token list → nothing.
        assert!(decode_message(&json!({ "tokens": [] })).is_empty());
        // Missing tokens / not an object → no panic, nothing.
        assert!(decode_message(&json!({ "error_code": 401 })).is_empty());
        assert!(decode_message(&json!("nope")).is_empty());
        // <end> with no preceding final text → nothing.
        assert!(
            decode_message(&json!({ "tokens": [{ "text": "<end>", "is_final": true }] }))
                .is_empty()
        );
    }

    /// Live smoke (requires `SONIOX_API_KEY`). Run:
    /// `SONIOX_API_KEY=… cargo test -p flowcat-services --features stt-soniox -- --ignored soniox_live`
    #[tokio::test]
    #[ignore = "requires SONIOX_API_KEY"]
    async fn soniox_live_connects_and_streams() {
        let key = std::env::var("SONIOX_API_KEY").expect("SONIOX_API_KEY");
        let mut stt = SonioxStt::new(key);
        stt.start(&StartParams::default()).await.expect("connect");
        let silence = Arc::new(AudioFrame::mono(vec![0i16; 1600], 16_000));
        let _ = stt.run_stt(silence).await.expect("run_stt");
    }
}
