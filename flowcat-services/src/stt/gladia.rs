// SPDX-License-Identifier: Apache-2.0
//
//! **Gladia** streaming STT (v2 live, init-then-stream).
//!
//! A **(D)istinct** two-step client (cross-checked against pipecat
//! `services/gladia/stt.py`):
//!
//! 1. **Init.** POST the audio settings to `https://api.gladia.io/v2/live` with an
//!    `X-Gladia-Key: <api-key>` header; the JSON response carries a single-use
//!    session `url` (the `wss://…` to stream into).
//! 2. **Stream.** Connect that session URL, send each PCM chunk as
//!    `{"type":"audio_chunk","data":{"chunk":"<base64>"}}`, and read transcript
//!    messages: `{"type":"transcript","data":{"is_final":bool,"utterance":{"text":…,"language":…}}}`.
//!
//! `is_final` true → final [`Frame::Transcription`]; otherwise
//! [`Frame::InterimTranscription`]. Acknowledgements / non-transcript messages
//! yield nothing. The WS transport is the shared [`ws_stt`] seam; only the init
//! handshake (which mints the session URL) is Gladia-specific.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use flowcat_core::error::{FlowcatError, Result};
use flowcat_core::processor::frame::{AudioFrame, Frame, Language, StartParams};
use flowcat_core::service::SttService;

#[allow(clippy::duplicate_mod)] // each WS provider owns its own copy (feature-independent)
#[path = "ws_stt_common.rs"]
pub mod ws_stt;

use ws_stt::{WsSttConfig, WsSttSession};

/// Gladia's fixed v2-live **init** endpoint (REST). The session WSS URL is minted
/// by the server in the init response — the API key only ever travels in the
/// `X-Gladia-Key` header, never in a URL.
pub const GLADIA_LIVE_INIT_URL: &str = "https://api.gladia.io/v2/live";

/// Gladia streaming-STT session.
pub struct GladiaStt {
    api_key: String,
    sample_rate: u32,
    session: Option<WsSttSession>,
    muted: bool,
}

impl GladiaStt {
    /// Construct bound to `api_key` (default 16 kHz input).
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            sample_rate: 16_000,
            muted: false,
            session: None,
        }
    }

    /// Override the input sample rate (default 16 kHz).
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// The init request body (PCM/wav settings). Testable without a socket.
    pub(crate) fn init_settings(&self) -> Value {
        json!({
            "encoding": "wav/pcm",
            "bit_depth": 16,
            "sample_rate": self.sample_rate,
            "channels": 1,
            // Pin English: without this Gladia auto-detects and mis-IDs short/clear
            // English audio as other languages → gibberish low-confidence transcripts.
            "language_config": { "languages": ["en"], "code_switching": false },
        })
    }

    /// POST the init settings and return the session WSS URL Gladia mints.
    async fn init_session(&self) -> Result<String> {
        let client = reqwest::Client::new();
        let resp = client
            .post(GLADIA_LIVE_INIT_URL)
            .header("X-Gladia-Key", &self.api_key)
            .json(&self.init_settings())
            .send()
            .await
            .map_err(|e| FlowcatError::Network(format!("gladia init: {e}")))?;
        if !resp.status().is_success() {
            let code = resp.status();
            return Err(FlowcatError::Network(format!(
                "gladia init failed: HTTP {code}"
            )));
        }
        let body: Value = resp
            .json()
            .await
            .map_err(|e| FlowcatError::Network(format!("gladia init body: {e}")))?;
        body.get("url")
            .and_then(|u| u.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| FlowcatError::Network("gladia init: missing session url".into()))
    }
}

#[async_trait]
impl SttService for GladiaStt {
    fn name(&self) -> &str {
        "gladia"
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        // Lazy: defer the init POST + WS connect to first audio (an eager connect in
        // start() stalls the pipeline Start handshake — no greeting, no audio).
        Ok(())
    }

    async fn run_stt(&mut self, audio: Arc<AudioFrame>) -> Result<Vec<Frame>> {
        let muted = self.muted;
        // Lazy (re)connect: Gladia mints a SINGLE-USE session URL via an init POST, so
        // we (re)establish it here on first use / after a drop — never in start(), and
        // never via the shared reconnect (a used URL can't be redialed).
        if self.session.is_none() {
            let url = self.init_session().await?;
            let cfg = WsSttConfig {
                url,
                headers: vec![],
                init_message: None,
                decode: decode_message,
            };
            self.session = Some(WsSttSession::connect(cfg).await?);
        }
        let session = ws_stt::require(&mut self.session, "gladia")?;
        // Gladia wants base64 PCM in a JSON audio_chunk envelope. While muted, send
        // SILENCE so the session stays warm without transcribing the bot's echo.
        let bytes = if muted {
            vec![0u8; audio.pcm.len() * 2] // i16-LE silence, same length
        } else {
            ws_stt::pcm_le_bytes(&audio)
        };
        let chunk = ws_stt::base64_encode(&bytes);
        let msg = json!({ "type": "audio_chunk", "data": { "chunk": chunk } }).to_string();
        let send = session
            .send_message(tokio_tungstenite::tungstenite::Message::text(msg))
            .await;
        let frames = session.drain();
        match send {
            Ok(()) => Ok(if muted { vec![] } else { frames }),
            // Dead session (single-use URL): drop it so the next chunk re-inits.
            Err(_) => {
                self.session = None;
                Ok(vec![])
            }
        }
    }

    async fn set_muted(&mut self, muted: bool) {
        self.muted = muted;
    }
}

/// Decode one Gladia server message into transcription frames. **Pure.** Only
/// `transcript` messages with a non-empty utterance produce a frame; ack /
/// non-transcript messages → nothing.
pub(crate) fn decode_message(value: &Value) -> Vec<Frame> {
    if value.get("type").and_then(|t| t.as_str()) != Some("transcript") {
        return vec![];
    }
    let data = match value.get("data") {
        Some(d) => d,
        None => return vec![],
    };
    let utterance = match data.get("utterance") {
        Some(u) => u,
        None => return vec![],
    };
    let transcript = utterance.get("text").and_then(|t| t.as_str()).unwrap_or("");
    if transcript.is_empty() {
        return vec![];
    }
    let is_final = data
        .get("is_final")
        .and_then(|f| f.as_bool())
        .unwrap_or(false);
    let language = utterance
        .get("language")
        .and_then(|l| l.as_str())
        .map(|l| Language(l.to_string()));
    let user_id: Arc<str> = Arc::from("user");
    if is_final {
        vec![Frame::Transcription {
            text: transcript.to_string(),
            user_id,
            language,
            final_: true,
        }]
    } else {
        vec![Frame::InterimTranscription {
            text: transcript.to_string(),
            user_id,
            language,
        }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn init_settings_encode_the_pcm_wav_config() {
        let s = GladiaStt::new("k").sample_rate(8000).init_settings();
        assert_eq!(s["encoding"], "wav/pcm");
        assert_eq!(s["bit_depth"], 16);
        assert_eq!(s["sample_rate"], 8000);
        assert_eq!(s["channels"], 1);
    }

    #[test]
    fn decode_final_transcript() {
        let msg = json!({
            "type": "transcript",
            "data": {
                "is_final": true,
                "utterance": { "text": "book a dentist", "language": "en" }
            }
        });
        match &decode_message(&msg)[..] {
            [Frame::Transcription {
                text,
                final_,
                language,
                ..
            }] => {
                assert_eq!(text, "book a dentist");
                assert!(final_);
                assert_eq!(language.as_ref().map(|l| l.0.as_str()), Some("en"));
            }
            other => panic!("expected final Transcription, got {other:?}"),
        }
    }

    #[test]
    fn decode_partial_transcript_is_interim() {
        let msg = json!({
            "type": "transcript",
            "data": { "is_final": false, "utterance": { "text": "book a", "language": "en" } }
        });
        assert!(matches!(
            decode_message(&msg).as_slice(),
            [Frame::InterimTranscription { .. }]
        ));
    }

    #[test]
    fn decode_ignores_acks_empty_and_malformed() {
        // audio_chunk acknowledgement → nothing.
        assert!(decode_message(&json!({
            "type": "audio_chunk", "acknowledged": true, "data": { "byte_range": [0, 100] }
        }))
        .is_empty());
        // Empty utterance text → nothing.
        assert!(decode_message(&json!({
            "type": "transcript", "data": { "is_final": true, "utterance": { "text": "" } }
        }))
        .is_empty());
        // Missing data / not an object → no panic, nothing.
        assert!(decode_message(&json!({ "type": "transcript" })).is_empty());
        assert!(decode_message(&json!({ "type": "other" })).is_empty());
        assert!(decode_message(&json!("nope")).is_empty());
    }

    /// Live smoke (requires `GLADIA_API_KEY`): init + connect + send silence. Run:
    /// `GLADIA_API_KEY=… cargo test -p flowcat-services --features stt-gladia -- --ignored gladia_live`
    #[tokio::test]
    #[ignore = "requires GLADIA_API_KEY"]
    async fn gladia_live_connects_and_streams() {
        let key = std::env::var("GLADIA_API_KEY").expect("GLADIA_API_KEY");
        let mut stt = GladiaStt::new(key);
        stt.start(&StartParams::default()).await.expect("connect");
        let silence = Arc::new(AudioFrame::mono(vec![0i16; 1600], 16_000));
        let _ = stt.run_stt(silence).await.expect("run_stt");
    }
}
