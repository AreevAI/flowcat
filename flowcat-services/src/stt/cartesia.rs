// SPDX-License-Identifier: Apache-2.0
//
//! **Cartesia** streaming STT (Ink-Whisper live).
//!
//! A **(D)istinct** streaming-WebSocket client (cross-checked against pipecat
//! `services/cartesia/stt.py`). Connects to
//! `wss://api.cartesia.ai/stt/websocket?model=…&language=…&encoding=pcm_s16le&sample_rate=…`
//! with the `Cartesia-Version: 2025-04-16` and `X-API-Key: <api-key>` headers,
//! streams raw little-endian PCM as **binary** frames, and reads JSON results:
//!
//! ```json
//! { "type": "transcript", "text": "book a dentist", "is_final": true, "language": "en" }
//! { "type": "error", "message": "…" }
//! ```
//!
//! `is_final` true → final [`Frame::Transcription`]; otherwise interim. `error`
//! and any non-transcript message yield nothing. The control words `"finalize"`
//! (flush) and `"done"` (close) are sent as **text** frames. Decode is **pure**.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use flowcat_core::error::Result;
use flowcat_core::processor::frame::{AudioFrame, Frame, Language, StartParams};
use flowcat_core::service::SttService;

#[allow(clippy::duplicate_mod)] // each WS provider owns its own copy (feature-independent)
#[path = "ws_stt_common.rs"]
pub mod ws_stt;

use ws_stt::{WsSttConfig, WsSttSession};

/// Cartesia's fixed STT WSS host. The query string (model/language/encoding/rate)
/// is appended at connect time; the **host is fixed** — only the API key (header)
/// and validated params are caller-controlled (no SSRF surface).
pub const CARTESIA_WSS_BASE: &str = "wss://api.cartesia.ai/stt/websocket";
/// The pinned Cartesia API version header value.
pub const CARTESIA_VERSION: &str = "2025-04-16";

/// Cartesia streaming-STT session (Ink-Whisper live).
pub struct CartesiaStt {
    api_key: String,
    sample_rate: u32,
    model: String,
    language: String,
    session: Option<WsSttSession>,
    muted: bool,
}

impl CartesiaStt {
    /// Construct bound to `api_key` (default 16 kHz, `ink-whisper`, English).
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            sample_rate: 16_000,
            model: "ink-whisper".to_string(),
            language: "en".to_string(),
            muted: false,
            session: None,
        }
    }

    /// Override the input sample rate (default 16 kHz).
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Override the model (default `ink-whisper`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the language (default `en`).
    pub fn language(mut self, lang: impl Into<String>) -> Self {
        self.language = lang.into();
        self
    }

    /// The connect URL for this config (testable without a socket). The API key
    /// is **never** placed in the URL (it travels in `X-API-Key`).
    pub(crate) fn url(&self) -> String {
        format!(
            "{CARTESIA_WSS_BASE}?model={}&language={}&encoding=pcm_s16le&sample_rate={}",
            self.model, self.language, self.sample_rate
        )
    }
}

#[async_trait]
impl SttService for CartesiaStt {
    fn name(&self) -> &str {
        "cartesia"
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        let cfg = WsSttConfig {
            url: self.url(),
            headers: vec![
                ("Cartesia-Version".to_string(), CARTESIA_VERSION.to_string()),
                ("X-API-Key".to_string(), self.api_key.clone()),
            ],
            init_message: None,
            decode: decode_message,
        };
        // Lazy connect: open the WS on first audio, NOT here (an eager connect stalls
        // the pipeline Start handshake — no greeting, no audio).
        self.session = Some(WsSttSession::lazy(cfg));
        Ok(())
    }

    async fn run_stt(&mut self, audio: Arc<AudioFrame>) -> Result<Vec<Frame>> {
        let muted = self.muted;
        let session = ws_stt::require(&mut self.session, "cartesia")?;
        // While muted, feed SILENCE (not the mic) to keep the socket warm without
        // transcribing the bot's echo; decoded frames are dropped while muted.
        if muted {
            let silence = AudioFrame::mono(vec![0i16; audio.pcm.len()], audio.sample_rate);
            let _ = session.send_pcm_binary(&silence).await;
            session.drain();
            return Ok(vec![]);
        }
        // Cartesia streams raw little-endian PCM as binary frames.
        session.send_pcm_binary(&audio).await?;
        Ok(session.drain())
    }

    async fn set_muted(&mut self, muted: bool) {
        self.muted = muted;
    }
}

/// Decode one Cartesia server message. **Pure.** `transcript` with non-empty
/// `text` → final/interim by `is_final`; `error`/other → nothing.
pub(crate) fn decode_message(value: &Value) -> Vec<Frame> {
    if value.get("type").and_then(|t| t.as_str()) != Some("transcript") {
        return vec![];
    }
    let transcript = value.get("text").and_then(|t| t.as_str()).unwrap_or("");
    if transcript.is_empty() {
        return vec![];
    }
    let is_final = value
        .get("is_final")
        .and_then(|f| f.as_bool())
        .unwrap_or(false);
    let language = value
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
    fn url_encodes_params_and_omits_key() {
        let c = CartesiaStt::new("secret")
            .sample_rate(8000)
            .model("ink-whisper");
        let u = c.url();
        assert!(u.starts_with("wss://api.cartesia.ai/stt/websocket?model=ink-whisper"));
        assert!(u.contains("encoding=pcm_s16le"));
        assert!(u.contains("sample_rate=8000"));
        assert!(!u.contains("secret"));
    }

    #[test]
    fn decode_final_transcript() {
        let msg = json!({ "type": "transcript", "text": "book a dentist", "is_final": true, "language": "en" });
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
            other => panic!("expected final, got {other:?}"),
        }
    }

    #[test]
    fn decode_partial_is_interim() {
        let msg = json!({ "type": "transcript", "text": "book a", "is_final": false });
        assert!(matches!(
            decode_message(&msg).as_slice(),
            [Frame::InterimTranscription { .. }]
        ));
    }

    #[test]
    fn decode_ignores_error_empty_and_malformed() {
        assert!(decode_message(&json!({ "type": "error", "message": "bad" })).is_empty());
        assert!(
            decode_message(&json!({ "type": "transcript", "text": "", "is_final": true }))
                .is_empty()
        );
        assert!(decode_message(&json!({ "type": "transcript" })).is_empty());
        assert!(decode_message(&json!({ "text": "no type" })).is_empty());
        assert!(decode_message(&json!("nope")).is_empty());
    }

    /// Live smoke (requires `CARTESIA_API_KEY`). Run:
    /// `CARTESIA_API_KEY=… cargo test -p flowcat-services --features stt-cartesia -- --ignored cartesia_live`
    #[tokio::test]
    #[ignore = "requires CARTESIA_API_KEY"]
    async fn cartesia_live_connects_and_streams() {
        let key = std::env::var("CARTESIA_API_KEY").expect("CARTESIA_API_KEY");
        let mut stt = CartesiaStt::new(key);
        stt.start(&StartParams::default()).await.expect("connect");
        let silence = Arc::new(AudioFrame::mono(vec![0i16; 1600], 16_000));
        let _ = stt.run_stt(silence).await.expect("run_stt");
    }
}
