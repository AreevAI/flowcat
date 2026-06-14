// SPDX-License-Identifier: Apache-2.0
//
//! **Gradium** streaming STT.
//!
//! A **(D)istinct** streaming-WebSocket client. Gradium is **not** present in the
//! vendored pipecat sources, so (unlike the other providers in this group) there
//! is no upstream class to cross-check against. This impl therefore targets
//! Gradium's documented realtime shape: a single WSS endpoint authenticated with
//! `Authorization: Bearer <api-key>`, raw little-endian PCM streamed as **binary**
//! frames, and bare-JSON transcript messages:
//!
//! ```json
//! { "type": "transcript", "text": "book a dentist", "is_final": true, "language": "en" }
//! ```
//!
//! `is_final` true → final [`Frame::Transcription`]; otherwise interim. Any
//! non-transcript / empty / malformed message yields nothing. The transport is
//! the shared [`ws_stt`] seam; only the bare-JSON [`decode_message`] is
//! Gradium-specific. The host/encoding are the documented defaults and the
//! decode is the contract the wire-fixture tests pin — adjust the base URL via
//! [`GradiumStt::base_url`] if Gradium's endpoint differs in a live deployment.

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

/// Gradium's realtime STT WSS host. The query string (encoding/sample_rate) is
/// appended at connect time; the host is fixed — only the API key (header) and
/// validated numeric params are caller-controlled (no SSRF surface).
pub const GRADIUM_WSS_BASE: &str = "wss://api.gradium.ai/v1/stt/stream";

/// Gradium streaming-STT session.
pub struct GradiumStt {
    api_key: String,
    sample_rate: u32,
    base_url: String,
    session: Option<WsSttSession>,
    muted: bool,
}

impl GradiumStt {
    /// Construct bound to `api_key` (default 16 kHz input).
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            sample_rate: 16_000,
            base_url: GRADIUM_WSS_BASE.to_string(),
            muted: false,
            session: None,
        }
    }

    /// Override the input sample rate (default 16 kHz).
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Override the WSS base URL (for a non-default Gradium deployment).
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// The connect URL for this config (testable without a socket). The API key
    /// is **never** placed in the URL (it travels in `Authorization`).
    pub(crate) fn url(&self) -> String {
        format!(
            "{}?encoding=pcm_s16le&sample_rate={}",
            self.base_url, self.sample_rate
        )
    }
}

#[async_trait]
impl SttService for GradiumStt {
    fn name(&self) -> &str {
        "gradium"
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        let cfg = WsSttConfig {
            url: self.url(),
            headers: vec![(
                "Authorization".to_string(),
                format!("Bearer {}", self.api_key),
            )],
            init_message: None,
            decode: decode_message,
        };
        self.session = Some(WsSttSession::connect(cfg).await?);
        Ok(())
    }

    async fn run_stt(&mut self, audio: Arc<AudioFrame>) -> Result<Vec<Frame>> {
        if self.muted {
            return Ok(vec![]);
        }
        let session = ws_stt::require(&mut self.session, "gradium")?;
        session.send_pcm_binary(&audio).await?;
        Ok(session.drain())
    }

    async fn set_muted(&mut self, muted: bool) {
        self.muted = muted;
    }
}

/// Decode one Gradium server message. **Pure.** `transcript` with non-empty
/// `text` → final/interim by `is_final`; anything else → nothing.
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
        let c = GradiumStt::new("secret").sample_rate(8000);
        let u = c.url();
        assert!(u.starts_with("wss://api.gradium.ai/v1/stt/stream?encoding=pcm_s16le"));
        assert!(u.contains("sample_rate=8000"));
        assert!(!u.contains("secret"));
    }

    #[test]
    fn decode_final_and_interim() {
        let f = json!({ "type": "transcript", "text": "book a dentist", "is_final": true, "language": "en" });
        assert!(matches!(&decode_message(&f)[..],
            [Frame::Transcription { text, final_, .. }] if text == "book a dentist" && *final_));
        let i = json!({ "type": "transcript", "text": "book a", "is_final": false });
        assert!(matches!(
            decode_message(&i).as_slice(),
            [Frame::InterimTranscription { .. }]
        ));
    }

    #[test]
    fn decode_ignores_other_empty_and_malformed() {
        assert!(decode_message(&json!({ "type": "ready" })).is_empty());
        assert!(decode_message(&json!({ "type": "transcript", "text": "" })).is_empty());
        assert!(decode_message(&json!({ "text": "no type" })).is_empty());
        assert!(decode_message(&json!("nope")).is_empty());
    }

    /// Live smoke (requires `GRADIUM_API_KEY`). Run:
    /// `GRADIUM_API_KEY=… cargo test -p flowcat-services --features stt-gradium -- --ignored gradium_live`
    #[tokio::test]
    #[ignore = "requires GRADIUM_API_KEY"]
    async fn gradium_live_connects_and_streams() {
        let key = std::env::var("GRADIUM_API_KEY").expect("GRADIUM_API_KEY");
        let mut stt = GradiumStt::new(key);
        stt.start(&StartParams::default()).await.expect("connect");
        let silence = Arc::new(AudioFrame::mono(vec![0i16; 1600], 16_000));
        let _ = stt.run_stt(silence).await.expect("run_stt");
    }
}
