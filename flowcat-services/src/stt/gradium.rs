// SPDX-License-Identifier: Apache-2.0
//
//! **Gradium** streaming STT.
//!
//! A **(D)istinct** streaming-WebSocket client for Gradium's realtime ASR
//! (<https://docs.gradium.ai/api-reference/endpoint/stt-websocket>):
//! connect to `wss://api.gradium.ai/api/speech/asr` with an `x-api-key` header,
//! send a JSON **setup** message (model + input PCM format), then stream each
//! audio chunk base64-encoded inside a JSON envelope
//! `{ "type": "audio", "audio": "<base64 pcm_s16le>" }`.
//!
//! The server returns transcribed segments as
//! `{ "type": "text", "text": "hello world", "start_s": 0.5, "stream_id": 0 }`.
//! Gradium streams these **per word/segment** and has **no per-message end-of-turn
//! flag** (turn-taking is client-driven from VAD). Emitting one final transcript per
//! `text` would split a single utterance into many turns, so instead each `text` is
//! carried as an interim, accumulated, and the connector emits **one** final
//! [`Frame::Transcription`] when transcription goes quiet for [`TURN_GAP_MS`] (the
//! end-of-utterance boundary). The audio-chunk cadence is the logical clock — no VAD
//! parsing or flush round-trip is needed because the segment text has already
//! arrived. The transport is the shared [`ws_stt`] seam; the setup/audio encode and
//! the bare-JSON [`decode_message`] are the Gradium-specific seams.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use flowcat_core::error::Result;
use flowcat_core::processor::frame::{AudioFrame, Frame, StartParams};
use flowcat_core::service::SttService;

#[allow(clippy::duplicate_mod)] // each WS provider owns its own copy (feature-independent)
#[path = "ws_stt_common.rs"]
pub mod ws_stt;

use ws_stt::{base64_encode, pcm_le_bytes, WsSttConfig, WsSttSession};

/// Gradium's realtime STT WSS endpoint. The API key travels in the `x-api-key`
/// header, never the URL — so the host is fixed (no SSRF surface).
pub const GRADIUM_WSS: &str = "wss://api.gradium.ai/api/speech/asr";

/// End-of-utterance gap: once a segment has arrived, this much further audio with
/// **no** new `text` closes the user turn (one final transcript).
pub const TURN_GAP_MS: u64 = 700;

/// Gradium streaming-STT session.
pub struct GradiumStt {
    api_key: String,
    sample_rate: u32,
    url: String,
    session: Option<WsSttSession>,
    muted: bool,
    /// Accumulated text of the in-progress user turn (joined `text` segments).
    turn_text: String,
    /// Audio duration (ms) elapsed since the last new `text` segment — the
    /// end-of-utterance clock, advanced by the incoming chunk cadence.
    quiet_ms: u64,
}

impl GradiumStt {
    /// Construct bound to `api_key` (default 16 kHz input — the cascaded carrier rate).
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            sample_rate: 16_000,
            url: GRADIUM_WSS.to_string(),
            muted: false,
            session: None,
            turn_text: String::new(),
            quiet_ms: 0,
        }
    }

    /// Override the input sample rate (default 16 kHz). Sent to Gradium as the
    /// `input_format` (`pcm_<rate>`), so it must match the PCM actually streamed.
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Override the WSS URL (for a non-default Gradium deployment).
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.url = url.into();
        self
    }

    /// The Gradium `setup` handshake (sent once, immediately after connect): the
    /// model and the explicit input PCM rate (`pcm_16000` for the 16 kHz carrier).
    fn setup_message(&self) -> String {
        json!({
            "type": "setup",
            "model_name": "default",
            "input_format": format!("pcm_{}", self.sample_rate),
            "json_config": { "language": "en" },
        })
        .to_string()
    }
}

#[async_trait]
impl SttService for GradiumStt {
    fn name(&self) -> &str {
        "gradium"
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        let cfg = WsSttConfig {
            url: self.url.clone(),
            headers: vec![("x-api-key".to_string(), self.api_key.clone())],
            init_message: Some(self.setup_message()),
            decode: decode_message,
        };
        // Lazy: connect (and send `setup`) on the first audio chunk, so the socket
        // connect never stalls the pipeline Start handshake.
        self.session = Some(WsSttSession::lazy(cfg));
        Ok(())
    }

    async fn run_stt(&mut self, audio: Arc<AudioFrame>) -> Result<Vec<Frame>> {
        if self.muted {
            return Ok(vec![]);
        }
        let chunk_ms = if audio.sample_rate > 0 {
            (audio.pcm.len() as u64 * 1000) / audio.sample_rate as u64
        } else {
            0
        };
        let session = ws_stt::require(&mut self.session, "gradium")?;
        // Gradium takes audio base64-encoded inside a JSON envelope, not raw binary.
        let envelope = json!({
            "type": "audio",
            "audio": base64_encode(&pcm_le_bytes(&audio)),
        })
        .to_string();
        session.send_text(envelope).await?;

        // Accrue any new `text` segments (carried as interims by `decode_message`)
        // into the current turn; reset the quiet clock when something arrives.
        let mut got_text = false;
        for frame in session.drain() {
            if let Frame::InterimTranscription { text, .. } = frame {
                let seg = text.trim();
                if !seg.is_empty() {
                    if !self.turn_text.is_empty() {
                        self.turn_text.push(' ');
                    }
                    self.turn_text.push_str(seg);
                    got_text = true;
                }
            }
        }
        if got_text {
            self.quiet_ms = 0;
        } else {
            self.quiet_ms = self.quiet_ms.saturating_add(chunk_ms);
        }

        // End of utterance: a segment exists and transcription has gone quiet long
        // enough — emit exactly one final transcript (closes the user turn).
        if !self.turn_text.is_empty() && self.quiet_ms >= TURN_GAP_MS {
            let text = std::mem::take(&mut self.turn_text);
            self.quiet_ms = 0;
            return Ok(vec![Frame::Transcription {
                text,
                user_id: Arc::from("user"),
                language: None,
                final_: true,
            }]);
        }
        Ok(vec![])
    }

    async fn set_muted(&mut self, muted: bool) {
        // A new turn (mute = bot speaking / turn just closed): drop any half-turn
        // remainder so the next user turn starts clean.
        if muted {
            self.turn_text.clear();
            self.quiet_ms = 0;
        }
        self.muted = muted;
    }
}

/// Decode one Gradium server message. **Pure.** A `text` segment with non-empty
/// `text` → one [`Frame::InterimTranscription`] (the connector accumulates these and
/// emits the turn-final itself); `end_text`/`ready`/VAD `step`/empty/malformed →
/// nothing. Gradium has no per-message end-of-turn flag.
pub(crate) fn decode_message(value: &Value) -> Vec<Frame> {
    if value.get("type").and_then(|t| t.as_str()) != Some("text") {
        return vec![];
    }
    let text = value.get("text").and_then(|t| t.as_str()).unwrap_or("");
    if text.trim().is_empty() {
        return vec![];
    }
    vec![Frame::InterimTranscription {
        text: text.to_string(),
        user_id: Arc::from("user"),
        language: None,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn connects_to_the_asr_endpoint() {
        assert_eq!(GRADIUM_WSS, "wss://api.gradium.ai/api/speech/asr");
        let c = GradiumStt::new("secret");
        assert_eq!(c.url, "wss://api.gradium.ai/api/speech/asr");
        assert!(!c.url.contains("secret"));
    }

    #[test]
    fn setup_carries_model_and_pcm_rate() {
        let setup = GradiumStt::new("k").sample_rate(16_000).setup_message();
        let v: Value = serde_json::from_str(&setup).unwrap();
        assert_eq!(v["type"], "setup");
        assert_eq!(v["model_name"], "default");
        assert_eq!(v["input_format"], "pcm_16000");
        assert_eq!(v["json_config"]["language"], "en");
    }

    #[test]
    fn decode_text_segment_is_interim() {
        // A `text` segment is carried as an interim; the connector accumulates these
        // and emits the single turn-final once transcription goes quiet.
        let f = json!({ "type": "text", "text": "book a dentist", "start_s": 0.5, "stream_id": 0 });
        assert!(matches!(&decode_message(&f)[..],
            [Frame::InterimTranscription { text, .. }] if text == "book a dentist"));
    }

    #[test]
    fn decode_ignores_other_empty_and_malformed() {
        assert!(decode_message(&json!({ "type": "end_text", "stop_s": 2.5 })).is_empty());
        assert!(decode_message(&json!({ "type": "ready" })).is_empty());
        assert!(decode_message(&json!({ "type": "text", "text": "  " })).is_empty());
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
        stt.start(&StartParams::default()).await.expect("start");
        let silence = Arc::new(AudioFrame::mono(vec![0i16; 1600], 16_000));
        let _ = stt.run_stt(silence).await.expect("run_stt");
    }
}
