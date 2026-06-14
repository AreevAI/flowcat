// SPDX-License-Identifier: Apache-2.0
//
//! **AssemblyAI** streaming STT (Universal-Streaming v3).
//!
//! A **(D)istinct** streaming-WebSocket client. Connects to
//! `wss://streaming.assemblyai.com/v3/ws?sample_rate=…&format_turns=…` with an
//! `Authorization: <api-key>` header (the raw key, **no** `Token`/`Bearer`
//! prefix — cross-checked against pipecat `services/assemblyai/stt.py`), streams
//! raw little-endian PCM as **binary** WS frames, and receives JSON messages:
//!
//! ```json
//! { "type": "Begin", "id": "…", "expires_at": 0 }
//! { "type": "Turn", "transcript": "book a dentist", "end_of_turn": true,
//!   "turn_is_formatted": true, "language_code": "en", "language_confidence": 0.99 }
//! { "type": "Termination", "audio_duration_seconds": 1.0 }
//! ```
//!
//! A `Turn` whose `end_of_turn` is true (and, when formatting is on,
//! `turn_is_formatted` is true) is a final [`Frame::Transcription`]; otherwise it
//! is an [`Frame::InterimTranscription`]. `Begin`/`Termination`/unknown messages
//! yield nothing.
//!
//! The streaming-WS transport (connect + reader-task → mpsc → per-`run_stt`
//! drain) is the shared [`ws_stt`] seam (`ws_stt_common.rs`) reused across this
//! group's distinct-schema WS providers; this file supplies only AssemblyAI's
//! URL/header and its **pure** [`decode_message`] wire-decode.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use flowcat_core::error::Result;
use flowcat_core::processor::frame::{AudioFrame, Frame, Language, StartParams};
use flowcat_core::service::SttService;

use ws_stt::{WsSttConfig, WsSttSession};

/// The shared streaming-WS STT transport, included from `ws_stt_common.rs`. It is
/// pulled in via `#[path]` (not a `mod` in `stt/mod.rs`, which the fan-out must
/// not edit) so every WS provider compiles its own copy and a single-feature
/// build never depends on a sibling provider's feature. The duplicate-mod lint is
/// intentional: each provider owning its own copy is what keeps the features
/// independent.
#[allow(clippy::duplicate_mod)]
#[path = "ws_stt_common.rs"]
pub mod ws_stt;

/// AssemblyAI's fixed Universal-Streaming v3 WSS host. The query string
/// (sample_rate/format_turns) is appended at connect time; the **host is fixed**
/// — only the API key (header) and validated numeric query params are
/// caller-controlled, so there is no SSRF surface.
pub const ASSEMBLYAI_WSS_BASE: &str = "wss://streaming.assemblyai.com/v3/ws";

/// AssemblyAI streaming-STT session (Universal-Streaming v3).
pub struct AssemblyAiStt {
    api_key: String,
    sample_rate: u32,
    format_turns: bool,
    session: Option<WsSttSession>,
    muted: bool,
    /// PCM accumulator: AssemblyAI v3 rejects audio chunks <50 ms (err 3007), but the
    /// pipeline feeds ~20 ms chunks, so we buffer and flush ~100 ms at a time.
    buf: Vec<i16>,
}

impl AssemblyAiStt {
    /// Construct bound to `api_key` (default 16 kHz input, turn formatting on).
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            sample_rate: 16_000,
            format_turns: true,
            muted: false,
            session: None,
            buf: Vec::new(),
        }
    }

    /// Override the input sample rate (default 16 kHz).
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Toggle AssemblyAI transcript turn-formatting (default on).
    pub fn format_turns(mut self, on: bool) -> Self {
        self.format_turns = on;
        self
    }

    /// The connect URL for this config (testable without a socket). The API key
    /// is **never** placed in the URL (it travels in the `Authorization` header).
    pub(crate) fn url(&self) -> String {
        format!(
            "{ASSEMBLYAI_WSS_BASE}?sample_rate={}&format_turns={}",
            self.sample_rate,
            if self.format_turns { "true" } else { "false" }
        )
    }

    fn config(&self) -> WsSttConfig {
        WsSttConfig {
            url: self.url(),
            headers: vec![("Authorization".to_string(), self.api_key.clone())],
            init_message: None,
            decode: decode_message,
        }
    }
}

#[async_trait]
impl SttService for AssemblyAiStt {
    fn name(&self) -> &str {
        "assemblyai"
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        // Lazy connect: open the WS on first audio, NOT here — an eager connect in
        // start() stalls the pipeline Start handshake (no greeting, no audio).
        self.session = Some(WsSttSession::lazy(self.config()));
        Ok(())
    }

    async fn run_stt(&mut self, audio: Arc<AudioFrame>) -> Result<Vec<Frame>> {
        let muted = self.muted;
        // Accumulate ~100 ms before sending: AssemblyAI v3 rejects <50 ms chunks
        // (err 3007), but the pipeline feeds ~20 ms. While muted, buffer SILENCE (not
        // the mic) so the socket stays warm without transcribing the bot's echo.
        if muted {
            self.buf.extend(std::iter::repeat_n(0i16, audio.pcm.len()));
        } else {
            self.buf.extend_from_slice(&audio.pcm);
        }
        let min_samples = (self.sample_rate / 10) as usize; // ~100 ms (within 50–1000 ms)
        let chunk = (self.buf.len() >= min_samples)
            .then(|| AudioFrame::mono(std::mem::take(&mut self.buf), self.sample_rate));

        let session = ws_stt::require(&mut self.session, "assemblyai")?;
        if let Some(chunk) = chunk {
            // AssemblyAI streams raw little-endian PCM as binary frames.
            if muted {
                let _ = session.send_pcm_binary(&chunk).await; // keepalive; ignore errors
            } else {
                session.send_pcm_binary(&chunk).await?;
            }
        }
        let frames = session.drain();
        Ok(if muted { vec![] } else { frames })
    }

    async fn set_muted(&mut self, muted: bool) {
        self.muted = muted;
    }
}

/// Decode one AssemblyAI v3 server message into transcription frames. **Pure** —
/// the seam the wire-fixture tests drive. Only `Turn` messages with a non-empty
/// `transcript` produce a frame; `Begin`/`Termination`/unknown → nothing.
pub(crate) fn decode_message(value: &Value) -> Vec<Frame> {
    if value.get("type").and_then(|t| t.as_str()) != Some("Turn") {
        return vec![];
    }
    let transcript = value
        .get("transcript")
        .and_then(|t| t.as_str())
        .unwrap_or("");
    if transcript.is_empty() {
        return vec![];
    }
    let end_of_turn = value
        .get("end_of_turn")
        .and_then(|b| b.as_bool())
        .unwrap_or(false);
    // Only treat as final once AssemblyAI has formatted the turn (mirrors the
    // pipecat `is_final_turn` gate: end_of_turn AND (not formatting OR formatted)).
    let formatted = value
        .get("turn_is_formatted")
        .and_then(|b| b.as_bool())
        .unwrap_or(true);
    let language = value
        .get("language_code")
        .and_then(|l| l.as_str())
        .map(|l| Language(l.to_string()));
    let user_id: Arc<str> = Arc::from("user");
    if end_of_turn && formatted {
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
    fn url_uses_the_fixed_host_and_omits_the_key() {
        let c = AssemblyAiStt::new("secret-key").sample_rate(8000);
        assert!(c
            .url()
            .starts_with("wss://streaming.assemblyai.com/v3/ws?sample_rate=8000"));
        assert!(c.url().contains("format_turns=true"));
        assert!(!c.url().contains("secret-key"));
    }

    #[test]
    fn decode_final_formatted_turn() {
        let msg = json!({
            "type": "Turn",
            "transcript": "book a dentist appointment",
            "end_of_turn": true,
            "turn_is_formatted": true,
            "language_code": "en",
            "language_confidence": 0.99
        });
        let frames = decode_message(&msg);
        assert_eq!(frames.len(), 1);
        match &frames[0] {
            Frame::Transcription {
                text,
                final_,
                language,
                ..
            } => {
                assert_eq!(text, "book a dentist appointment");
                assert!(final_);
                assert_eq!(language.as_ref().map(|l| l.0.as_str()), Some("en"));
            }
            other => panic!("expected final Transcription, got {}", other.name()),
        }
    }

    #[test]
    fn decode_end_of_turn_but_unformatted_is_interim() {
        // end_of_turn true but not yet formatted → still interim (pipecat gate).
        let msg = json!({
            "type": "Turn",
            "transcript": "book a dentist",
            "end_of_turn": true,
            "turn_is_formatted": false
        });
        assert!(matches!(
            decode_message(&msg).as_slice(),
            [Frame::InterimTranscription { .. }]
        ));
    }

    #[test]
    fn decode_partial_turn_is_interim() {
        let msg = json!({
            "type": "Turn",
            "transcript": "book a",
            "end_of_turn": false,
            "turn_is_formatted": false
        });
        assert!(matches!(
            decode_message(&msg).as_slice(),
            [Frame::InterimTranscription { .. }]
        ));
    }

    #[test]
    fn decode_ignores_begin_termination_empty_and_malformed() {
        assert!(decode_message(&json!({ "type": "Begin", "id": "x", "expires_at": 0 })).is_empty());
        assert!(
            decode_message(&json!({ "type": "Termination", "audio_duration_seconds": 1.0 }))
                .is_empty()
        );
        // Empty transcript → nothing.
        assert!(decode_message(&json!({
            "type": "Turn", "transcript": "", "end_of_turn": true, "turn_is_formatted": true
        }))
        .is_empty());
        // Wrong/unknown type → nothing; missing fields → no panic.
        assert!(decode_message(&json!({ "type": "Whatever" })).is_empty());
        assert!(decode_message(&json!({ "transcript": "no type" })).is_empty());
        assert!(decode_message(&json!("not even an object")).is_empty());
    }

    #[test]
    fn pcm_le_bytes_are_little_endian() {
        let af = AudioFrame::mono(vec![1, -2, 256], 16_000);
        assert_eq!(ws_stt::pcm_le_bytes(&af), vec![1, 0, 254, 255, 0, 1]);
    }

    /// Live smoke (requires `ASSEMBLYAI_API_KEY`): connect + send a beat of
    /// silence, confirm the socket stays open. Run with:
    /// `ASSEMBLYAI_API_KEY=… cargo test -p flowcat-services --features stt-assemblyai -- --ignored assemblyai_live`
    #[tokio::test]
    #[ignore = "requires ASSEMBLYAI_API_KEY"]
    async fn assemblyai_live_connects_and_streams() {
        let key = std::env::var("ASSEMBLYAI_API_KEY").expect("ASSEMBLYAI_API_KEY");
        let mut stt = AssemblyAiStt::new(key);
        stt.start(&StartParams::default()).await.expect("connect");
        let silence = Arc::new(AudioFrame::mono(vec![0i16; 1600], 16_000));
        let _ = stt.run_stt(silence).await.expect("run_stt");
    }
}
