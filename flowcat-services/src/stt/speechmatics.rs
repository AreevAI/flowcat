// SPDX-License-Identifier: Apache-2.0
//
//! **Speechmatics** streaming STT (Realtime v2).
//!
//! A **(D)istinct** streaming-WebSocket client. Connects to
//! `wss://eu2.rt.speechmatics.com/v2` with `Authorization: Bearer <api-key>`,
//! sends a `StartRecognition` init message (audio format + transcription config),
//! then streams raw little-endian PCM as **binary** frames. The server replies
//! with the Speechmatics RT v2 message stream (discriminated by a top-level
//! `message` field):
//!
//! ```json
//! { "message": "RecognitionStarted", "id": "…" }
//! { "message": "AddPartialTranscript", "metadata": { "transcript": "book a" } }
//! { "message": "AddTranscript",        "metadata": { "transcript": "book a dentist" } }
//! { "message": "EndOfTranscript" }
//! ```
//!
//! `AddTranscript` → final [`Frame::Transcription`]; `AddPartialTranscript` →
//! [`Frame::InterimTranscription`]; every other message (RecognitionStarted,
//! AudioAdded, EndOfTranscript, Error, …) yields nothing. Decode is **pure**.

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

/// Speechmatics' default Realtime v2 WSS endpoint. The **host is fixed**; the API
/// key travels only in the `Authorization` header, never in the URL.
pub const SPEECHMATICS_WSS: &str = "wss://eu2.rt.speechmatics.com/v2";

/// Speechmatics streaming-STT session (Realtime v2).
pub struct SpeechmaticsStt {
    api_key: String,
    sample_rate: u32,
    language: String,
    session: Option<WsSttSession>,
    muted: bool,
}

impl SpeechmaticsStt {
    /// Construct bound to `api_key` (default 16 kHz input, English).
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            sample_rate: 16_000,
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

    /// Override the transcription language (default `en`).
    pub fn language(mut self, lang: impl Into<String>) -> Self {
        self.language = lang.into();
        self
    }

    /// The `StartRecognition` handshake message. Testable without a socket.
    pub(crate) fn start_recognition(&self) -> Value {
        json!({
            "message": "StartRecognition",
            "audio_format": {
                "type": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": self.sample_rate,
            },
            "transcription_config": {
                "language": self.language,
                "enable_partials": true,
            },
        })
    }
}

#[async_trait]
impl SttService for SpeechmaticsStt {
    fn name(&self) -> &str {
        "speechmatics"
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        let cfg = WsSttConfig {
            url: SPEECHMATICS_WSS.to_string(),
            headers: vec![(
                "Authorization".to_string(),
                format!("Bearer {}", self.api_key),
            )],
            init_message: Some(self.start_recognition().to_string()),
            decode: decode_message,
        };
        // Lazy connect: open the WS (sending StartRecognition) on first audio, NOT
        // here — an eager connect stalls the pipeline Start handshake.
        self.session = Some(WsSttSession::lazy(cfg));
        Ok(())
    }

    async fn run_stt(&mut self, audio: Arc<AudioFrame>) -> Result<Vec<Frame>> {
        let muted = self.muted;
        let session = ws_stt::require(&mut self.session, "speechmatics")?;
        // While muted, feed SILENCE (not the mic) to keep the socket warm without
        // transcribing the bot's echo; decoded frames are dropped while muted.
        if muted {
            let silence = AudioFrame::mono(vec![0i16; audio.pcm.len()], audio.sample_rate);
            let _ = session.send_pcm_binary(&silence).await;
            session.drain();
            return Ok(vec![]);
        }
        // Speechmatics' AddAudio is the raw binary PCM frame itself.
        session.send_pcm_binary(&audio).await?;
        Ok(session.drain())
    }

    async fn set_muted(&mut self, muted: bool) {
        self.muted = muted;
    }
}

/// Decode one Speechmatics RT v2 server message. **Pure.** `AddTranscript` →
/// final, `AddPartialTranscript` → interim; anything else → nothing.
pub(crate) fn decode_message(value: &Value) -> Vec<Frame> {
    let kind = value.get("message").and_then(|m| m.as_str()).unwrap_or("");
    let is_final = match kind {
        "AddTranscript" => true,
        "AddPartialTranscript" => false,
        _ => return vec![],
    };
    let transcript = value
        .get("metadata")
        .and_then(|m| m.get("transcript"))
        .and_then(|t| t.as_str())
        .unwrap_or("");
    if transcript.is_empty() {
        return vec![];
    }
    let user_id: Arc<str> = Arc::from("user");
    if is_final {
        vec![Frame::Transcription {
            text: transcript.to_string(),
            user_id,
            language: None,
            final_: true,
        }]
    } else {
        vec![Frame::InterimTranscription {
            text: transcript.to_string(),
            user_id,
            language: None,
        }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn start_recognition_encodes_raw_pcm_and_language() {
        let m = SpeechmaticsStt::new("k")
            .sample_rate(8000)
            .language("de")
            .start_recognition();
        assert_eq!(m["message"], "StartRecognition");
        assert_eq!(m["audio_format"]["encoding"], "pcm_s16le");
        assert_eq!(m["audio_format"]["sample_rate"], 8000);
        assert_eq!(m["transcription_config"]["language"], "de");
    }

    #[test]
    fn decode_add_transcript_is_final() {
        let msg = json!({
            "message": "AddTranscript",
            "metadata": { "transcript": "book a dentist", "start_time": 0.0, "end_time": 1.0 },
            "results": []
        });
        match &decode_message(&msg)[..] {
            [Frame::Transcription { text, final_, .. }] => {
                assert_eq!(text, "book a dentist");
                assert!(final_);
            }
            other => panic!("expected final, got {other:?}"),
        }
    }

    #[test]
    fn decode_add_partial_transcript_is_interim() {
        let msg = json!({
            "message": "AddPartialTranscript",
            "metadata": { "transcript": "book a" }
        });
        assert!(matches!(
            decode_message(&msg).as_slice(),
            [Frame::InterimTranscription { .. }]
        ));
    }

    #[test]
    fn decode_ignores_control_empty_and_malformed() {
        assert!(decode_message(&json!({ "message": "RecognitionStarted", "id": "x" })).is_empty());
        assert!(decode_message(&json!({ "message": "AudioAdded", "seq_no": 3 })).is_empty());
        assert!(decode_message(&json!({ "message": "EndOfTranscript" })).is_empty());
        // Empty transcript → nothing.
        assert!(decode_message(&json!({
            "message": "AddTranscript", "metadata": { "transcript": "" }
        }))
        .is_empty());
        // Missing metadata / not an object → no panic.
        assert!(decode_message(&json!({ "message": "AddTranscript" })).is_empty());
        assert!(decode_message(&json!("nope")).is_empty());
    }

    /// Live smoke (requires `SPEECHMATICS_API_KEY`). Run:
    /// `SPEECHMATICS_API_KEY=… cargo test -p flowcat-services --features stt-speechmatics -- --ignored speechmatics_live`
    #[tokio::test]
    #[ignore = "requires SPEECHMATICS_API_KEY"]
    async fn speechmatics_live_connects_and_streams() {
        let key = std::env::var("SPEECHMATICS_API_KEY").expect("SPEECHMATICS_API_KEY");
        let mut stt = SpeechmaticsStt::new(key);
        stt.start(&StartParams::default()).await.expect("connect");
        let silence = Arc::new(AudioFrame::mono(vec![0i16; 1600], 16_000));
        let _ = stt.run_stt(silence).await.expect("run_stt");
    }
}
