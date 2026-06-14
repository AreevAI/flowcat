// SPDX-License-Identifier: Apache-2.0
//
//! **ElevenLabs** streaming TTS (`stream-input` WebSocket).
//!
//! A **(D)istinct** streaming-WebSocket client (cross-checked against pipecat
//! `services/elevenlabs/tts.py`). Connects to the fixed
//! `wss://api.elevenlabs.io/v1/text-to-speech/{voice}/stream-input?model_id=…&output_format=pcm_<rate>`
//! with the API key in the `xi-api-key` header, then drives one utterance as the
//! BOS → text → EOS sequence the single-stream endpoint expects:
//!
//! ```json
//! { "text": " ", "voice_settings": { … } }   // BOS opens the stream
//! { "text": "hello there" }                    // the utterance
//! { "text": "" }                               // EOS flushes + ends
//! ```
//!
//! Server messages carry base64 PCM + optional character alignment:
//!
//! ```json
//! { "audio": "<base64 pcm>",
//!   "alignment": { "chars": ["h","i"], "charStartTimesMs": [0, 90], "charDurationsMs": [90, 80] } }
//! { "isFinal": true }
//! ```
//!
//! Base64 PCM → [`Frame::TtsAudio`]; the character alignment is folded into
//! word timings → [`Frame::TtsText`] (split on spaces, each word stamped at its
//! first character's start time). `isFinal` ends the utterance. The request
//! encode + message decode are **pure functions** so the wire format is
//! unit-tested without a socket.

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

/// ElevenLabs' TTS WebSocket host. The voice id + query is appended at connect.
pub const ELEVENLABS_WSS_BASE: &str = "wss://api.elevenlabs.io";

/// ElevenLabs streaming-TTS session.
pub struct ElevenLabsTts {
    api_key: String,
    voice_id: String,
    model: String,
    sample_rate: u32,
    session: Option<WsTtsSession>,
    ctx_counter: u64,
}

impl ElevenLabsTts {
    /// Construct bound to `api_key` + `voice_id` (default `eleven_flash_v2_5`,
    /// 24 kHz raw PCM output).
    pub fn new(api_key: impl Into<String>, voice_id: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            voice_id: voice_id.into(),
            model: "eleven_flash_v2_5".to_string(),
            sample_rate: 24_000,
            session: None,
            ctx_counter: 0,
        }
    }

    /// Override the model (default `eleven_flash_v2_5`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the output sample rate (default 24 kHz). ElevenLabs supports
    /// 8000/16000/22050/24000/32000/44100/48000.
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// The `stream-input` connect URL (host fixed; key in the header, not the URL).
    fn url(&self) -> String {
        format!(
            "{ELEVENLABS_WSS_BASE}/v1/text-to-speech/{}/stream-input?model_id={}&output_format=pcm_{}",
            self.voice_id, self.model, self.sample_rate
        )
    }
}

#[async_trait]
impl TtsService for ElevenLabsTts {
    fn name(&self) -> &str {
        "elevenlabs"
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        let cfg = WsTtsConfig {
            url: self.url(),
            headers: vec![("xi-api-key".to_string(), self.api_key.clone())],
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
        let session = ws_tts::require(&mut self.session, "elevenlabs")?;
        session.synthesize(msgs, context_id, rate).await
    }
}

/// The BOS / text / EOS message sequence for one utterance (pure — the
/// wire-fixture seam). The first message opens the stream with default voice
/// settings; the second carries the text; the empty `text` flushes + closes.
fn build_messages(text: &str) -> Vec<OutMsg> {
    let bos = json!({
        "text": " ",
        "voice_settings": { "stability": 0.5, "similarity_boost": 0.8 },
    });
    let utterance = json!({ "text": text });
    let eos = json!({ "text": "" });
    vec![
        OutMsg::Text(bos.to_string()),
        OutMsg::Text(utterance.to_string()),
        OutMsg::Text(eos.to_string()),
    ]
}

/// Fold ElevenLabs character alignment into `(word, start_seconds)` pairs (pure).
/// Splits the char stream on spaces; each word is stamped at the start time of
/// its first character. A length mismatch between `chars` and `charStartTimesMs`
/// yields no words (never panics).
fn words_from_alignment(alignment: &Value) -> Vec<(String, f32)> {
    let chars = alignment.get("chars").and_then(|c| c.as_array());
    let starts = alignment.get("charStartTimesMs").and_then(|c| c.as_array());
    let (Some(chars), Some(starts)) = (chars, starts) else {
        return vec![];
    };
    if chars.len() != starts.len() {
        return vec![];
    }
    let mut out = Vec::new();
    let mut word = String::new();
    let mut word_start: Option<f32> = None;
    for (c, s) in chars.iter().zip(starts.iter()) {
        let ch = c.as_str().unwrap_or("");
        let start_ms = s.as_f64().unwrap_or(0.0) as f32;
        if ch == " " {
            if !word.is_empty() {
                out.push((
                    std::mem::take(&mut word),
                    word_start.unwrap_or(0.0) / 1000.0,
                ));
                word_start = None;
            }
        } else {
            if word_start.is_none() {
                word_start = Some(start_ms);
            }
            word.push_str(ch);
        }
    }
    if !word.is_empty() {
        out.push((word, word_start.unwrap_or(0.0) / 1000.0));
    }
    out
}

/// Decode one ElevenLabs server message (pure — the wire-fixture seam). `audio`
/// → PCM; `alignment`/`normalizedAlignment` → word timings; `isFinal` ends the
/// run. Binary frames and anything else are ignored.
pub(crate) fn decode_message(json: Option<&Value>, _binary: Option<&[u8]>) -> Decoded {
    let Some(value) = json else {
        return Decoded::Ignore;
    };
    if value.get("isFinal").and_then(|v| v.as_bool()) == Some(true) {
        return Decoded::Done;
    }
    if let Some(audio) = value.get("audio").and_then(|a| a.as_str()) {
        let pcm = ws_tts::pcm_from_le_bytes(&ws_tts::base64_decode(audio));
        return Decoded::Audio(pcm);
    }
    if let Some(alignment) = value
        .get("normalizedAlignment")
        .or_else(|| value.get("alignment"))
        .filter(|v| !v.is_null())
    {
        let words = words_from_alignment(alignment);
        if !words.is_empty() {
            return Decoded::Words(words);
        }
    }
    Decoded::Ignore
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine as _;

    #[test]
    fn url_uses_the_fixed_host_with_voice_and_format() {
        let t = ElevenLabsTts::new("secret-key-xyz", "voice-x").sample_rate(16_000);
        let url = t.url();
        assert!(url.starts_with("wss://api.elevenlabs.io/v1/text-to-speech/voice-x/stream-input?"));
        assert!(url.contains("model_id=eleven_flash_v2_5"));
        assert!(url.contains("output_format=pcm_16000"));
        // The key never appears in the URL (it travels in the xi-api-key header).
        assert!(!url.contains("secret-key-xyz"));
    }

    #[test]
    fn build_messages_is_bos_text_eos() {
        let msgs = build_messages("hello there");
        assert_eq!(msgs.len(), 3);
        let texts: Vec<Value> = msgs
            .iter()
            .map(|m| match m {
                OutMsg::Text(t) => serde_json::from_str(t).unwrap(),
                OutMsg::Binary(_) => panic!("expected text"),
            })
            .collect();
        assert_eq!(texts[0]["text"], " ");
        assert!(texts[0]["voice_settings"].is_object());
        assert_eq!(texts[1]["text"], "hello there");
        assert_eq!(texts[2]["text"], "");
    }

    #[test]
    fn decode_audio_message_into_pcm() {
        // Two LE i16 samples: 1 and -1 → bytes [1,0, 255,255].
        let b64 = base64::engine::general_purpose::STANDARD.encode([1u8, 0, 255, 255]);
        let msg = json!({ "audio": b64 });
        match decode_message(Some(&msg), None) {
            Decoded::Audio(pcm) => assert_eq!(pcm, vec![1, -1]),
            _ => panic!("expected Audio"),
        }
    }

    #[test]
    fn decode_alignment_into_word_timestamps() {
        let msg = json!({
            "audio": null,
            "alignment": {
                "chars": ["b", "o", "o", "k", " ", "i", "t"],
                "charStartTimesMs": [0, 30, 60, 90, 120, 150, 180],
                "charDurationsMs": [30, 30, 30, 30, 30, 30, 30],
            }
        });
        match decode_message(Some(&msg), None) {
            Decoded::Words(words) => {
                assert_eq!(words.len(), 2);
                assert_eq!(words[0].0, "book");
                assert!((words[0].1 - 0.0).abs() < 1e-6);
                assert_eq!(words[1].0, "it");
                assert!((words[1].1 - 0.150).abs() < 1e-6);
            }
            _ => panic!("expected Words"),
        }
    }

    #[test]
    fn decode_final_and_ignore_and_malformed() {
        assert!(matches!(
            decode_message(Some(&json!({ "isFinal": true })), None),
            Decoded::Done
        ));
        // No audio / alignment → ignore.
        assert!(matches!(
            decode_message(Some(&json!({ "type": "metadata" })), None),
            Decoded::Ignore
        ));
        // Mismatched alignment lengths → no words → ignore (no panic).
        let bad = json!({ "alignment": { "chars": ["a"], "charStartTimesMs": [0, 1] } });
        assert!(matches!(decode_message(Some(&bad), None), Decoded::Ignore));
        // A binary frame → ignore.
        assert!(matches!(
            decode_message(None, Some(&[1, 2, 3])),
            Decoded::Ignore
        ));
    }

    /// Live smoke (requires `ELEVENLABS_API_KEY` + `ELEVENLABS_VOICE_ID`). Run:
    /// `ELEVENLABS_API_KEY=… ELEVENLABS_VOICE_ID=… cargo test -p flowcat-services --features tts-elevenlabs -- --ignored elevenlabs_live`
    #[tokio::test]
    #[ignore = "requires ELEVENLABS_API_KEY + ELEVENLABS_VOICE_ID"]
    async fn elevenlabs_live_synthesizes_audio() {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let key = std::env::var("ELEVENLABS_API_KEY").expect("ELEVENLABS_API_KEY");
        let voice = std::env::var("ELEVENLABS_VOICE_ID").expect("ELEVENLABS_VOICE_ID");
        let mut tts = ElevenLabsTts::new(key, voice);
        tts.start(&StartParams::default()).await.expect("connect");
        let frames = tts.run_tts("Hello from flowcat.").await.expect("run_tts");
        let audio_chunks = frames
            .iter()
            .filter(|f| matches!(f, Frame::TtsAudio { .. }))
            .count();
        assert!(audio_chunks > 0, "expected at least one TtsAudio chunk");
    }
}
