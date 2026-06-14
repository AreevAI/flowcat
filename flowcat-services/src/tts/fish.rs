// SPDX-License-Identifier: Apache-2.0
//
//! **Fish Audio** TTS — interruptible HTTP client (Group H).
//!
//! Fish Audio's realtime API is a WebSocket framed in **MessagePack** (pipecat
//! `services/fish/tts.py`). The Group-H `tts-fish` feature enables only
//! `reqwest`+`tokio`, so this client uses Fish's REST endpoint, which takes the same
//! MessagePack request body and returns the audio bytes:
//!
//! ```text
//! POST https://api.fish.audio/v1/tts
//!   Authorization: Bearer <key>
//!   Content-Type: application/msgpack
//!   model: <model header, e.g. "s1">
//!   msgpack{ "text": "...", "reference_id": "<voice>", "format": "pcm",
//!            "sample_rate": 24000, "latency": "balanced", "normalize": true }
//! ```
//!
//! With `format: "pcm"` the body is raw little-endian s16 PCM. MessagePack has no
//! crate on this feature set, so we hand-encode the small request map in
//! [`msgpack`] (a fully unit-tested pure function — the request seam).

use std::sync::Arc;

use async_trait::async_trait;

use flowcat_core::error::{FlowcatError, Result};
use flowcat_core::processor::frame::{Frame, StartParams};
use flowcat_core::service::TtsService;

#[path = "tail_tts_common.rs"]
#[allow(clippy::duplicate_mod)] // shared header pattern: included into each Group-H module
mod tail;

/// Fish Audio REST synthesis endpoint.
pub const FISH_TTS_URL: &str = "https://api.fish.audio/v1/tts";
/// Default Fish model (sent as the `model` header).
pub const FISH_DEFAULT_MODEL: &str = "s1";

/// Fish Audio TTS service (HTTP REST, MessagePack request, raw PCM response).
pub struct FishTts {
    api_key: String,
    voice_id: String,
    sample_rate: u32,
    model: String,
    url: String,
    http: reqwest::Client,
    ctx_counter: u64,
}

impl FishTts {
    /// Construct bound to `api_key` + `voice_id` (default 24000 Hz, model `s1`).
    pub fn new(api_key: impl Into<String>, voice_id: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            voice_id: voice_id.into(),
            sample_rate: 24_000,
            model: FISH_DEFAULT_MODEL.to_string(),
            url: FISH_TTS_URL.to_string(),
            http: reqwest::Client::new(),
            ctx_counter: 0,
        }
    }

    /// Override the model (default `s1`).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the output sample rate (default 24000 Hz).
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }
}

/// The Fish synthesis request, as ordered string→value pairs. Pure data — fed to
/// [`msgpack`] for the wire body.
fn request_fields(text: &str, voice_id: &str, sample_rate: u32) -> Vec<(&'static str, MsgVal)> {
    vec![
        ("text", MsgVal::Str(text.to_string())),
        ("reference_id", MsgVal::Str(voice_id.to_string())),
        ("format", MsgVal::Str("pcm".to_string())),
        ("sample_rate", MsgVal::UInt(sample_rate as u64)),
        ("latency", MsgVal::Str("balanced".to_string())),
        ("normalize", MsgVal::Bool(true)),
    ]
}

/// A MessagePack value we need to encode for the Fish request (the small subset the
/// request body uses).
enum MsgVal {
    Str(String),
    UInt(u64),
    Bool(bool),
}

/// Hand-encode an ordered string-keyed map into MessagePack bytes (pure — the
/// request seam). Supports only the value kinds the Fish request uses; sufficient
/// because the request shape is fixed.
fn msgpack(fields: &[(&str, MsgVal)]) -> Vec<u8> {
    let mut out = Vec::new();
    // map header: fixmap (<=15 entries) is 0x80 | len.
    let n = fields.len();
    debug_assert!(n <= 15, "fish request map exceeds fixmap size");
    out.push(0x80 | (n as u8 & 0x0f));
    for (k, v) in fields {
        encode_str(&mut out, k);
        match v {
            MsgVal::Str(s) => encode_str(&mut out, s),
            MsgVal::UInt(u) => encode_uint(&mut out, *u),
            MsgVal::Bool(b) => out.push(if *b { 0xc3 } else { 0xc2 }),
        }
    }
    out
}

/// Encode a UTF-8 string (fixstr / str8 / str16 / str32).
fn encode_str(out: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    let len = bytes.len();
    if len <= 31 {
        out.push(0xa0 | (len as u8));
    } else if len <= 0xff {
        out.push(0xd9);
        out.push(len as u8);
    } else if len <= 0xffff {
        out.push(0xda);
        out.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        out.push(0xdb);
        out.extend_from_slice(&(len as u32).to_be_bytes());
    }
    out.extend_from_slice(bytes);
}

/// Encode an unsigned integer (positive fixint / uint8 / uint16 / uint32 / uint64).
fn encode_uint(out: &mut Vec<u8>, u: u64) {
    if u <= 0x7f {
        out.push(u as u8);
    } else if u <= 0xff {
        out.push(0xcc);
        out.push(u as u8);
    } else if u <= 0xffff {
        out.push(0xcd);
        out.extend_from_slice(&(u as u16).to_be_bytes());
    } else if u <= 0xffff_ffff {
        out.push(0xce);
        out.extend_from_slice(&(u as u32).to_be_bytes());
    } else {
        out.push(0xcf);
        out.extend_from_slice(&u.to_be_bytes());
    }
}

#[async_trait]
impl TtsService for FishTts {
    fn name(&self) -> &str {
        "fish"
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        Ok(())
    }

    async fn run_tts(&mut self, text: &str) -> Result<Vec<Frame>> {
        self.ctx_counter += 1;
        let context_id: Arc<str> = Arc::from(format!("ctx-{}", self.ctx_counter));
        let body = msgpack(&request_fields(text, &self.voice_id, self.sample_rate));

        let resp = self
            .http
            .post(&self.url)
            .bearer_auth(&self.api_key)
            .header("Content-Type", "application/msgpack")
            .header("model", &self.model)
            .body(body)
            .send()
            .await
            .map_err(|e| FlowcatError::Network(format!("fish send: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(FlowcatError::Network(format!("fish http {status}: {body}")));
        }
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| FlowcatError::Network(format!("fish body: {e}")))?;
        let pcm = tail::strip_wav_header(&bytes);
        Ok(tail::one_shot_frames(pcm, self.sample_rate, context_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_str_uses_fixstr_for_short() {
        let mut out = Vec::new();
        encode_str(&mut out, "pcm");
        assert_eq!(out, vec![0xa3, b'p', b'c', b'm']);
    }

    #[test]
    fn encode_uint_known_widths() {
        let mut a = Vec::new();
        encode_uint(&mut a, 5);
        assert_eq!(a, vec![5]); // positive fixint

        let mut b = Vec::new();
        encode_uint(&mut b, 24_000);
        assert_eq!(b, vec![0xcd, 0x5d, 0xc0]); // uint16 BE

        let mut c = Vec::new();
        encode_uint(&mut c, 200);
        assert_eq!(c, vec![0xcc, 200]); // uint8
    }

    #[test]
    fn msgpack_request_is_a_fixmap_with_expected_keys() {
        let fields = request_fields("hi", "voice-x", 24_000);
        let bytes = msgpack(&fields);
        // fixmap header for 6 entries.
        assert_eq!(bytes[0], 0x80 | 6);
        // First key is "text" (fixstr len 4) followed by its value.
        assert_eq!(&bytes[1..6], &[0xa4, b't', b'e', b'x', b't']);
        // The "normalize" value (true) → 0xc3 must appear.
        assert!(bytes.contains(&0xc3));
        // The sample_rate uint16 (24000 = 0x5dc0) BE must appear.
        assert!(bytes.windows(3).any(|w| w == [0xcd, 0x5d, 0xc0]));
    }

    #[test]
    fn pcm_body_frames_audio() {
        let frames = tail::one_shot_frames(&[1, 0, 255, 255], 24_000, Arc::from("c"));
        match &frames[1] {
            Frame::TtsAudio { audio, .. } => assert_eq!(audio.pcm, vec![1, -1]),
            _ => panic!("expected TtsAudio"),
        }
    }

    /// Live smoke (requires `FISH_API_KEY` + `FISH_VOICE_ID`).
    #[tokio::test]
    #[ignore = "requires FISH_API_KEY + FISH_VOICE_ID"]
    async fn fish_live_synthesizes_audio() {
        let key = std::env::var("FISH_API_KEY").expect("FISH_API_KEY");
        let voice = std::env::var("FISH_VOICE_ID").expect("FISH_VOICE_ID");
        let mut tts = FishTts::new(key, voice);
        tts.start(&StartParams::default()).await.expect("start");
        let frames = tts.run_tts("Hello from flowcat.").await.expect("run_tts");
        assert!(frames.iter().any(|f| matches!(f, Frame::TtsAudio { .. })));
    }
}
