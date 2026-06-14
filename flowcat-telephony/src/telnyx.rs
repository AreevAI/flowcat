// SPDX-License-Identifier: Apache-2.0
//
//! Telnyx Media Streaming WebSocket serializer.
//!
//! Telnyx speaks JSON **text** frames in both directions; the audio payload is
//! base64 G.711, either **PCMU** (μ-law) or **PCMA** (a-law) @ 8 kHz, selected
//! per call. Wire shapes are ported from the vendored pipecat
//! `TelnyxFrameSerializer` (`pipecat/src/pipecat/serializers/telnyx.py`).
//!
//! Inbound (Telnyx → us), snake_case keys:
//! ```json
//! {"event":"start","start":{"stream_id":"…","call_control_id":"…",
//!   "media_format":{"encoding":"PCMU","sample_rate":8000}}}
//! {"event":"media","media":{"payload":"<base64 G.711>"}}
//! {"event":"dtmf","dtmf":{"digit":"1"}}
//! {"event":"stop"}
//! ```
//! Outbound (us → Telnyx) — Telnyx does **not** echo a stream id on `media`:
//! ```json
//! {"event":"media","media":{"payload":"<base64 G.711>"}}
//! {"event":"clear"}        // barge-in / interruption
//! ```
//!
//! Pure framing only — no I/O, no panics on malformed wire data.

use base64::Engine;
use serde_json::{json, Value};

use flowcat_core::codec::{alaw_to_pcm16, pcm16_to_alaw, pcm16_to_ulaw, ulaw_to_pcm16};
use flowcat_core::{AudioChunk, MediaSerializer, SerIn, WsIn, WsOut};

/// G.711 companding selected for a Telnyx stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Encoding {
    /// μ-law (G.711 PCMU) — the North-American default.
    #[default]
    Pcmu,
    /// A-law (G.711 PCMA) — the European default.
    Pcma,
}

impl Encoding {
    /// Parse a Telnyx `media_format.encoding` string (case-insensitive).
    pub fn from_str_lossy(s: &str) -> Option<Self> {
        match s.to_ascii_uppercase().as_str() {
            "PCMU" => Some(Encoding::Pcmu),
            "PCMA" => Some(Encoding::Pcma),
            _ => None,
        }
    }
}

/// Serializer for Telnyx's Media Streaming WebSocket protocol.
#[derive(Debug, Default)]
pub struct TelnyxSerializer {
    rate: u32,
    encoding: Encoding,
    stream_id: Option<String>,
}

impl TelnyxSerializer {
    /// Create a Telnyx serializer at the given carrier sample rate (typically
    /// 8000) and codec. The codec is also refreshed from the `start` frame's
    /// `media_format.encoding` if present.
    pub fn new(rate: u32, encoding: Encoding) -> Self {
        Self {
            rate,
            encoding,
            stream_id: None,
        }
    }

    /// The negotiated G.711 companding.
    pub fn encoding(&self) -> Encoding {
        self.encoding
    }

    /// The `stream_id` learned from the carrier's `start` frame, if seen.
    pub fn stream_id(&self) -> Option<&str> {
        self.stream_id.as_deref()
    }

    fn decode_g711(&self, bytes: &[u8]) -> Vec<i16> {
        match self.encoding {
            Encoding::Pcmu => ulaw_to_pcm16(bytes),
            Encoding::Pcma => alaw_to_pcm16(bytes),
        }
    }

    fn encode_g711(&self, pcm: &[i16]) -> Vec<u8> {
        match self.encoding {
            Encoding::Pcmu => pcm16_to_ulaw(pcm),
            Encoding::Pcma => pcm16_to_alaw(pcm),
        }
    }
}

impl MediaSerializer for TelnyxSerializer {
    fn on_message(&mut self, msg: &WsIn) -> SerIn {
        let text = match msg {
            WsIn::Text(t) => t,
            WsIn::Close => return SerIn::Stop,
            WsIn::Binary(_) => return SerIn::Ignore,
        };

        let v: Value = match serde_json::from_str(text) {
            Ok(v) => v,
            Err(_) => return SerIn::Ignore,
        };

        match v.get("event").and_then(Value::as_str) {
            Some("start") => {
                let start = v.get("start").cloned().unwrap_or(Value::Null);
                let stream_id = start
                    .get("stream_id")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                // Refresh the codec from the negotiated media_format if present.
                if let Some(enc) = start
                    .get("media_format")
                    .and_then(|m| m.get("encoding"))
                    .and_then(Value::as_str)
                    .and_then(Encoding::from_str_lossy)
                {
                    self.encoding = enc;
                }
                let call_id = start
                    .get("call_control_id")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .or_else(|| stream_id.clone())
                    .unwrap_or_default();
                self.stream_id = stream_id.clone();
                SerIn::StreamStart { call_id, stream_id }
            }
            Some("media") => {
                let payload = v
                    .get("media")
                    .and_then(|m| m.get("payload"))
                    .and_then(Value::as_str);
                match payload {
                    Some(b64) => match base64::engine::general_purpose::STANDARD.decode(b64) {
                        Ok(g711) => SerIn::Audio(AudioChunk {
                            pcm: self.decode_g711(&g711),
                            sample_rate: self.rate,
                        }),
                        Err(_) => SerIn::Ignore,
                    },
                    None => SerIn::Ignore,
                }
            }
            Some("stop") => SerIn::Stop,
            // `connected`, `dtmf`, `mark`, unknown events: no action here.
            _ => SerIn::Ignore,
        }
    }

    fn encode_audio(&self, chunk: &AudioChunk) -> WsOut {
        let g711 = self.encode_g711(&chunk.pcm);
        let payload = base64::engine::general_purpose::STANDARD.encode(g711);
        // Telnyx does not echo a stream id on outbound media.
        let answer = json!({
            "event": "media",
            "media": { "payload": payload },
        });
        WsOut::Text(answer.to_string())
    }

    fn encode_clear(&self) -> Option<WsOut> {
        // pipecat: `{"event":"clear"}` (no stream id) on interruption.
        Some(WsOut::Text(json!({ "event": "clear" }).to_string()))
    }

    fn carrier_rate(&self) -> u32 {
        self.rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b64(bytes: &[u8]) -> String {
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    #[test]
    fn start_event_parses_ids_and_refreshes_codec() {
        let mut s = TelnyxSerializer::new(8000, Encoding::Pcmu);
        let frame = WsIn::Text(
            json!({
                "event": "start",
                "start": {
                    "stream_id": "strm-1",
                    "call_control_id": "ccid-9",
                    "media_format": {"encoding": "PCMA", "sample_rate": 8000}
                }
            })
            .to_string(),
        );
        match s.on_message(&frame) {
            SerIn::StreamStart { call_id, stream_id } => {
                assert_eq!(call_id, "ccid-9");
                assert_eq!(stream_id.as_deref(), Some("strm-1"));
            }
            other => panic!("expected StreamStart, got {other:?}"),
        }
        assert_eq!(s.encoding(), Encoding::Pcma);
        assert_eq!(s.stream_id(), Some("strm-1"));
    }

    #[test]
    fn media_pcmu_decodes_with_ulaw() {
        let mut s = TelnyxSerializer::new(8000, Encoding::Pcmu);
        let g711 = vec![0x10u8, 0x80, 0xFF];
        match s.on_message(&WsIn::Text(
            json!({"event": "media", "media": {"payload": b64(&g711)}}).to_string(),
        )) {
            SerIn::Audio(c) => assert_eq!(c.pcm, ulaw_to_pcm16(&g711)),
            other => panic!("expected Audio, got {other:?}"),
        }
    }

    #[test]
    fn media_pcma_decodes_with_alaw() {
        let mut s = TelnyxSerializer::new(8000, Encoding::Pcma);
        let g711 = vec![0x10u8, 0x80, 0xFF];
        match s.on_message(&WsIn::Text(
            json!({"event": "media", "media": {"payload": b64(&g711)}}).to_string(),
        )) {
            SerIn::Audio(c) => assert_eq!(c.pcm, alaw_to_pcm16(&g711)),
            other => panic!("expected Audio, got {other:?}"),
        }
    }

    #[test]
    fn stop_close_and_malformed_handled_safely() {
        let mut s = TelnyxSerializer::new(8000, Encoding::Pcmu);
        assert!(matches!(
            s.on_message(&WsIn::Text(json!({"event": "stop"}).to_string())),
            SerIn::Stop
        ));
        assert!(matches!(s.on_message(&WsIn::Close), SerIn::Stop));
        assert!(matches!(
            s.on_message(&WsIn::Text("oops".into())),
            SerIn::Ignore
        ));
        assert!(matches!(
            s.on_message(&WsIn::Text(
                json!({"event": "media", "media": {"payload": "###"}}).to_string()
            )),
            SerIn::Ignore
        ));
        assert!(matches!(
            s.on_message(&WsIn::Binary(vec![1, 2])),
            SerIn::Ignore
        ));
    }

    #[test]
    fn encode_audio_pcmu_roundtrip_no_stream_id() {
        let s = TelnyxSerializer::new(8000, Encoding::Pcmu);
        let pcm = vec![0i16, 500, -500, 20000];
        let WsOut::Text(text) = s.encode_audio(&AudioChunk {
            pcm: pcm.clone(),
            sample_rate: 8000,
        }) else {
            panic!("expected Text")
        };
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["event"], "media");
        assert!(v.get("stream_id").is_none() && v.get("streamId").is_none());
        let g711 = base64::engine::general_purpose::STANDARD
            .decode(v["media"]["payload"].as_str().unwrap())
            .unwrap();
        assert_eq!(g711, pcm16_to_ulaw(&pcm));
    }

    #[test]
    fn encode_audio_pcma_uses_alaw() {
        let s = TelnyxSerializer::new(8000, Encoding::Pcma);
        let pcm = vec![1i16, -1, 12345];
        let WsOut::Text(text) = s.encode_audio(&AudioChunk {
            pcm: pcm.clone(),
            sample_rate: 8000,
        }) else {
            panic!("expected Text")
        };
        let v: Value = serde_json::from_str(&text).unwrap();
        let g711 = base64::engine::general_purpose::STANDARD
            .decode(v["media"]["payload"].as_str().unwrap())
            .unwrap();
        assert_eq!(g711, pcm16_to_alaw(&pcm));
    }

    #[test]
    fn encode_clear_has_no_stream_id() {
        let s = TelnyxSerializer::new(8000, Encoding::Pcmu);
        let WsOut::Text(text) = s.encode_clear().unwrap() else {
            panic!("expected Text")
        };
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["event"], "clear");
        assert!(v.get("stream_id").is_none());
    }

    #[test]
    fn carrier_rate_reflects_construction() {
        assert_eq!(
            TelnyxSerializer::new(8000, Encoding::Pcmu).carrier_rate(),
            8000
        );
    }
}
