// SPDX-License-Identifier: Apache-2.0
//
//! Vonage Audio Connector WebSocket serializer.
//!
//! Vonage's Audio Connector streams **raw 16-bit little-endian PCM as binary**
//! WebSocket frames (not base64, not JSON-wrapped, not μ-law) — default 16 kHz.
//! Control events (`websocket:connected`/`:cleared`/`:dtmf`/`:notify`) arrive as
//! JSON **text** frames. Wire shapes are ported from the vendored pipecat
//! `VonageFrameSerializer` (`pipecat/src/pipecat/serializers/vonage.py`).
//!
//! Inbound (Vonage → us):
//! - **binary** frame → raw PCM16LE audio.
//! - text `{"event":"websocket:connected","content-type":"audio/l16;rate=16000"}` → stream start.
//! - text `{"event":"websocket:dtmf","digit":"1"}` → ignored at this layer.
//! - text `{"event":"websocket:cleared"}` → ignored (playback-cleared ack).
//!
//! Outbound (us → Vonage):
//! - audio → **binary** PCM16LE bytes ([`WsOut::Binary`]).
//! - barge-in → text `{"action":"clear"}`.
//!
//! Pure framing only — no I/O, no panics on malformed wire data.

use serde_json::{json, Value};

use flowcat_core::{AudioChunk, MediaSerializer, SerIn, WsIn, WsOut};

/// Decode raw 16-bit little-endian PCM bytes into samples (drops a trailing odd byte).
fn pcm_bytes_to_i16(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Encode PCM samples into raw 16-bit little-endian bytes.
fn i16_to_pcm_bytes(pcm: &[i16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(pcm.len() * 2);
    for s in pcm {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// Serializer for Vonage's Audio Connector WebSocket protocol (binary PCM).
#[derive(Debug, Default)]
pub struct VonageSerializer {
    rate: u32,
    started: bool,
}

impl VonageSerializer {
    /// Create a Vonage serializer at the given carrier sample rate (commonly
    /// 8000 / 16000 / 24000; pipecat defaults to 16000).
    pub fn new(rate: u32) -> Self {
        Self {
            rate,
            started: false,
        }
    }
}

impl MediaSerializer for VonageSerializer {
    fn on_message(&mut self, msg: &WsIn) -> SerIn {
        match msg {
            // Binary frame = raw PCM16LE audio.
            WsIn::Binary(bytes) => SerIn::Audio(AudioChunk {
                pcm: pcm_bytes_to_i16(bytes),
                sample_rate: self.rate,
            }),
            WsIn::Close => SerIn::Stop,
            WsIn::Text(text) => {
                let v: Value = match serde_json::from_str(text) {
                    Ok(v) => v,
                    Err(_) => return SerIn::Ignore,
                };
                match v.get("event").and_then(Value::as_str) {
                    Some("websocket:connected") => {
                        // The connected event is the closest thing to a stream
                        // start; Vonage does not carry a call id here. Emit it
                        // once (idempotent) so the pipeline can begin.
                        if self.started {
                            return SerIn::Ignore;
                        }
                        self.started = true;
                        SerIn::StreamStart {
                            call_id: String::new(),
                            stream_id: None,
                        }
                    }
                    // dtmf / cleared / notify / unknown: no action at this layer.
                    _ => SerIn::Ignore,
                }
            }
        }
    }

    fn encode_audio(&self, chunk: &AudioChunk) -> WsOut {
        // Vonage expects raw binary PCM (not base64, not JSON).
        WsOut::Binary(i16_to_pcm_bytes(&chunk.pcm))
    }

    fn encode_clear(&self) -> Option<WsOut> {
        // pipecat: `{"action":"clear"}` on interruption.
        Some(WsOut::Text(json!({ "action": "clear" }).to_string()))
    }

    fn carrier_rate(&self) -> u32 {
        self.rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_frame_is_decoded_as_pcm() {
        let mut s = VonageSerializer::new(16000);
        let pcm = vec![0i16, 1000, -1000, 32767, -32768];
        let bytes = i16_to_pcm_bytes(&pcm);
        match s.on_message(&WsIn::Binary(bytes)) {
            SerIn::Audio(c) => {
                assert_eq!(c.sample_rate, 16000);
                assert_eq!(c.pcm, pcm);
            }
            other => panic!("expected Audio, got {other:?}"),
        }
    }

    #[test]
    fn connected_event_starts_stream_once() {
        let mut s = VonageSerializer::new(16000);
        let connected = WsIn::Text(
            json!({"event": "websocket:connected", "content-type": "audio/l16;rate=16000"})
                .to_string(),
        );
        assert!(matches!(
            s.on_message(&connected),
            SerIn::StreamStart { .. }
        ));
        // A second connected event is idempotent (no duplicate StreamStart).
        assert!(matches!(s.on_message(&connected), SerIn::Ignore));
    }

    #[test]
    fn dtmf_cleared_and_malformed_are_ignored() {
        let mut s = VonageSerializer::new(16000);
        assert!(matches!(
            s.on_message(&WsIn::Text(
                json!({"event": "websocket:dtmf", "digit": "1"}).to_string()
            )),
            SerIn::Ignore
        ));
        assert!(matches!(
            s.on_message(&WsIn::Text(
                json!({"event": "websocket:cleared"}).to_string()
            )),
            SerIn::Ignore
        ));
        assert!(matches!(
            s.on_message(&WsIn::Text("not json".into())),
            SerIn::Ignore
        ));
    }

    #[test]
    fn close_is_stop() {
        let mut s = VonageSerializer::new(16000);
        assert!(matches!(s.on_message(&WsIn::Close), SerIn::Stop));
    }

    #[test]
    fn encode_audio_is_binary_pcm_roundtrip() {
        let s = VonageSerializer::new(16000);
        let pcm = vec![7i16, -7, 4242, -4242];
        let WsOut::Binary(bytes) = s.encode_audio(&AudioChunk {
            pcm: pcm.clone(),
            sample_rate: 16000,
        }) else {
            panic!("expected Binary")
        };
        assert_eq!(pcm_bytes_to_i16(&bytes), pcm);
    }

    #[test]
    fn encode_clear_is_action_clear() {
        let s = VonageSerializer::new(16000);
        let WsOut::Text(text) = s.encode_clear().unwrap() else {
            panic!("expected Text")
        };
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["action"], "clear");
    }

    #[test]
    fn carrier_rate_reflects_construction() {
        assert_eq!(VonageSerializer::new(24000).carrier_rate(), 24000);
    }
}
