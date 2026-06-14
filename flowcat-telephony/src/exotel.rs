// SPDX-License-Identifier: Apache-2.0
//
//! Exotel Media Streams WebSocket serializer.
//!
//! Unlike the μ-law carriers, Exotel streams **raw 16-bit little-endian PCM**
//! (no G.711 companding), base64-encoded inside JSON text frames. Wire shapes are
//! ported from the vendored pipecat `ExotelFrameSerializer`
//! (`pipecat/src/pipecat/serializers/exotel.py`), which resamples raw PCM rather
//! than transcoding.
//!
//! Inbound (Exotel → us):
//! ```json
//! {"event":"start","stream_sid":"…","start":{"call_sid":"…"}}
//! {"event":"media","media":{"payload":"<base64 PCM16LE>"}}
//! {"event":"dtmf","dtmf":{"digit":"1"}}
//! {"event":"stop"}
//! ```
//! Outbound (us → Exotel):
//! ```json
//! {"event":"media","streamSid":"…","media":{"payload":"<base64 PCM16LE>"}}
//! {"event":"clear","streamSid":"…"}        // barge-in / interruption
//! ```
//!
//! Pure framing only — no I/O, no panics on malformed wire data. PCM byte parsing
//! is length-tolerant (a trailing odd byte is dropped, never panics).

use base64::Engine;
use serde_json::{json, Value};

use flowcat_core::{AudioChunk, MediaSerializer, SerIn, WsIn, WsOut};

/// Decode raw 16-bit little-endian PCM bytes into samples. A trailing odd byte
/// (truncated final sample) is dropped rather than panicking.
fn pcm_bytes_to_i16(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Encode PCM samples back into raw 16-bit little-endian bytes.
fn i16_to_pcm_bytes(pcm: &[i16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(pcm.len() * 2);
    for s in pcm {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// Serializer for Exotel's Media Streams WebSocket protocol (raw PCM).
#[derive(Debug, Default)]
pub struct ExotelSerializer {
    rate: u32,
    stream_sid: Option<String>,
}

impl ExotelSerializer {
    /// Create an Exotel serializer at the given carrier sample rate (typically 8000).
    pub fn new(rate: u32) -> Self {
        Self {
            rate,
            stream_sid: None,
        }
    }

    /// The `stream_sid`/`streamSid` learned from the carrier's `start` frame, if seen.
    pub fn stream_sid(&self) -> Option<&str> {
        self.stream_sid.as_deref()
    }
}

impl MediaSerializer for ExotelSerializer {
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
                // Exotel carries stream_sid at the top level; tolerate camelCase too.
                let stream_sid = v
                    .get("stream_sid")
                    .or_else(|| v.get("streamSid"))
                    .or_else(|| v.get("start").and_then(|s| s.get("stream_sid")))
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                let call_id = v
                    .get("start")
                    .and_then(|s| s.get("call_sid"))
                    .or_else(|| v.get("start").and_then(|s| s.get("callSid")))
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .or_else(|| stream_sid.clone())
                    .unwrap_or_default();
                self.stream_sid = stream_sid.clone();
                SerIn::StreamStart {
                    call_id,
                    stream_id: stream_sid,
                }
            }
            Some("media") => {
                let payload = v
                    .get("media")
                    .and_then(|m| m.get("payload"))
                    .and_then(Value::as_str);
                match payload {
                    Some(b64) => match base64::engine::general_purpose::STANDARD.decode(b64) {
                        Ok(pcm_bytes) => SerIn::Audio(AudioChunk {
                            pcm: pcm_bytes_to_i16(&pcm_bytes),
                            sample_rate: self.rate,
                        }),
                        Err(_) => SerIn::Ignore,
                    },
                    None => SerIn::Ignore,
                }
            }
            Some("stop") => SerIn::Stop,
            _ => SerIn::Ignore,
        }
    }

    fn encode_audio(&self, chunk: &AudioChunk) -> WsOut {
        let pcm_bytes = i16_to_pcm_bytes(&chunk.pcm);
        let payload = base64::engine::general_purpose::STANDARD.encode(pcm_bytes);
        let mut answer = json!({
            "event": "media",
            "media": { "payload": payload },
        });
        if let Some(sid) = &self.stream_sid {
            answer["streamSid"] = Value::String(sid.clone());
        }
        WsOut::Text(answer.to_string())
    }

    fn encode_clear(&self) -> Option<WsOut> {
        // pipecat: `{"event":"clear","streamSid": <stream_sid>}` on interruption.
        let mut answer = json!({ "event": "clear" });
        if let Some(sid) = &self.stream_sid {
            answer["streamSid"] = Value::String(sid.clone());
        }
        Some(WsOut::Text(answer.to_string()))
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
    fn start_event_reads_top_level_stream_sid() {
        let mut s = ExotelSerializer::new(8000);
        match s.on_message(&WsIn::Text(
            json!({"event": "start", "stream_sid": "ex-strm", "start": {"call_sid": "ex-call"}})
                .to_string(),
        )) {
            SerIn::StreamStart { call_id, stream_id } => {
                assert_eq!(call_id, "ex-call");
                assert_eq!(stream_id.as_deref(), Some("ex-strm"));
            }
            other => panic!("expected StreamStart, got {other:?}"),
        }
        assert_eq!(s.stream_sid(), Some("ex-strm"));
    }

    #[test]
    fn media_decodes_raw_pcm_le() {
        let mut s = ExotelSerializer::new(8000);
        let pcm = vec![0i16, 256, -256, 12345, -12345];
        let bytes = i16_to_pcm_bytes(&pcm);
        match s.on_message(&WsIn::Text(
            json!({"event": "media", "media": {"payload": b64(&bytes)}}).to_string(),
        )) {
            SerIn::Audio(c) => {
                assert_eq!(c.sample_rate, 8000);
                assert_eq!(c.pcm, pcm);
            }
            other => panic!("expected Audio, got {other:?}"),
        }
    }

    #[test]
    fn odd_length_pcm_does_not_panic() {
        let mut s = ExotelSerializer::new(8000);
        // 3 bytes = one full sample + a dangling byte.
        let bytes = vec![0x01u8, 0x02, 0x03];
        match s.on_message(&WsIn::Text(
            json!({"event": "media", "media": {"payload": b64(&bytes)}}).to_string(),
        )) {
            SerIn::Audio(c) => assert_eq!(c.pcm.len(), 1),
            other => panic!("expected Audio, got {other:?}"),
        }
    }

    #[test]
    fn stop_close_malformed_safe() {
        let mut s = ExotelSerializer::new(8000);
        assert!(matches!(
            s.on_message(&WsIn::Text(json!({"event": "stop"}).to_string())),
            SerIn::Stop
        ));
        assert!(matches!(s.on_message(&WsIn::Close), SerIn::Stop));
        assert!(matches!(
            s.on_message(&WsIn::Text("nope".into())),
            SerIn::Ignore
        ));
        assert!(matches!(
            s.on_message(&WsIn::Text(
                json!({"event": "media", "media": {"payload": "**"}}).to_string()
            )),
            SerIn::Ignore
        ));
    }

    #[test]
    fn encode_audio_roundtrips_pcm() {
        let mut s = ExotelSerializer::new(8000);
        s.on_message(&WsIn::Text(
            json!({"event": "start", "stream_sid": "ex9", "start": {}}).to_string(),
        ));
        let pcm = vec![5i16, -5, 4096, -4096];
        let WsOut::Text(text) = s.encode_audio(&AudioChunk {
            pcm: pcm.clone(),
            sample_rate: 8000,
        }) else {
            panic!("expected Text")
        };
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["event"], "media");
        assert_eq!(v["streamSid"], "ex9");
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(v["media"]["payload"].as_str().unwrap())
            .unwrap();
        assert_eq!(pcm_bytes_to_i16(&bytes), pcm);
    }

    #[test]
    fn encode_clear_is_clear_with_stream_sid() {
        let mut s = ExotelSerializer::new(8000);
        s.on_message(&WsIn::Text(
            json!({"event": "start", "stream_sid": "ex2", "start": {}}).to_string(),
        ));
        let WsOut::Text(text) = s.encode_clear().unwrap() else {
            panic!("expected Text")
        };
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["event"], "clear");
        assert_eq!(v["streamSid"], "ex2");
    }

    #[test]
    fn carrier_rate_reflects_construction() {
        assert_eq!(ExotelSerializer::new(8000).carrier_rate(), 8000);
    }
}
