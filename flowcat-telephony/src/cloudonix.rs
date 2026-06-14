// SPDX-License-Identifier: Apache-2.0
//
//! Cloudonix Media Streams WebSocket serializer.
//!
//! Cloudonix uses the **same wire protocol as Twilio** (pipecat's
//! `CloudonixFrameSerializer` literally subclasses `TwilioFrameSerializer`):
//! JSON text frames, base64 G.711 μ-law @ 8 kHz, a `start` frame carrying
//! `streamSid`/`callSid`, outbound `media` echoing the `streamSid`, and a
//! `clear` event for barge-in. The carriers differ only in call-control (hang-up)
//! REST APIs, which are out of scope for a pure framing serializer. To avoid a
//! cross-feature dependency on the `twilio` module, this serializer is
//! self-contained.
//!
//! Pure framing only — no I/O, no panics on malformed wire data.

use base64::Engine;
use serde_json::{json, Value};

use flowcat_core::codec::{pcm16_to_ulaw, ulaw_to_pcm16};
use flowcat_core::{AudioChunk, MediaSerializer, SerIn, WsIn, WsOut};

/// Serializer for Cloudonix's (Twilio-compatible) Media Streams WS protocol.
#[derive(Debug, Default)]
pub struct CloudonixSerializer {
    rate: u32,
    stream_sid: Option<String>,
}

impl CloudonixSerializer {
    /// Create a Cloudonix serializer at the given carrier sample rate (typically 8000).
    pub fn new(rate: u32) -> Self {
        Self {
            rate,
            stream_sid: None,
        }
    }

    /// The `streamSid` learned from the carrier's `start` frame, if seen.
    pub fn stream_sid(&self) -> Option<&str> {
        self.stream_sid.as_deref()
    }
}

impl MediaSerializer for CloudonixSerializer {
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
                let stream_sid = start
                    .get("streamSid")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                let call_id = start
                    .get("callSid")
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
                        Ok(ulaw) => SerIn::Audio(AudioChunk {
                            pcm: ulaw_to_pcm16(&ulaw),
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
        let ulaw = pcm16_to_ulaw(&chunk.pcm);
        let payload = base64::engine::general_purpose::STANDARD.encode(ulaw);
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
    fn start_media_stop_roundtrip() {
        let mut s = CloudonixSerializer::new(8000);
        let start = WsIn::Text(
            json!({"event": "start", "start": {"streamSid": "MZ1", "callSid": "CA1"}}).to_string(),
        );
        match s.on_message(&start) {
            SerIn::StreamStart { call_id, stream_id } => {
                assert_eq!(call_id, "CA1");
                assert_eq!(stream_id.as_deref(), Some("MZ1"));
            }
            other => panic!("expected StreamStart, got {other:?}"),
        }
        let ulaw_in = vec![0x10u8, 0x80, 0xFF];
        match s.on_message(&WsIn::Text(
            json!({"event": "media", "media": {"payload": b64(&ulaw_in)}}).to_string(),
        )) {
            SerIn::Audio(c) => assert_eq!(c.pcm, ulaw_to_pcm16(&ulaw_in)),
            other => panic!("expected Audio, got {other:?}"),
        }
        assert!(matches!(
            s.on_message(&WsIn::Text(json!({"event": "stop"}).to_string())),
            SerIn::Stop
        ));
    }

    #[test]
    fn malformed_is_ignored_not_panicked() {
        let mut s = CloudonixSerializer::new(8000);
        assert!(matches!(
            s.on_message(&WsIn::Text("nope".into())),
            SerIn::Ignore
        ));
        assert!(matches!(
            s.on_message(&WsIn::Binary(vec![1])),
            SerIn::Ignore
        ));
        assert!(matches!(
            s.on_message(&WsIn::Text(
                json!({"event": "media", "media": {"payload": "%%%"}}).to_string()
            )),
            SerIn::Ignore
        ));
    }

    #[test]
    fn encode_audio_and_clear_match_twilio_shape() {
        let mut s = CloudonixSerializer::new(8000);
        s.on_message(&WsIn::Text(
            json!({"event": "start", "start": {"streamSid": "MZ7"}}).to_string(),
        ));
        let pcm = vec![1i16, -1, 1000];
        let WsOut::Text(text) = s.encode_audio(&AudioChunk {
            pcm: pcm.clone(),
            sample_rate: 8000,
        }) else {
            panic!("expected Text")
        };
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["event"], "media");
        assert_eq!(v["streamSid"], "MZ7");
        let ulaw = base64::engine::general_purpose::STANDARD
            .decode(v["media"]["payload"].as_str().unwrap())
            .unwrap();
        assert_eq!(ulaw, pcm16_to_ulaw(&pcm));

        let WsOut::Text(clear) = s.encode_clear().unwrap() else {
            panic!("expected Text")
        };
        let cv: Value = serde_json::from_str(&clear).unwrap();
        assert_eq!(cv["event"], "clear");
        assert_eq!(cv["streamSid"], "MZ7");
    }

    #[test]
    fn carrier_rate_reflects_construction() {
        assert_eq!(CloudonixSerializer::new(8000).carrier_rate(), 8000);
    }
}
