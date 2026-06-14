// SPDX-License-Identifier: Apache-2.0
//
//! Twilio Media Streams WebSocket serializer.
//!
//! Twilio speaks JSON **text** frames in both directions; the audio payload is
//! base64 G.711 μ-law @ 8 kHz. Wire shapes are ported from the vendored pipecat
//! `TwilioFrameSerializer` (`pipecat/src/pipecat/serializers/twilio.py`).
//!
//! Inbound (Twilio → us):
//! ```json
//! {"event":"connected","protocol":"Call","version":"1.0.0"}
//! {"event":"start","start":{"streamSid":"MZ…","callSid":"CA…",
//!   "mediaFormat":{"encoding":"audio/x-mulaw","sampleRate":8000,"channels":1}}}
//! {"event":"media","media":{"track":"inbound","payload":"<base64 μ-law>"}}
//! {"event":"dtmf","dtmf":{"digit":"1"}}
//! {"event":"stop"}
//! ```
//! Outbound (us → Twilio):
//! ```json
//! {"event":"media","streamSid":"MZ…","media":{"payload":"<base64 μ-law>"}}
//! {"event":"clear","streamSid":"MZ…"}        // barge-in / interruption
//! ```
//!
//! Pure framing only — no I/O, no panics on malformed wire data. Carrier-side
//! `dtmf` JSON events map to [`SerIn::Ignore`] here (the frozen [`SerIn`] has no
//! DTMF arm); in-band / RFC2833 DTMF is handled by [`crate::dtmf`].

use base64::Engine;
use serde_json::{json, Value};

use flowcat_core::codec::{pcm16_to_ulaw, ulaw_to_pcm16};
use flowcat_core::{AudioChunk, MediaSerializer, SerIn, WsIn, WsOut};

/// Serializer for Twilio's Media Streams WebSocket protocol.
#[derive(Debug, Default)]
pub struct TwilioSerializer {
    /// Carrier sample rate (Twilio telephony μ-law = 8000).
    rate: u32,
    /// The Twilio `streamSid`, learned from the `start` frame; echoed back on
    /// every outbound `media`/`clear` frame.
    stream_sid: Option<String>,
}

impl TwilioSerializer {
    /// Create a Twilio serializer at the given carrier sample rate (typically 8000).
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

impl MediaSerializer for TwilioSerializer {
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
            // `connected`, `dtmf`, `mark` acks, unknown events: no action here.
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
    fn start_event_parses_stream_and_call_sid() {
        let mut s = TwilioSerializer::new(8000);
        let frame = WsIn::Text(
            json!({
                "event": "start",
                "start": {
                    "streamSid": "MZ123",
                    "callSid": "CA456",
                    "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1}
                }
            })
            .to_string(),
        );
        match s.on_message(&frame) {
            SerIn::StreamStart { call_id, stream_id } => {
                assert_eq!(call_id, "CA456");
                assert_eq!(stream_id.as_deref(), Some("MZ123"));
            }
            other => panic!("expected StreamStart, got {other:?}"),
        }
        assert_eq!(s.stream_sid(), Some("MZ123"));
    }

    #[test]
    fn connected_event_is_ignored() {
        let mut s = TwilioSerializer::new(8000);
        assert!(matches!(
            s.on_message(&WsIn::Text(
                json!({"event": "connected", "protocol": "Call", "version": "1.0.0"}).to_string()
            )),
            SerIn::Ignore
        ));
    }

    #[test]
    fn media_event_decodes_ulaw_to_pcm() {
        let mut s = TwilioSerializer::new(8000);
        let ulaw_in = vec![0xFFu8, 0x00, 0x80, 0x7F];
        let frame = WsIn::Text(
            json!({"event": "media", "media": {"track": "inbound", "payload": b64(&ulaw_in)}})
                .to_string(),
        );
        match s.on_message(&frame) {
            SerIn::Audio(chunk) => {
                assert_eq!(chunk.sample_rate, 8000);
                assert_eq!(chunk.pcm, ulaw_to_pcm16(&ulaw_in));
            }
            other => panic!("expected Audio, got {other:?}"),
        }
    }

    #[test]
    fn stop_and_close_map_to_stop() {
        let mut s = TwilioSerializer::new(8000);
        assert!(matches!(
            s.on_message(&WsIn::Text(json!({"event": "stop"}).to_string())),
            SerIn::Stop
        ));
        assert!(matches!(s.on_message(&WsIn::Close), SerIn::Stop));
    }

    #[test]
    fn malformed_and_dtmf_and_binary_are_ignored() {
        let mut s = TwilioSerializer::new(8000);
        assert!(matches!(
            s.on_message(&WsIn::Text(
                json!({"event": "dtmf", "dtmf": {"digit": "1"}}).to_string()
            )),
            SerIn::Ignore
        ));
        assert!(matches!(
            s.on_message(&WsIn::Text("not json".into())),
            SerIn::Ignore
        ));
        assert!(matches!(
            s.on_message(&WsIn::Binary(vec![9, 9])),
            SerIn::Ignore
        ));
        assert!(matches!(
            s.on_message(&WsIn::Text(
                json!({"event": "media", "media": {"payload": "@@bad@@"}}).to_string()
            )),
            SerIn::Ignore
        ));
    }

    #[test]
    fn encode_audio_uses_media_event_with_stream_sid() {
        let mut s = TwilioSerializer::new(8000);
        s.on_message(&WsIn::Text(
            json!({"event": "start", "start": {"streamSid": "MZ1", "callSid": "CA1"}}).to_string(),
        ));
        let pcm = vec![0i16, 50, -50, 30000, -30000];
        let out = s.encode_audio(&AudioChunk {
            pcm: pcm.clone(),
            sample_rate: 8000,
        });
        let WsOut::Text(text) = out else {
            panic!("expected Text")
        };
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["event"], "media");
        assert_eq!(v["streamSid"], "MZ1");
        let payload = v["media"]["payload"].as_str().unwrap();
        let ulaw = base64::engine::general_purpose::STANDARD
            .decode(payload)
            .unwrap();
        assert_eq!(ulaw, pcm16_to_ulaw(&pcm));
        assert_eq!(ulaw_to_pcm16(&ulaw).len(), pcm.len());
    }

    #[test]
    fn encode_clear_is_clear_event_with_stream_sid() {
        let mut s = TwilioSerializer::new(8000);
        s.on_message(&WsIn::Text(
            json!({"event": "start", "start": {"streamSid": "MZ9"}}).to_string(),
        ));
        let WsOut::Text(text) = s.encode_clear().expect("twilio supports clear") else {
            panic!("expected Text")
        };
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["event"], "clear");
        assert_eq!(v["streamSid"], "MZ9");
    }

    #[test]
    fn carrier_rate_reflects_construction() {
        assert_eq!(TwilioSerializer::new(8000).carrier_rate(), 8000);
    }
}
