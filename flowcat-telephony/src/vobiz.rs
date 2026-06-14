// SPDX-License-Identifier: Apache-2.0
//
//! Vobiz Media Streams WebSocket serializer.
//!
//! Vobiz uses a Plivo-style protocol: JSON **text** frames, base64 G.711 μ-law @
//! 8 kHz, outbound `playAudio` echoing `streamId`, and a `clearAudio` event for
//! barge-in. Wire shapes are ported from the vendored pipecat
//! `VobizFrameSerializer` (`pipecat/src/pipecat/serializers/vobiz.py`).
//!
//! Inbound (Vobiz → us):
//! ```json
//! {"event":"start","start":{"streamId":"…","callId":"…"}}
//! {"event":"media","media":{"payload":"<base64 μ-law>"}}
//! {"event":"dtmf","dtmf":{"digit":"1"}}
//! {"event":"stop"}
//! ```
//! Outbound (us → Vobiz):
//! ```json
//! {"event":"playAudio","media":{"contentType":"audio/x-mulaw","sampleRate":8000,
//!   "payload":"<base64 μ-law>"},"streamId":"…"}
//! {"event":"clearAudio","streamId":"…"}      // barge-in / interruption
//! ```
//!
//! Pure framing only — no I/O, no panics on malformed wire data.

use base64::Engine;
use serde_json::{json, Value};

use flowcat_core::codec::{pcm16_to_ulaw, ulaw_to_pcm16};
use flowcat_core::{AudioChunk, MediaSerializer, SerIn, WsIn, WsOut};

/// Serializer for Vobiz's Media Streams WebSocket protocol.
#[derive(Debug, Default)]
pub struct VobizSerializer {
    rate: u32,
    stream_id: Option<String>,
}

impl VobizSerializer {
    /// Create a Vobiz serializer at the given carrier sample rate (typically 8000).
    pub fn new(rate: u32) -> Self {
        Self {
            rate,
            stream_id: None,
        }
    }

    /// The `streamId` learned from the carrier's `start` frame, if seen.
    pub fn stream_id(&self) -> Option<&str> {
        self.stream_id.as_deref()
    }
}

impl MediaSerializer for VobizSerializer {
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
                    .get("streamId")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                let call_id = start
                    .get("callId")
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
            "event": "playAudio",
            "media": {
                "contentType": "audio/x-mulaw",
                "sampleRate": self.rate,
                "payload": payload,
            },
        });
        if let Some(sid) = &self.stream_id {
            answer["streamId"] = Value::String(sid.clone());
        }
        WsOut::Text(answer.to_string())
    }

    fn encode_clear(&self) -> Option<WsOut> {
        // pipecat: `{"event":"clearAudio","streamId": <stream_id>}` on interruption.
        let mut answer = json!({ "event": "clearAudio" });
        if let Some(sid) = &self.stream_id {
            answer["streamId"] = Value::String(sid.clone());
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
    fn start_media_stop() {
        let mut s = VobizSerializer::new(8000);
        match s.on_message(&WsIn::Text(
            json!({"event": "start", "start": {"streamId": "vs1", "callId": "vc1"}}).to_string(),
        )) {
            SerIn::StreamStart { call_id, stream_id } => {
                assert_eq!(call_id, "vc1");
                assert_eq!(stream_id.as_deref(), Some("vs1"));
            }
            other => panic!("expected StreamStart, got {other:?}"),
        }
        let ulaw_in = vec![0x20u8, 0x80, 0xFE];
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
        assert!(matches!(s.on_message(&WsIn::Close), SerIn::Stop));
    }

    #[test]
    fn malformed_is_ignored() {
        let mut s = VobizSerializer::new(8000);
        assert!(matches!(
            s.on_message(&WsIn::Text("x".into())),
            SerIn::Ignore
        ));
        assert!(matches!(
            s.on_message(&WsIn::Binary(vec![0])),
            SerIn::Ignore
        ));
        assert!(matches!(
            s.on_message(&WsIn::Text(
                json!({"event": "media", "media": {"payload": "&&"}}).to_string()
            )),
            SerIn::Ignore
        ));
    }

    #[test]
    fn encode_audio_is_play_audio_with_stream_id() {
        let mut s = VobizSerializer::new(8000);
        s.on_message(&WsIn::Text(
            json!({"event": "start", "start": {"streamId": "vs9"}}).to_string(),
        ));
        let pcm = vec![3i16, -3, 9000];
        let WsOut::Text(text) = s.encode_audio(&AudioChunk {
            pcm: pcm.clone(),
            sample_rate: 8000,
        }) else {
            panic!("expected Text")
        };
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["event"], "playAudio");
        assert_eq!(v["media"]["contentType"], "audio/x-mulaw");
        assert_eq!(v["streamId"], "vs9");
        let ulaw = base64::engine::general_purpose::STANDARD
            .decode(v["media"]["payload"].as_str().unwrap())
            .unwrap();
        assert_eq!(ulaw, pcm16_to_ulaw(&pcm));
    }

    #[test]
    fn encode_clear_is_clear_audio() {
        let mut s = VobizSerializer::new(8000);
        s.on_message(&WsIn::Text(
            json!({"event": "start", "start": {"streamId": "vs2"}}).to_string(),
        ));
        let WsOut::Text(text) = s.encode_clear().unwrap() else {
            panic!("expected Text")
        };
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["event"], "clearAudio");
        assert_eq!(v["streamId"], "vs2");
    }

    #[test]
    fn carrier_rate_reflects_construction() {
        assert_eq!(VobizSerializer::new(8000).carrier_rate(), 8000);
    }
}
