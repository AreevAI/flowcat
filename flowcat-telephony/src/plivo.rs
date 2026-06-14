// SPDX-License-Identifier: Apache-2.0
//
//! Plivo `<Stream>` WebSocket serializer (ported from
//! `flowcat_core::serializer::plivo`).
//!
//! Plivo's audio-stream protocol is JSON **text** frames in both directions; the
//! audio payload is base64 G.711 μ-law @ 8 kHz. Wire shapes are ported from the
//! vendored pipecat `PlivoFrameSerializer`
//! (`pipecat/src/pipecat/serializers/plivo.py`).
//!
//! Inbound (Plivo → us):
//! ```json
//! {"event":"start","start":{"streamId":"…","callId":"…",
//!   "mediaFormat":{"encoding":"audio/x-mulaw","sampleRate":8000}}}
//! {"event":"media","media":{"payload":"<base64 μ-law>"}}
//! {"event":"stop"}
//! ```
//! Outbound (us → Plivo):
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

/// Serializer for Plivo's audio-stream WebSocket protocol.
#[derive(Debug, Default)]
pub struct PlivoSerializer {
    /// Carrier sample rate (Plivo telephony μ-law = 8000).
    rate: u32,
    /// The Plivo `streamId`, learned from the `start` frame. Plivo's outbound
    /// `playAudio`/`clearAudio` frames echo it back (pipecat does the same).
    stream_id: Option<String>,
}

impl PlivoSerializer {
    /// Create a Plivo serializer at the given carrier sample rate (typically 8000).
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

impl MediaSerializer for PlivoSerializer {
    fn on_message(&mut self, msg: &WsIn) -> SerIn {
        // Plivo speaks JSON over text frames; binary/close carry nothing here.
        let text = match msg {
            WsIn::Text(t) => t,
            WsIn::Close => return SerIn::Stop,
            WsIn::Binary(_) => return SerIn::Ignore,
        };

        let v: Value = match serde_json::from_str(text) {
            Ok(v) => v,
            // A malformed/non-JSON text frame is not actionable; don't panic.
            Err(_) => return SerIn::Ignore,
        };

        match v.get("event").and_then(Value::as_str) {
            Some("start") => {
                let start = v.get("start").cloned().unwrap_or(Value::Null);
                let stream_id = start
                    .get("streamId")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                // `callId` is the carrier call id; fall back to streamId so we
                // always surface *some* identifier to the pipeline.
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
            // `dtmf`, `mark`/checkpoint acks, clears, unknown events: no action.
            _ => SerIn::Ignore,
        }
    }

    fn encode_audio(&self, chunk: &AudioChunk) -> WsOut {
        // Outbound to Plivo is μ-law @ the carrier rate. The pipeline is
        // expected to hand us a chunk already at `self.rate`; we encode as-is
        // (resampling is the codec/pipeline's job, mirroring pipecat which
        // resamples before serializing).
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
        // Echo the streamId when known (pipecat always sets it).
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
    fn start_event_parses_to_stream_start_with_ids() {
        let mut s = PlivoSerializer::new(8000);
        let frame = WsIn::Text(
            json!({
                "event": "start",
                "start": {
                    "streamId": "strm-123",
                    "callId": "call-abc",
                    "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000}
                }
            })
            .to_string(),
        );
        match s.on_message(&frame) {
            SerIn::StreamStart { call_id, stream_id } => {
                assert_eq!(call_id, "call-abc");
                assert_eq!(stream_id.as_deref(), Some("strm-123"));
            }
            other => panic!("expected StreamStart, got {other:?}"),
        }
        assert_eq!(s.stream_id(), Some("strm-123"));
    }

    #[test]
    fn media_event_base64_ulaw_decodes_to_pcm() {
        let mut s = PlivoSerializer::new(8000);
        let ulaw_in = vec![0xFFu8, 0x00, 0x80, 0x7F];
        let expected_pcm = ulaw_to_pcm16(&ulaw_in);
        let frame =
            WsIn::Text(json!({"event": "media", "media": {"payload": b64(&ulaw_in)}}).to_string());
        match s.on_message(&frame) {
            SerIn::Audio(chunk) => {
                assert_eq!(chunk.sample_rate, 8000);
                assert_eq!(chunk.pcm.len(), ulaw_in.len());
                assert_eq!(chunk.pcm, expected_pcm);
            }
            other => panic!("expected Audio, got {other:?}"),
        }
    }

    #[test]
    fn stop_event_and_close_map_to_stop() {
        let mut s = PlivoSerializer::new(8000);
        assert!(matches!(
            s.on_message(&WsIn::Text(json!({"event": "stop"}).to_string())),
            SerIn::Stop
        ));
        assert!(matches!(s.on_message(&WsIn::Close), SerIn::Stop));
    }

    #[test]
    fn unknown_and_malformed_frames_are_ignored() {
        let mut s = PlivoSerializer::new(8000);
        assert!(matches!(
            s.on_message(&WsIn::Text(json!({"event": "dtmf"}).to_string())),
            SerIn::Ignore
        ));
        assert!(matches!(
            s.on_message(&WsIn::Text("not json".to_string())),
            SerIn::Ignore
        ));
        assert!(matches!(
            s.on_message(&WsIn::Binary(vec![1, 2, 3])),
            SerIn::Ignore
        ));
        // media with a non-base64 payload must not panic.
        assert!(matches!(
            s.on_message(&WsIn::Text(
                json!({"event": "media", "media": {"payload": "!!!not-b64!!!"}}).to_string()
            )),
            SerIn::Ignore
        ));
    }

    #[test]
    fn encode_audio_has_plivo_keys_and_round_trippable_payload() {
        let mut s = PlivoSerializer::new(8000);
        s.on_message(&WsIn::Text(
            json!({"event": "start", "start": {"streamId": "S1", "callId": "C1"}}).to_string(),
        ));

        let pcm = vec![0i16, 100, -100, 32000, -32000];
        let out = s.encode_audio(&AudioChunk {
            pcm: pcm.clone(),
            sample_rate: 8000,
        });
        let text = match out {
            WsOut::Text(t) => t,
            other => panic!("expected Text, got {other:?}"),
        };
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["event"], "playAudio");
        assert_eq!(v["media"]["contentType"], "audio/x-mulaw");
        assert_eq!(v["media"]["sampleRate"], 8000);
        assert_eq!(v["streamId"], "S1");

        let payload = v["media"]["payload"].as_str().unwrap();
        let ulaw = base64::engine::general_purpose::STANDARD
            .decode(payload)
            .unwrap();
        assert_eq!(ulaw, pcm16_to_ulaw(&pcm));
        assert_eq!(ulaw_to_pcm16(&ulaw).len(), pcm.len());
    }

    #[test]
    fn encode_clear_is_plivo_clear_audio() {
        let mut s = PlivoSerializer::new(8000);
        s.on_message(&WsIn::Text(
            json!({"event": "start", "start": {"streamId": "S9"}}).to_string(),
        ));
        let out = s.encode_clear().expect("plivo supports clearAudio");
        let text = match out {
            WsOut::Text(t) => t,
            other => panic!("expected Text, got {other:?}"),
        };
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["event"], "clearAudio");
        assert_eq!(v["streamId"], "S9");
    }

    #[test]
    fn carrier_rate_reflects_construction() {
        assert_eq!(PlivoSerializer::new(8000).carrier_rate(), 8000);
    }
}
