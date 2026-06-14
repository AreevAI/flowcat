// SPDX-License-Identifier: Apache-2.0
//
//! Asterisk ARI (`chan_websocket` / externalMedia) WebSocket serializer.
//!
//! Asterisk streams **raw G.711 μ-law (PCMU) @ 8 kHz as binary** WebSocket
//! frames — no JSON wrapper, no base64. Wire shapes are ported from the vendored
//! pipecat `AsteriskFrameSerializer`
//! (`pipecat/src/pipecat/serializers/asterisk.py`).
//!
//! Inbound (Asterisk → us):
//! - **binary** frame → raw μ-law audio bytes.
//! - text JSON control events (if any) → ignored at this layer.
//!
//! Outbound (us → Asterisk):
//! - audio → **binary** μ-law bytes ([`WsOut::Binary`]).
//! - **no buffer-clear command exists** over the audio WS, so [`encode_clear`]
//!   returns `None` (the transport simply stops sending audio on barge-in).
//!
//! Pure framing only — no I/O, no panics on malformed wire data.

use serde_json::Value;

use flowcat_core::codec::{pcm16_to_ulaw, ulaw_to_pcm16};
use flowcat_core::{AudioChunk, MediaSerializer, SerIn, WsIn, WsOut};

/// Serializer for Asterisk's ARI binary-μ-law WebSocket audio.
#[derive(Debug, Default)]
pub struct AsteriskSerializer {
    rate: u32,
}

impl AsteriskSerializer {
    /// Create an Asterisk serializer at the given carrier sample rate (typically 8000).
    pub fn new(rate: u32) -> Self {
        Self { rate }
    }
}

impl MediaSerializer for AsteriskSerializer {
    fn on_message(&mut self, msg: &WsIn) -> SerIn {
        match msg {
            // Binary frame = raw μ-law audio bytes. Asterisk has no JSON start
            // event over this socket, so the transport owns session start; here
            // we just decode media.
            WsIn::Binary(bytes) => SerIn::Audio(AudioChunk {
                pcm: ulaw_to_pcm16(bytes),
                sample_rate: self.rate,
            }),
            WsIn::Close => SerIn::Stop,
            // Text control events (ARI status JSON) carry no media; ignore safely.
            WsIn::Text(text) => match serde_json::from_str::<Value>(text) {
                Ok(_) => SerIn::Ignore,
                Err(_) => SerIn::Ignore,
            },
        }
    }

    fn encode_audio(&self, chunk: &AudioChunk) -> WsOut {
        // Asterisk expects raw binary μ-law bytes (no JSON, no base64).
        WsOut::Binary(pcm16_to_ulaw(&chunk.pcm))
    }

    fn encode_clear(&self) -> Option<WsOut> {
        // Asterisk has no playback-clear command over the audio WS.
        None
    }

    fn carrier_rate(&self) -> u32 {
        self.rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn binary_frame_decodes_ulaw_to_pcm() {
        let mut s = AsteriskSerializer::new(8000);
        let ulaw = vec![0x10u8, 0x80, 0xFF, 0x00];
        match s.on_message(&WsIn::Binary(ulaw.clone())) {
            SerIn::Audio(c) => {
                assert_eq!(c.sample_rate, 8000);
                assert_eq!(c.pcm, ulaw_to_pcm16(&ulaw));
            }
            other => panic!("expected Audio, got {other:?}"),
        }
    }

    #[test]
    fn text_control_event_is_ignored() {
        let mut s = AsteriskSerializer::new(8000);
        assert!(matches!(
            s.on_message(&WsIn::Text(
                json!({"type": "ChannelStateChange", "channel": {"id": "c1"}}).to_string()
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
        let mut s = AsteriskSerializer::new(8000);
        assert!(matches!(s.on_message(&WsIn::Close), SerIn::Stop));
    }

    #[test]
    fn encode_audio_is_binary_ulaw_roundtrip() {
        let s = AsteriskSerializer::new(8000);
        let pcm = vec![0i16, 100, -100, 20000, -20000];
        let WsOut::Binary(bytes) = s.encode_audio(&AudioChunk {
            pcm: pcm.clone(),
            sample_rate: 8000,
        }) else {
            panic!("expected Binary")
        };
        assert_eq!(bytes, pcm16_to_ulaw(&pcm));
        assert_eq!(ulaw_to_pcm16(&bytes).len(), pcm.len());
    }

    #[test]
    fn encode_clear_is_none() {
        let s = AsteriskSerializer::new(8000);
        assert!(s.encode_clear().is_none());
    }

    #[test]
    fn carrier_rate_reflects_construction() {
        assert_eq!(AsteriskSerializer::new(8000).carrier_rate(), 8000);
    }
}
