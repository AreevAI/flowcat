// SPDX-License-Identifier: Apache-2.0
//
//! Genesys Cloud **AudioHook** WebSocket serializer.
//!
//! AudioHook is more stateful than the streaming carriers: a protocol handshake
//! (`open`→`opened`, `ping`→`pong`, `close`→`closed`) is interleaved with
//! **binary G.711 μ-law (PCMU) @ 8 kHz** audio. Wire shapes are ported from the
//! vendored pipecat `GenesysAudioHookSerializer`
//! (`pipecat/src/pipecat/serializers/genesys.py`).
//!
//! The frozen [`MediaSerializer`] surface returns a single [`SerIn`] per inbound
//! frame and encodes audio/clear as a single [`WsOut`]. AudioHook also requires
//! the server to **emit protocol responses** (`opened`/`pong`/`closed`) — these
//! are not media, so they are queued and drained by the transport via
//! [`GenesysSerializer::take_responses`] after each `on_message`. This keeps the
//! media trait frozen while still speaking the full handshake.
//!
//! Inbound (Genesys → us):
//! - **binary** frame → μ-law audio (only while the session is open + not paused).
//! - text `{"version":"2","type":"open","seq":1,"id":"…","parameters":{…}}`
//!   → [`SerIn::StreamStart`] + queues an `opened` response.
//! - text `{"type":"ping",…}` → [`SerIn::Ignore`] + queues a `pong`.
//! - text `{"type":"close",…}` → [`SerIn::Stop`] + queues a `closed`.
//! - text `{"type":"pause"/"dtmf"/"update"/"error",…}` → [`SerIn::Ignore`].
//!
//! Outbound (us → Genesys):
//! - audio → **binary** μ-law bytes ([`WsOut::Binary`]).
//! - barge-in → a JSON `event` message carrying a `barge_in` entity.
//!
//! Pure framing only — no I/O, no panics on malformed wire data.

use serde_json::{json, Value};

use flowcat_core::codec::{pcm16_to_ulaw, ulaw_to_pcm16};
use flowcat_core::{AudioChunk, MediaSerializer, SerIn, WsIn, WsOut};

/// The AudioHook protocol version this serializer speaks.
const PROTOCOL_VERSION: &str = "2";

/// Serializer for the Genesys AudioHook WebSocket protocol.
#[derive(Debug)]
pub struct GenesysSerializer {
    rate: u32,
    session_id: Option<String>,
    is_open: bool,
    is_paused: bool,
    server_seq: u64,
    client_seq: u64,
    /// Protocol responses (`opened`/`pong`/`closed`) the transport must send.
    pending: Vec<WsOut>,
}

impl Default for GenesysSerializer {
    fn default() -> Self {
        Self::new(8000)
    }
}

impl GenesysSerializer {
    /// Create a Genesys AudioHook serializer at the given carrier sample rate
    /// (AudioHook PCMU is 8000).
    pub fn new(rate: u32) -> Self {
        Self {
            rate,
            session_id: None,
            is_open: false,
            is_paused: false,
            server_seq: 0,
            client_seq: 0,
            pending: Vec::new(),
        }
    }

    /// Whether the AudioHook session is currently open.
    pub fn is_open(&self) -> bool {
        self.is_open
    }

    /// Whether audio streaming is currently paused (Genesys hold).
    pub fn is_paused(&self) -> bool {
        self.is_paused
    }

    /// The AudioHook session id (from the `open` message), if seen.
    pub fn session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    /// Drain queued protocol responses (`opened`/`pong`/`closed`) that the
    /// transport must send back to Genesys. Call after each `on_message`.
    pub fn take_responses(&mut self) -> Vec<WsOut> {
        std::mem::take(&mut self.pending)
    }

    fn next_server_seq(&mut self) -> u64 {
        self.server_seq += 1;
        self.server_seq
    }

    /// Build a protocol message envelope with the common AudioHook fields.
    fn message(&mut self, msg_type: &str, parameters: Value) -> Value {
        let seq = self.next_server_seq();
        json!({
            "version": PROTOCOL_VERSION,
            "type": msg_type,
            "seq": seq,
            "clientseq": self.client_seq,
            "id": self.session_id.clone().unwrap_or_default(),
            "parameters": parameters,
        })
    }

    fn queue_opened(&mut self) {
        let params = json!({
            "startPaused": false,
            "media": [{
                "type": "audio",
                "format": "PCMU",
                "channels": ["external"],
                "rate": self.rate,
            }],
        });
        let msg = self.message("opened", params);
        self.pending.push(WsOut::Text(msg.to_string()));
    }

    fn queue_pong(&mut self) {
        let msg = self.message("pong", json!({}));
        self.pending.push(WsOut::Text(msg.to_string()));
    }

    fn queue_closed(&mut self) {
        let msg = self.message("closed", json!({}));
        self.pending.push(WsOut::Text(msg.to_string()));
    }
}

impl MediaSerializer for GenesysSerializer {
    fn on_message(&mut self, msg: &WsIn) -> SerIn {
        match msg {
            WsIn::Binary(bytes) => {
                if !self.is_open || self.is_paused {
                    return SerIn::Ignore;
                }
                SerIn::Audio(AudioChunk {
                    pcm: ulaw_to_pcm16(bytes),
                    sample_rate: self.rate,
                })
            }
            WsIn::Close => SerIn::Stop,
            WsIn::Text(text) => {
                let v: Value = match serde_json::from_str(text) {
                    Ok(v) => v,
                    Err(_) => return SerIn::Ignore,
                };
                // Track the client's sequence number for response envelopes.
                if let Some(seq) = v.get("seq").and_then(Value::as_u64) {
                    self.client_seq = seq;
                }
                match v.get("type").and_then(Value::as_str) {
                    Some("open") => {
                        let session_id = v.get("id").and_then(Value::as_str).map(str::to_owned);
                        let params = v.get("parameters").cloned().unwrap_or(Value::Null);
                        let conversation_id = params
                            .get("conversationId")
                            .and_then(Value::as_str)
                            .map(str::to_owned);
                        self.session_id = session_id.clone();
                        self.is_open = true;
                        self.queue_opened();
                        SerIn::StreamStart {
                            call_id: conversation_id
                                .or_else(|| session_id.clone())
                                .unwrap_or_default(),
                            stream_id: session_id,
                        }
                    }
                    Some("ping") => {
                        self.queue_pong();
                        SerIn::Ignore
                    }
                    Some("close") => {
                        self.is_open = false;
                        self.queue_closed();
                        SerIn::Stop
                    }
                    Some("pause") => {
                        self.is_paused = true;
                        SerIn::Ignore
                    }
                    Some("resume") => {
                        self.is_paused = false;
                        SerIn::Ignore
                    }
                    // dtmf / update / error / unknown: no media action at this layer.
                    _ => SerIn::Ignore,
                }
            }
        }
    }

    fn encode_audio(&self, chunk: &AudioChunk) -> WsOut {
        // AudioHook PCMU audio is sent as raw binary μ-law bytes.
        WsOut::Binary(pcm16_to_ulaw(&chunk.pcm))
    }

    fn encode_clear(&self) -> Option<WsOut> {
        // AudioHook barge-in is an `event` message with a `barge_in` entity.
        // (Built without &mut self, so it does not bump the server seq; the
        // transport may renumber if strict ordering is required.)
        let msg = json!({
            "version": PROTOCOL_VERSION,
            "type": "event",
            "id": self.session_id.clone().unwrap_or_default(),
            "parameters": { "entities": [{"type": "barge_in", "data": {}}] },
        });
        Some(WsOut::Text(msg.to_string()))
    }

    fn carrier_rate(&self) -> u32 {
        self.rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open_msg(id: &str) -> WsIn {
        WsIn::Text(
            json!({
                "version": "2",
                "type": "open",
                "seq": 1,
                "id": id,
                "parameters": {
                    "conversationId": "conv-1",
                    "participant": {"ani": "+1555", "dnis": "+1999"},
                    "media": [{"type": "audio", "format": "PCMU", "channels": ["external"], "rate": 8000}]
                }
            })
            .to_string(),
        )
    }

    #[test]
    fn open_starts_session_and_queues_opened() {
        let mut s = GenesysSerializer::new(8000);
        match s.on_message(&open_msg("sess-1")) {
            SerIn::StreamStart { call_id, stream_id } => {
                assert_eq!(call_id, "conv-1");
                assert_eq!(stream_id.as_deref(), Some("sess-1"));
            }
            other => panic!("expected StreamStart, got {other:?}"),
        }
        assert!(s.is_open());
        assert_eq!(s.session_id(), Some("sess-1"));
        let resp = s.take_responses();
        assert_eq!(resp.len(), 1);
        let WsOut::Text(text) = &resp[0] else {
            panic!("expected Text")
        };
        let v: Value = serde_json::from_str(text).unwrap();
        assert_eq!(v["type"], "opened");
        assert_eq!(v["clientseq"], 1);
        assert_eq!(v["parameters"]["media"][0]["format"], "PCMU");
    }

    #[test]
    fn audio_only_flows_while_open_and_not_paused() {
        let mut s = GenesysSerializer::new(8000);
        let ulaw = vec![0x10u8, 0x80, 0xFF];
        // Before open: ignored.
        assert!(matches!(
            s.on_message(&WsIn::Binary(ulaw.clone())),
            SerIn::Ignore
        ));
        s.on_message(&open_msg("s"));
        let _ = s.take_responses();
        match s.on_message(&WsIn::Binary(ulaw.clone())) {
            SerIn::Audio(c) => assert_eq!(c.pcm, ulaw_to_pcm16(&ulaw)),
            other => panic!("expected Audio, got {other:?}"),
        }
        // Pause → audio ignored.
        s.on_message(&WsIn::Text(json!({"type": "pause", "seq": 5}).to_string()));
        assert!(matches!(
            s.on_message(&WsIn::Binary(ulaw.clone())),
            SerIn::Ignore
        ));
        // Resume → audio flows again.
        s.on_message(&WsIn::Text(json!({"type": "resume", "seq": 6}).to_string()));
        assert!(matches!(s.on_message(&WsIn::Binary(ulaw)), SerIn::Audio(_)));
    }

    #[test]
    fn ping_queues_pong_and_is_ignored() {
        let mut s = GenesysSerializer::new(8000);
        s.on_message(&open_msg("s"));
        let _ = s.take_responses();
        assert!(matches!(
            s.on_message(&WsIn::Text(json!({"type": "ping", "seq": 7}).to_string())),
            SerIn::Ignore
        ));
        let resp = s.take_responses();
        assert_eq!(resp.len(), 1);
        let WsOut::Text(text) = &resp[0] else {
            panic!("expected Text")
        };
        let v: Value = serde_json::from_str(text).unwrap();
        assert_eq!(v["type"], "pong");
        assert_eq!(v["clientseq"], 7);
    }

    #[test]
    fn close_stops_and_queues_closed() {
        let mut s = GenesysSerializer::new(8000);
        s.on_message(&open_msg("s"));
        let _ = s.take_responses();
        assert!(matches!(
            s.on_message(&WsIn::Text(json!({"type": "close", "seq": 9}).to_string())),
            SerIn::Stop
        ));
        assert!(!s.is_open());
        let resp = s.take_responses();
        let WsOut::Text(text) = &resp[0] else {
            panic!("expected Text")
        };
        let v: Value = serde_json::from_str(text).unwrap();
        assert_eq!(v["type"], "closed");
    }

    #[test]
    fn dtmf_and_malformed_are_ignored() {
        let mut s = GenesysSerializer::new(8000);
        s.on_message(&open_msg("s"));
        let _ = s.take_responses();
        assert!(matches!(
            s.on_message(&WsIn::Text(
                json!({"type": "dtmf", "seq": 3, "parameters": {"digit": "1"}}).to_string()
            )),
            SerIn::Ignore
        ));
        assert!(matches!(
            s.on_message(&WsIn::Text("not json".into())),
            SerIn::Ignore
        ));
        // No protocol responses queued for these.
        assert!(s.take_responses().is_empty());
    }

    #[test]
    fn close_ws_frame_is_stop() {
        let mut s = GenesysSerializer::new(8000);
        assert!(matches!(s.on_message(&WsIn::Close), SerIn::Stop));
    }

    #[test]
    fn encode_audio_is_binary_ulaw() {
        let s = GenesysSerializer::new(8000);
        let pcm = vec![0i16, 200, -200, 15000];
        let WsOut::Binary(bytes) = s.encode_audio(&AudioChunk {
            pcm: pcm.clone(),
            sample_rate: 8000,
        }) else {
            panic!("expected Binary")
        };
        assert_eq!(bytes, pcm16_to_ulaw(&pcm));
    }

    #[test]
    fn encode_clear_is_barge_in_event() {
        let mut s = GenesysSerializer::new(8000);
        s.on_message(&open_msg("s"));
        let _ = s.take_responses();
        let WsOut::Text(text) = s.encode_clear().unwrap() else {
            panic!("expected Text")
        };
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["type"], "event");
        assert_eq!(v["parameters"]["entities"][0]["type"], "barge_in");
    }

    #[test]
    fn carrier_rate_reflects_construction() {
        assert_eq!(GenesysSerializer::new(8000).carrier_rate(), 8000);
    }
}
