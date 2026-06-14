// SPDX-License-Identifier: Apache-2.0
//
//! `WsCarrierTransport` — adapts the WS path to the [`MediaTransport`] seam.
//!
//! It composes the two lower WS seams — a [`MediaSocket`] (raw text/binary
//! frames) and a [`MediaSerializer`] (pure per-carrier framing) — into the
//! decoded-audio [`MediaTransport`] the pipeline drives. This is how the Plivo
//! path plugs into the transport-agnostic [S2S pipeline](crate::pipeline::build_s2s_task):
//! the embedder builds `WsCarrierTransport::new(AxumWsSocket, PlivoSerializer)`.
//!
//! See SIP-DESIGN.md §1 "WS path preserved via an adapter".

use async_trait::async_trait;

use crate::error::FlowcatError;
use crate::serializer::MediaSerializer;
use crate::transport::media::{MediaIn, MediaTransport};
use crate::transport::socket::MediaSocket;
use crate::types::{AudioChunk, SerIn, WsOut};

/// A [`MediaTransport`] built from a raw [`MediaSocket`] plus a per-carrier
/// [`MediaSerializer`].
///
/// `recv` pulls socket frames and runs them through the serializer until a
/// non-[`SerIn::Ignore`] event appears, then maps it to a [`MediaIn`];
/// `send_audio`/`send_clear` go the other way through the serializer's
/// `encode_audio`/`encode_clear` and out the socket; `carrier_rate` is the
/// serializer's.
pub struct WsCarrierTransport<So, Se>
where
    So: MediaSocket,
    Se: MediaSerializer,
{
    socket: So,
    serializer: Se,
}

impl<So, Se> WsCarrierTransport<So, Se>
where
    So: MediaSocket,
    Se: MediaSerializer,
{
    /// Wrap a socket + serializer as a single [`MediaTransport`].
    pub fn new(socket: So, serializer: Se) -> Self {
        Self { socket, serializer }
    }

    /// Send one already-encoded [`WsOut`] over the socket, mapping it to the
    /// matching frame kind.
    async fn send_out(&mut self, out: WsOut) -> Result<(), FlowcatError> {
        match out {
            WsOut::Text(s) => self.socket.send_text(s).await,
            WsOut::Binary(b) => self.socket.send_binary(b).await,
        }
    }
}

#[async_trait]
impl<So, Se> MediaTransport for WsCarrierTransport<So, Se>
where
    So: MediaSocket,
    Se: MediaSerializer,
{
    async fn recv(&mut self) -> Option<MediaIn> {
        // Loop until the serializer yields something actionable: keepalives,
        // mark-acks, and unparseable frames map to `SerIn::Ignore`, which we
        // skip so the pipeline only ever sees StreamStart / Audio / Stop.
        loop {
            // `None` from the socket = exhausted: signal end-of-stream. (A peer
            // `Close` frame surfaces as `WsIn::Close`, which the serializer maps
            // to `SerIn::Stop` below — that path returns `Stop`, not `None`.)
            let frame = self.socket.recv().await?;
            match self.serializer.on_message(&frame) {
                SerIn::StreamStart { call_id, .. } => {
                    return Some(MediaIn::StreamStart { call_id });
                }
                SerIn::Audio(chunk) => return Some(MediaIn::Audio(chunk)),
                SerIn::Stop => return Some(MediaIn::Stop),
                SerIn::Ignore => continue,
            }
        }
    }

    async fn send_audio(&mut self, chunk: AudioChunk) -> Result<(), FlowcatError> {
        let out = self.serializer.encode_audio(&chunk);
        self.send_out(out).await
    }

    async fn send_clear(&mut self) -> Result<(), FlowcatError> {
        // Some carriers have no clear/flush frame; the serializer returns `None`
        // in that case and barge-in is a no-op here.
        if let Some(out) = self.serializer.encode_clear() {
            self.send_out(out).await
        } else {
            Ok(())
        }
    }

    fn carrier_rate(&self) -> u32 {
        self.serializer.carrier_rate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serializer::PlivoSerializer;
    use crate::types::{WsIn, WsOut};
    use serde_json::{json, Value};
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    fn b64(bytes: &[u8]) -> String {
        use base64::Engine as _;
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    /// A scripted [`MediaSocket`]: drains a queue of inbound frames (then `None`)
    /// and captures every outbound frame for assertions.
    struct ScriptSocket {
        inbound: VecDeque<WsIn>,
        sent: Arc<Mutex<Vec<WsOut>>>,
    }

    impl ScriptSocket {
        fn new(inbound: Vec<WsIn>, sent: Arc<Mutex<Vec<WsOut>>>) -> Self {
            Self {
                inbound: inbound.into(),
                sent,
            }
        }
    }

    #[async_trait]
    impl MediaSocket for ScriptSocket {
        async fn recv(&mut self) -> Option<WsIn> {
            self.inbound.pop_front()
        }
        async fn send_text(&mut self, s: String) -> Result<(), FlowcatError> {
            self.sent.lock().unwrap().push(WsOut::Text(s));
            Ok(())
        }
        async fn send_binary(&mut self, b: Vec<u8>) -> Result<(), FlowcatError> {
            self.sent.lock().unwrap().push(WsOut::Binary(b));
            Ok(())
        }
    }

    /// A Plivo start/media/stop sequence maps to StreamStart → Audio → Stop,
    /// and the trailing exhausted socket yields `None`. Keepalive-ish frames in
    /// between (here a `dtmf` event → `SerIn::Ignore`) are transparently skipped.
    #[tokio::test]
    async fn recv_maps_plivo_start_media_stop_and_skips_ignore() {
        let pcm: Vec<i16> = (0..160).map(|i| ((i as i16 % 16) * 300) - 2400).collect();
        let ulaw = crate::codec::pcm16_to_ulaw(&pcm);
        let inbound = vec![
            WsIn::Text(
                json!({
                    "event": "start",
                    "start": { "streamId": "strm-1", "callId": "call-9",
                               "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000} }
                })
                .to_string(),
            ),
            // An ignorable frame between start and media — must be skipped, not
            // surfaced (proves the recv() loop drops `SerIn::Ignore`).
            WsIn::Text(json!({ "event": "dtmf", "dtmf": { "digit": "5" } }).to_string()),
            WsIn::Text(json!({ "event": "media", "media": { "payload": b64(&ulaw) } }).to_string()),
            WsIn::Text(json!({ "event": "stop" }).to_string()),
        ];
        let sent = Arc::new(Mutex::new(Vec::new()));
        let mut tr = WsCarrierTransport::new(
            ScriptSocket::new(inbound, sent.clone()),
            PlivoSerializer::new(8000),
        );

        assert_eq!(tr.carrier_rate(), 8000);

        // start → StreamStart{call_id} (stream_id is dropped at this seam).
        match tr.recv().await {
            Some(MediaIn::StreamStart { call_id }) => assert_eq!(call_id, "call-9"),
            other => panic!("expected StreamStart, got {other:?}"),
        }
        // dtmf was skipped; next actionable event is the decoded audio @ 8000.
        match tr.recv().await {
            Some(MediaIn::Audio(chunk)) => {
                assert_eq!(chunk.sample_rate, 8000);
                assert_eq!(chunk.pcm.len(), ulaw.len());
            }
            other => panic!("expected Audio, got {other:?}"),
        }
        // stop → Stop.
        assert_eq!(tr.recv().await, Some(MediaIn::Stop));
        // socket exhausted → None.
        assert_eq!(tr.recv().await, None);
    }

    /// A peer `Close` frame (not an explicit `stop` event) still maps to `Stop`
    /// via the serializer — the loop must not turn it into `None`.
    #[tokio::test]
    async fn recv_maps_socket_close_to_stop() {
        let sent = Arc::new(Mutex::new(Vec::new()));
        let mut tr = WsCarrierTransport::new(
            ScriptSocket::new(vec![WsIn::Close], sent.clone()),
            PlivoSerializer::new(8000),
        );
        assert_eq!(tr.recv().await, Some(MediaIn::Stop));
    }

    /// `send_audio` encodes through the serializer into a Plivo `playAudio` text
    /// frame; `send_clear` emits a `clearAudio` frame (Plivo supports it).
    #[tokio::test]
    async fn send_audio_and_clear_go_out_through_the_serializer() {
        // Prime the serializer with a streamId so outbound frames echo it.
        let start = WsIn::Text(
            json!({ "event": "start", "start": { "streamId": "S7", "callId": "C7" } }).to_string(),
        );
        let sent = Arc::new(Mutex::new(Vec::new()));
        let mut tr = WsCarrierTransport::new(
            ScriptSocket::new(vec![start], sent.clone()),
            PlivoSerializer::new(8000),
        );
        // Consume the start so the serializer learns the streamId.
        let _ = tr.recv().await;

        tr.send_audio(AudioChunk::new(vec![0i16, 100, -100, 200], 8000))
            .await
            .unwrap();
        tr.send_clear().await.unwrap();

        let out = sent.lock().unwrap();
        assert_eq!(out.len(), 2, "one playAudio + one clearAudio");
        let play = match &out[0] {
            WsOut::Text(t) => serde_json::from_str::<Value>(t).unwrap(),
            other => panic!("expected Text playAudio, got {other:?}"),
        };
        assert_eq!(play["event"], "playAudio");
        assert_eq!(play["media"]["contentType"], "audio/x-mulaw");
        assert_eq!(play["streamId"], "S7");

        let clear = match &out[1] {
            WsOut::Text(t) => serde_json::from_str::<Value>(t).unwrap(),
            other => panic!("expected Text clearAudio, got {other:?}"),
        };
        assert_eq!(clear["event"], "clearAudio");
        assert_eq!(clear["streamId"], "S7");
    }
}
