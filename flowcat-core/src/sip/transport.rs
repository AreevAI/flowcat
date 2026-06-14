// SPDX-License-Identifier: Apache-2.0
//
//! `SipTransport` — bridges one SIP dialog's RTP media to the
//! [`MediaTransport`](crate::transport::MediaTransport) seam.
//!
//! Once the agent has negotiated a call (answered an inbound INVITE or got a 200
//! OK to an outbound one), it hands `SipTransport` a **bound RTP `UdpSocket`**,
//! the negotiated **peer address + G.711 codec**, the **Call-ID**, and a hangup
//! signal. From there the pipeline drives it exactly like the Plivo transport:
//!
//! - [`recv`](MediaTransport::recv) yields [`MediaIn::StreamStart`] once, then a
//!   stream of [`MediaIn::Audio`] at 8 kHz (inbound RTP → depacketize → jitter
//!   buffer → G.711 decode), then [`MediaIn::Stop`] on BYE / timeout / socket close.
//! - [`send_audio`](MediaTransport::send_audio) takes 8 kHz PCM (the pipeline has
//!   already resampled 24 kHz Gemini-out → carrier_rate), G.711-encodes it, and
//!   emits RTP packets **paced at 20 ms / 160-sample frames**.
//! - [`send_clear`](MediaTransport::send_clear) is a no-op (RTP has no flush;
//!   barge-in is simply "stop feeding playout" — handled upstream).
//! - [`carrier_rate`](MediaTransport::carrier_rate) is `8000`.
//!
//! ## Concurrency
//!
//! Two background tasks own the socket halves:
//! - **RX task**: reads datagrams, learns the symmetric-RTP peer from the first
//!   inbound packet, drops packets from any other source, depacketizes, runs the
//!   jitter buffer, decodes per the negotiated codec, and forwards 20 ms
//!   `AudioChunk`s over a channel that `recv` reads.
//! - **TX pump**: drains a channel of PCM the pipeline hands to `send_audio`,
//!   slices it into 160-sample (20 ms) frames, and sends one RTP packet every
//!   20 ms (a `tokio::time::interval`), so outbound pacing is correct regardless
//!   of how bursty the upstream resampler output is.
//!
//! Both tasks stop when the hangup [`CancellationToken`] fires (agent saw a BYE /
//! the dialog terminated) or the channel/socket closes.

use std::net::SocketAddr;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::net::UdpSocket;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::error::FlowcatError;
use crate::sip::rtp::{depacketize, JitterBuffer, RtpSender};
use crate::sip::sdp::G711Codec;
use crate::transport::media::{MediaIn, MediaTransport};
use crate::types::AudioChunk;

/// The carrier sample rate for telephony G.711 (8 kHz).
const CARRIER_RATE: u32 = 8000;
/// Samples per 20 ms G.711 frame at 8 kHz (also the byte count, 1 sample/byte).
const FRAME_SAMPLES: usize = 160;
/// The fixed RTP frame interval.
const FRAME_INTERVAL: std::time::Duration = std::time::Duration::from_millis(20);
/// Leading comfort-silence frames the TX pump sends before the first real audio,
/// to prime the far-end jitter buffer (see [`tx_loop`]). A freshly-answered RTP
/// receiver commonly discards the first packets of a fresh talkspurt while its
/// jitter buffer locks on, which would otherwise eat the agent's opening
/// syllable. 10 frames × 20 ms ≈ 200 ms.
const PREROLL_FRAMES: usize = 10;
/// Max UDP datagram we read for an RTP packet (well above a 160-byte G.711 frame).
const RTP_RECV_BUF: usize = 2048;
/// Bound on the inbound decoded-audio channel (frames). ~1 s of 20 ms frames;
/// generous so a momentarily slow consumer doesn't drop media, bounded so a
/// stalled consumer can't grow memory without limit.
const INBOUND_CHAN_DEPTH: usize = 50;
/// Bound on the outbound PCM channel (chunks the pipeline pushes to `send_audio`).
const OUTBOUND_CHAN_DEPTH: usize = 64;

/// A [`MediaTransport`] for one SIP dialog's RTP media leg.
///
/// Construct via [`SipTransport::start`] from a bound RTP socket + the negotiated
/// session parameters. See the module docs for the task/channel layout.
pub struct SipTransport {
    /// Carrier-side Call-ID, surfaced once as [`MediaIn::StreamStart`].
    call_id: String,
    /// Set after the first `recv()` so `StreamStart` is emitted exactly once.
    started: bool,
    /// Inbound decoded 8 kHz audio frames from the RX task.
    inbound_rx: mpsc::Receiver<AudioChunk>,
    /// Outbound PCM to the TX pump (`None` once dropped / shut down).
    outbound_tx: Option<mpsc::Sender<Vec<i16>>>,
    /// Fires to tear down both tasks (agent saw BYE / dialog terminated, or drop).
    hangup: CancellationToken,
    /// RX / TX task handles, aborted on drop for clean teardown.
    rx_task: tokio::task::JoinHandle<()>,
    tx_task: tokio::task::JoinHandle<()>,
}

impl SipTransport {
    /// Start the RTP bridge for an established dialog.
    ///
    /// - `socket` — an already-bound RTP `UdpSocket` (the local media port we put
    ///   in our SDP). Its local addr is the one the peer sends to.
    /// - `peer` — the peer's RTP address from their SDP. Outbound packets go here;
    ///   inbound packets from any *other* source are dropped (symmetric RTP). If
    ///   the very first inbound packet comes from a slightly different port (NAT),
    ///   the RX task re-learns the peer from it.
    /// - `codec` — the negotiated G.711 codec (PCMU/PCMA), fixing the RTP payload
    ///   type and the decode/encode functions.
    /// - `call_id` — surfaced as [`MediaIn::StreamStart`].
    /// - `hangup` — cancelled by the agent when the dialog terminates (BYE/timeout)
    ///   to drive [`MediaIn::Stop`]; also fired on `Drop`.
    pub fn start(
        socket: UdpSocket,
        peer: SocketAddr,
        codec: G711Codec,
        call_id: String,
        hangup: CancellationToken,
    ) -> Self {
        let socket = Arc::new(socket);
        let (inbound_tx, inbound_rx) = mpsc::channel::<AudioChunk>(INBOUND_CHAN_DEPTH);
        let (outbound_tx, outbound_rx) = mpsc::channel::<Vec<i16>>(OUTBOUND_CHAN_DEPTH);

        let rx_task = tokio::spawn(rx_loop(
            socket.clone(),
            peer,
            codec,
            inbound_tx,
            hangup.clone(),
        ));
        let tx_task = tokio::spawn(tx_loop(socket, peer, codec, outbound_rx, hangup.clone()));

        Self {
            call_id,
            started: false,
            inbound_rx,
            outbound_tx: Some(outbound_tx),
            hangup,
            rx_task,
            tx_task,
        }
    }
}

impl Drop for SipTransport {
    fn drop(&mut self) {
        // Ensure the media tasks can't outlive the transport.
        self.hangup.cancel();
        self.rx_task.abort();
        self.tx_task.abort();
    }
}

#[async_trait]
impl MediaTransport for SipTransport {
    async fn recv(&mut self) -> Option<MediaIn> {
        // Emit StreamStart exactly once, before any audio.
        if !self.started {
            self.started = true;
            return Some(MediaIn::StreamStart {
                call_id: self.call_id.clone(),
            });
        }

        tokio::select! {
            // Inbound decoded audio from the RX task.
            chunk = self.inbound_rx.recv() => match chunk {
                Some(c) => Some(MediaIn::Audio(c)),
                // RX task ended and channel drained → end of media → Stop.
                None => Some(MediaIn::Stop),
            },
            // Agent signalled hangup (BYE / dialog terminated / timeout). Map to
            // Stop so the pipeline finalizes. (If audio is still buffered, the
            // branch above is also ready; either way we converge on Stop next.)
            _ = self.hangup.cancelled() => Some(MediaIn::Stop),
        }
    }

    async fn send_audio(&mut self, chunk: AudioChunk) -> Result<(), FlowcatError> {
        // The pipeline already supplies carrier-rate (8 kHz) PCM; we only pace +
        // packetize. Hand the samples to the TX pump; it slices/encodes/sends.
        let Some(tx) = self.outbound_tx.as_ref() else {
            return Err(FlowcatError::Transport("SIP transport closed".into()));
        };
        tx.send(chunk.pcm).await.map_err(|_| {
            FlowcatError::Transport("SIP RTP send pump stopped (peer hung up?)".into())
        })
    }

    async fn send_clear(&mut self) -> Result<(), FlowcatError> {
        // RTP has no flush. Barge-in upstream simply stops producing AudioOut, so
        // the TX pump naturally drains. Nothing to do here.
        Ok(())
    }

    fn carrier_rate(&self) -> u32 {
        CARRIER_RATE
    }
}

/// Decode a G.711 payload to PCM16 per the negotiated codec.
fn decode_g711(codec: G711Codec, payload: &[u8]) -> Vec<i16> {
    match codec {
        G711Codec::Pcmu => crate::codec::ulaw_to_pcm16(payload),
        G711Codec::Pcma => crate::codec::alaw_to_pcm16(payload),
    }
}

/// Encode PCM16 to a G.711 payload per the negotiated codec.
fn encode_g711(codec: G711Codec, pcm: &[i16]) -> Vec<u8> {
    match codec {
        G711Codec::Pcmu => crate::codec::pcm16_to_ulaw(pcm),
        G711Codec::Pcma => crate::codec::pcm16_to_alaw(pcm),
    }
}

/// RX task: inbound RTP → symmetric-source filter → jitter buffer → G.711 decode
/// → 8 kHz `AudioChunk` forwarded to `recv`. Ends on hangup or socket error.
async fn rx_loop(
    socket: Arc<UdpSocket>,
    initial_peer: SocketAddr,
    codec: G711Codec,
    inbound_tx: mpsc::Sender<AudioChunk>,
    hangup: CancellationToken,
) {
    let mut jitter = JitterBuffer::new();
    // Symmetric RTP: lock onto the peer. We seed with the SDP peer, but learn the
    // real source from the first packet (handles NAT'd source ports). Packets
    // from any other address are dropped.
    let mut learned_peer: Option<SocketAddr> = None;
    // Track the RTP SSRC so a mid-call change (gateway restart / re-INVITE / hold
    // resume) resets the jitter buffer — otherwise the new stream's sequence
    // numbers are unrelated to the old and get dropped as "late" → audio dropout.
    let mut current_ssrc: Option<u32> = None;
    let mut buf = vec![0u8; RTP_RECV_BUF];

    loop {
        let (len, src) = tokio::select! {
            _ = hangup.cancelled() => break,
            r = socket.recv_from(&mut buf) => match r {
                Ok(v) => v,
                // Socket error is terminal for this leg; drop the sender so
                // `recv` sees the channel close and yields Stop.
                Err(e) => {
                    tracing::debug!(error = %e, "SIP RTP recv error; ending RX loop");
                    break;
                }
            },
        };

        // Learn / enforce the symmetric peer.
        match learned_peer {
            None => {
                // First packet: adopt its source (even if it differs from the SDP
                // peer's port, which is common behind NAT).
                if src != initial_peer {
                    tracing::debug!(
                        sdp_peer = %initial_peer, learned = %src,
                        "symmetric RTP: learned peer source differs from SDP"
                    );
                }
                learned_peer = Some(src);
            }
            Some(p) if src != p => {
                // Spoofed / stray packet from a different source — drop it.
                tracing::trace!(expected = %p, got = %src, "dropping RTP from foreign source");
                continue;
            }
            _ => {}
        }

        let pkt = match depacketize(&buf[..len]) {
            Ok(p) => p,
            Err(e) => {
                tracing::trace!(error = %e, "dropping malformed RTP packet");
                continue;
            }
        };

        // SSRC change mid-call (gateway restart / re-INVITE / hold resume): flush
        // the old stream's tail, then reset the jitter window so the new stream's
        // unrelated (often lower) sequence numbers aren't dropped as "late".
        match current_ssrc {
            Some(s) if s != pkt.ssrc => {
                tracing::debug!(
                    old = s,
                    new = pkt.ssrc,
                    "RTP SSRC changed; resetting jitter buffer"
                );
                for released in jitter.drain() {
                    let pcm = decode_g711(codec, &released.payload);
                    if !pcm.is_empty()
                        && inbound_tx
                            .send(AudioChunk::new(pcm, CARRIER_RATE))
                            .await
                            .is_err()
                    {
                        return;
                    }
                }
                jitter.reset();
                current_ssrc = Some(pkt.ssrc);
            }
            None => current_ssrc = Some(pkt.ssrc),
            _ => {}
        }

        // Through the jitter buffer (reorder/dedupe/drop-late), then decode each
        // released frame and forward it. Usually 0 or 1 frames out per packet.
        for released in jitter.push(pkt) {
            let pcm = decode_g711(codec, &released.payload);
            if pcm.is_empty() {
                continue;
            }
            if inbound_tx
                .send(AudioChunk::new(pcm, CARRIER_RATE))
                .await
                .is_err()
            {
                // Consumer (the Call loop) is gone; nothing more to do.
                return;
            }
        }
    }

    // On the way out, flush anything still in the jitter window so trailing audio
    // isn't lost, then let the sender drop (signals end-of-media to `recv`).
    for released in jitter.drain() {
        let pcm = decode_g711(codec, &released.payload);
        if !pcm.is_empty() {
            let _ = inbound_tx.send(AudioChunk::new(pcm, CARRIER_RATE)).await;
        }
    }
}

/// TX pump: drain PCM from `send_audio`, slice into 160-sample (20 ms) frames,
/// and emit one RTP packet every 20 ms. Ends on hangup or when the channel closes.
async fn tx_loop(
    socket: Arc<UdpSocket>,
    peer: SocketAddr,
    codec: G711Codec,
    mut outbound_rx: mpsc::Receiver<Vec<i16>>,
    hangup: CancellationToken,
) {
    let mut sender = RtpSender::new(codec.payload_type());
    // Accumulates PCM until we have a full 160-sample frame to packetize.
    let mut pending: Vec<i16> = Vec::with_capacity(FRAME_SAMPLES * 2);
    let mut ticker = tokio::time::interval(FRAME_INTERVAL);
    // If the pump falls behind (slow scheduler), don't try to "catch up" by
    // bursting frames — just resume cadence (a frame may be a touch late, which
    // is inaudible, vs. a burst which sounds worse and breaks pacing).
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    // Prime the far-end jitter buffer with a short lead of comfort silence
    // before the first real audio (see `PREROLL_FRAMES`). Paced at the normal
    // cadence through the same `RtpSender`, so seq/timestamp stay continuous
    // into the first greeting frame; any packets the receiver drops while
    // locking on are silence, not the opening syllable. This also fills the
    // brief gap between answer and the model's first audio (connect +
    // generation latency).
    let silence = encode_g711(codec, &[0i16; FRAME_SAMPLES]);
    for _ in 0..PREROLL_FRAMES {
        tokio::select! {
            _ = hangup.cancelled() => return,
            _ = ticker.tick() => {
                let dgram = sender.packetize(&silence);
                if let Err(e) = socket.send_to(&dgram, peer).await {
                    tracing::debug!(error = %e, "SIP RTP preroll send error; ending TX loop");
                    return;
                }
            }
        }
    }

    let mut channel_open = true;

    loop {
        tokio::select! {
            _ = hangup.cancelled() => break,

            // Pull whatever the pipeline produced into the pending buffer.
            maybe = outbound_rx.recv(), if channel_open => match maybe {
                Some(mut pcm) => pending.append(&mut pcm),
                // Pipeline stopped sending (call ending). Stop pulling, but keep
                // ticking to flush whatever full frames remain in `pending`.
                None => channel_open = false,
            },

            // Cadence tick: send exactly one frame if we have a full one buffered.
            _ = ticker.tick() => {
                if pending.len() >= FRAME_SAMPLES {
                    let frame: Vec<i16> = pending.drain(..FRAME_SAMPLES).collect();
                    let payload = encode_g711(codec, &frame);
                    let dgram = sender.packetize(&payload);
                    if let Err(e) = socket.send_to(&dgram, peer).await {
                        tracing::debug!(error = %e, "SIP RTP send error; ending TX loop");
                        break;
                    }
                } else if !channel_open {
                    // Channel closed and no full frame remains: we're done. (We
                    // intentionally drop a sub-frame remainder rather than pad it —
                    // <20 ms of trailing audio at end-of-call is inaudible.)
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sip::rtp::{RtpSender, PT_PCMU};

    /// Bind two UDP sockets on the loopback for a fake "peer" ↔ transport pair.
    async fn socket_pair() -> (UdpSocket, SocketAddr, UdpSocket, SocketAddr) {
        let a = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let b = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let a_addr = a.local_addr().unwrap();
        let b_addr = b.local_addr().unwrap();
        (a, a_addr, b, b_addr)
    }

    /// A fed RTP stream → `recv` yields StreamStart then decoded Audio @ 8000.
    #[tokio::test]
    async fn fed_rtp_packets_surface_as_audio_after_streamstart() {
        // `transport_sock` is the transport's RTP socket; `peer_sock` is the fake
        // remote that sends RTP to it.
        let (transport_sock, transport_addr, peer_sock, peer_addr) = socket_pair().await;
        let hangup = CancellationToken::new();
        let mut tr = SipTransport::start(
            transport_sock,
            peer_addr,
            G711Codec::Pcmu,
            "call-sip-1".to_string(),
            hangup.clone(),
        );

        // First recv is always StreamStart{call_id}.
        match tr.recv().await {
            Some(MediaIn::StreamStart { call_id }) => assert_eq!(call_id, "call-sip-1"),
            other => panic!("expected StreamStart, got {other:?}"),
        }

        // Send a few in-order G.711 frames from the peer. JITTER_DEPTH=4 holds the
        // first 4, so send enough that some are released.
        let mut sender = RtpSender::with_seeds(PT_PCMU, 0x1234, 1000, 0);
        // A recognizable μ-law payload (0xFF decodes to PCM 0 in this impl's table;
        // use a varied payload so decode produces non-trivial PCM).
        let payload: Vec<u8> = (0..FRAME_SAMPLES).map(|i| (i % 200) as u8 + 1).collect();
        for _ in 0..6 {
            let dgram = sender.packetize(&payload);
            peer_sock.send_to(&dgram, transport_addr).await.unwrap();
        }

        // We should get at least the first released frame as Audio @ 8000 with
        // 160 samples (one G.711 byte → one PCM sample).
        let got = tokio::time::timeout(std::time::Duration::from_secs(2), tr.recv())
            .await
            .expect("recv timed out");
        match got {
            Some(MediaIn::Audio(chunk)) => {
                assert_eq!(chunk.sample_rate, 8000);
                assert_eq!(chunk.pcm.len(), FRAME_SAMPLES);
                assert_eq!(chunk.pcm, crate::codec::ulaw_to_pcm16(&payload));
            }
            other => panic!("expected Audio, got {other:?}"),
        }
    }

    /// A hangup signal → recv yields Stop.
    #[tokio::test]
    async fn hangup_signal_surfaces_as_stop() {
        let (transport_sock, _t_addr, _peer_sock, peer_addr) = socket_pair().await;
        let hangup = CancellationToken::new();
        let mut tr = SipTransport::start(
            transport_sock,
            peer_addr,
            G711Codec::Pcmu,
            "call-sip-2".to_string(),
            hangup.clone(),
        );
        // Consume StreamStart.
        assert!(matches!(tr.recv().await, Some(MediaIn::StreamStart { .. })));
        // Fire hangup (simulating a BYE seen by the agent).
        hangup.cancel();
        assert_eq!(tr.recv().await, Some(MediaIn::Stop));
    }

    /// RTP from a foreign source (after the peer is learned) is dropped: it must
    /// not surface as Audio.
    #[tokio::test]
    async fn rtp_from_foreign_source_is_dropped() {
        let (transport_sock, transport_addr, peer_sock, peer_addr) = socket_pair().await;
        // A third, unexpected sender.
        let foreign = UdpSocket::bind("127.0.0.1:0").await.unwrap();

        let hangup = CancellationToken::new();
        let mut tr = SipTransport::start(
            transport_sock,
            peer_addr,
            G711Codec::Pcmu,
            "call-sip-3".to_string(),
            hangup.clone(),
        );
        assert!(matches!(tr.recv().await, Some(MediaIn::StreamStart { .. })));

        // Lock the peer in with one real frame, then send enough real frames to
        // push some out of the jitter window.
        let mut good = RtpSender::with_seeds(PT_PCMU, 1, 10, 0);
        let payload: Vec<u8> = vec![0x55u8; FRAME_SAMPLES];
        // First (real) packet learns the peer.
        peer_sock
            .send_to(&good.packetize(&payload), transport_addr)
            .await
            .unwrap();
        // Foreign packet — must be dropped, never surfaced.
        let mut bad = RtpSender::with_seeds(PT_PCMU, 999, 5000, 0);
        let bad_payload: Vec<u8> = vec![0x01u8; FRAME_SAMPLES];
        foreign
            .send_to(&bad.packetize(&bad_payload), transport_addr)
            .await
            .unwrap();
        // More real packets to flush the window.
        for _ in 0..6 {
            peer_sock
                .send_to(&good.packetize(&payload), transport_addr)
                .await
                .unwrap();
        }

        // Every Audio frame we get must be the GOOD payload, never the foreign one.
        let good_pcm = crate::codec::ulaw_to_pcm16(&payload);
        let bad_pcm = crate::codec::ulaw_to_pcm16(&bad_payload);
        for _ in 0..3 {
            match tokio::time::timeout(std::time::Duration::from_secs(2), tr.recv())
                .await
                .expect("recv timed out")
            {
                Some(MediaIn::Audio(chunk)) => {
                    assert_eq!(chunk.pcm, good_pcm, "got foreign audio!");
                    assert_ne!(chunk.pcm, bad_pcm);
                }
                Some(MediaIn::Stop) => break,
                other => panic!("unexpected {other:?}"),
            }
        }
    }

    /// `send_audio` → the TX pump packetizes 20 ms frames and sends them to the
    /// peer; the peer socket receives well-formed RTP carrying our G.711.
    #[tokio::test]
    async fn send_audio_emits_paced_rtp_frames_to_peer() {
        let (transport_sock, _t_addr, peer_sock, peer_addr) = socket_pair().await;
        let hangup = CancellationToken::new();
        let mut tr = SipTransport::start(
            transport_sock,
            peer_addr,
            G711Codec::Pcmu,
            "call-sip-4".to_string(),
            hangup.clone(),
        );
        let _ = tr.recv().await; // StreamStart

        // Two full frames' worth of PCM in one send_audio call (the pump must
        // slice it into two 20 ms RTP packets).
        let pcm: Vec<i16> = (0..(FRAME_SAMPLES * 2))
            .map(|i| ((i % 100) as i16 - 50) * 100)
            .collect();
        tr.send_audio(AudioChunk::new(pcm.clone(), 8000))
            .await
            .unwrap();

        // The TX pump first emits PREROLL_FRAMES of comfort silence (jitter-
        // buffer priming), then the two audio frames. Collect them all.
        let silence_payload = crate::codec::pcm16_to_ulaw(&[0i16; FRAME_SAMPLES]);
        let mut buf = vec![0u8; RTP_RECV_BUF];
        let mut pkts = Vec::new();
        for _ in 0..(PREROLL_FRAMES + 2) {
            let (len, from) = tokio::time::timeout(
                std::time::Duration::from_secs(2),
                peer_sock.recv_from(&mut buf),
            )
            .await
            .expect("no RTP received")
            .unwrap();
            // The datagram came from our transport's RTP socket (loopback).
            assert_eq!(from.ip(), std::net::Ipv4Addr::LOCALHOST);
            pkts.push(depacketize(&buf[..len]).unwrap());
        }

        // First PREROLL_FRAMES are comfort silence with monotonic seq.
        for i in 0..PREROLL_FRAMES {
            assert_eq!(pkts[i].payload_type, PT_PCMU);
            assert_eq!(
                pkts[i].payload, silence_payload,
                "preroll frame {i} not silence"
            );
            if i > 0 {
                assert_eq!(pkts[i].seq, pkts[i - 1].seq.wrapping_add(1));
            }
        }

        // Then the first real audio frame: μ-law of the first 160 PCM samples,
        // continuing the RTP sequence from the preroll.
        let audio0 = &pkts[PREROLL_FRAMES];
        assert_eq!(audio0.payload_type, PT_PCMU);
        assert_eq!(audio0.payload.len(), FRAME_SAMPLES);
        assert_eq!(
            audio0.payload,
            crate::codec::pcm16_to_ulaw(&pcm[..FRAME_SAMPLES])
        );
        assert_eq!(audio0.seq, pkts[PREROLL_FRAMES - 1].seq.wrapping_add(1));

        // The second audio frame: seq+1 and ts+160.
        let audio1 = &pkts[PREROLL_FRAMES + 1];
        assert_eq!(audio1.seq, audio0.seq.wrapping_add(1));
        assert_eq!(
            audio1.timestamp,
            audio0.timestamp.wrapping_add(FRAME_SAMPLES as u32)
        );
    }
}
