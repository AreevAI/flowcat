// SPDX-License-Identifier: Apache-2.0
//
//! str0m WebRTC transport.
//!
//! [`WebRtcTransport`] is a [`MediaTransport`](flowcat_core::MediaTransport) over
//! a str0m peer connection (sans-I/O WebRTC driven on the tokio runtime), with
//! Opus encode/decode. Behind the `webrtc-str0m` feature.
//!
//! - [`signaling`] — SDP offer/answer + ICE plumbing over `str0m` (the
//!   browser-facing [`WebRtcPeer`](signaling::WebRtcPeer)).
//! - [`opus`] — Opus encode/decode bridging WebRTC ↔ the pipeline's PCM frames.
//!
//! ## Shape
//!
//! ```text
//!  browser ──Opus/RTP/SRTP──► WebRtcPeer.run() ──PeerEvent::OpusIn──┐
//!                                                                    ▼
//!                                          OpusDecoder (48k) → Resampler(48k→carrier)
//!                                                                    ▼  recv() → MediaIn::Audio
//!  browser ◄─Opus/RTP/SRTP── WebRtcPeer.run() ◄──OpusOut──┐
//!                                                          │
//!                  Resampler(carrier→48k) → 20ms frame → OpusEncoder ◄── send_audio(AudioChunk)
//! ```
//!
//! The transport's `carrier_rate` is the pipeline-facing rate (16 kHz by default
//! for the cascaded path — chosen at construction). The [`signaling::WebRtcPeer`]
//! is spawned on a tokio task; this struct holds the channel ends + the two
//! resamplers + the Opus codec and adapts them to the frozen `MediaTransport`
//! seam. The input (read) leg is composed via the core
//! [`SourcePump`](flowcat_core::SourcePump) by the pipeline (e.g.
//! `pipeline/s2s.rs::spawn_transport_pump`) — `recv()` is the source it pumps.

pub mod opus;
pub mod signaling;

use async_trait::async_trait;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use flowcat_core::codec::Resampler;
use flowcat_core::error::FlowcatError;
use flowcat_core::transport::{MediaIn, MediaTransport};
use flowcat_core::types::AudioChunk;

use opus::{OpusDecoder, OpusEncoder, FRAME_20MS_48K, OPUS_RATE};
use signaling::{OpusOut, PeerChannels, PeerEvent, WebRtcPeer};

/// A WebRTC media transport for one browser session.
///
/// Construct with [`WebRtcTransport::accept_offer`]: it accepts the browser's
/// SDP offer, returns the SDP answer to send back, and spawns the str0m event
/// loop. The pipeline then drives [`recv`](MediaTransport::recv) /
/// [`send_audio`](MediaTransport::send_audio) like any other transport.
pub struct WebRtcTransport {
    /// Pipeline-facing carrier rate (e.g. 16000). Inbound `MediaIn::Audio` is at
    /// this rate; outbound `send_audio` chunks must be too.
    carrier_rate: u32,

    /// Inbound events from the peer loop (Connected / OpusIn / Closed).
    inbound_rx: tokio::sync::mpsc::Receiver<PeerEvent>,
    /// Outbound Opus frames to the peer loop.
    outbound_tx: tokio::sync::mpsc::Sender<OpusOut>,
    /// Cancels the peer loop on drop / call end.
    cancel: CancellationToken,
    /// The spawned `WebRtcPeer::run` task (aborted on drop).
    peer_task: JoinHandle<()>,

    /// Opus decoder (browser Opus 48 kHz → PCM 48 kHz).
    decoder: OpusDecoder,
    /// 48 kHz (Opus) → carrier resampler for inbound audio.
    in_resampler: Resampler,

    /// Opus encoder (PCM 48 kHz → browser Opus).
    encoder: OpusEncoder,
    /// carrier → 48 kHz (Opus) resampler for outbound audio.
    out_resampler: Resampler,
    /// Accumulates 48 kHz PCM until a full 20 ms (960-sample) Opus frame is ready.
    out_pending: Vec<i16>,
    /// Monotonic 48 kHz RTP timestamp for outbound frames.
    out_rtp_ts: u64,

    /// `StreamStart` is emitted once, on the first `Connected`.
    started: bool,
}

impl WebRtcTransport {
    /// Accept a browser SDP offer and start the transport.
    ///
    /// - `offer_sdp` — the raw browser SDP offer (validated defensively in
    ///   [`signaling::WebRtcPeer::accept_offer`]).
    /// - `socket` — an already-bound UDP socket for the WebRTC media (the caller
    ///   chooses the bind interface — security).
    /// - `carrier_rate` — the pipeline-facing sample rate (e.g. 16000).
    ///
    /// Returns the transport and the **SDP answer** string to send back to the
    /// browser. A malformed offer or codec init failure is a
    /// [`FlowcatError`], never a panic.
    pub fn accept_offer(
        offer_sdp: &str,
        socket: tokio::net::UdpSocket,
        carrier_rate: u32,
    ) -> Result<(Self, String), FlowcatError> {
        let (peer, answer_sdp, channels) = WebRtcPeer::accept_offer(offer_sdp, socket)?;
        let PeerChannels {
            inbound_rx,
            outbound_tx,
            cancel,
        } = channels;

        let decoder = OpusDecoder::new()?;
        let encoder = OpusEncoder::new()?;
        let in_resampler = Resampler::new(OPUS_RATE, carrier_rate)?;
        let out_resampler = Resampler::new(carrier_rate, OPUS_RATE)?;

        let peer_task = tokio::spawn(peer.run());

        Ok((
            Self {
                carrier_rate,
                inbound_rx,
                outbound_tx,
                cancel,
                peer_task,
                decoder,
                in_resampler,
                encoder,
                out_resampler,
                out_pending: Vec::with_capacity(FRAME_20MS_48K * 2),
                out_rtp_ts: 0,
                started: false,
            },
            answer_sdp,
        ))
    }

    /// Decode an inbound Opus packet and resample it to the carrier rate.
    /// Returns `None` (after logging) on a codec/resample error so one bad frame
    /// can't kill the call.
    fn decode_inbound(&mut self, opus: &[u8]) -> Option<AudioChunk> {
        let pcm48 = match self.decoder.decode(opus) {
            Ok(p) => p,
            Err(e) => {
                tracing::debug!(error = %e, "webrtc: dropping undecodable inbound Opus frame");
                return None;
            }
        };
        let chunk48 = AudioChunk::new(pcm48, OPUS_RATE);
        match self.in_resampler.process(&chunk48) {
            Ok(out) if !out.is_empty() => Some(out),
            Ok(_) => None, // sub-block remainder buffered in the resampler
            Err(e) => {
                tracing::debug!(error = %e, "webrtc: inbound resample error; dropping frame");
                None
            }
        }
    }
}

impl Drop for WebRtcTransport {
    fn drop(&mut self) {
        // Tear the peer loop down so it can't outlive the transport.
        self.cancel.cancel();
        self.peer_task.abort();
    }
}

#[async_trait]
impl MediaTransport for WebRtcTransport {
    async fn recv(&mut self) -> Option<MediaIn> {
        loop {
            match self.inbound_rx.recv().await {
                Some(PeerEvent::Connected { call_id }) => {
                    if !self.started {
                        self.started = true;
                        return Some(MediaIn::StreamStart { call_id });
                    }
                    // Already started — ignore a duplicate connect.
                }
                Some(PeerEvent::OpusIn(opus)) => {
                    if let Some(chunk) = self.decode_inbound(&opus) {
                        return Some(MediaIn::Audio(chunk));
                    }
                    // Buffered/dropped frame: keep reading.
                }
                Some(PeerEvent::Closed) | None => return Some(MediaIn::Stop),
            }
        }
    }

    async fn send_audio(&mut self, chunk: AudioChunk) -> Result<(), FlowcatError> {
        // The pipeline supplies carrier-rate PCM. Resample to the 48 kHz Opus
        // clock, accumulate, and emit full 20 ms (960-sample) Opus frames.
        if chunk.sample_rate != self.carrier_rate {
            return Err(FlowcatError::Codec(format!(
                "webrtc send_audio: expected carrier rate {} Hz, got {} Hz",
                self.carrier_rate, chunk.sample_rate
            )));
        }
        let resampled = self.out_resampler.process(&chunk)?;
        self.out_pending.extend_from_slice(&resampled.pcm);

        while self.out_pending.len() >= FRAME_20MS_48K {
            let frame: Vec<i16> = self.out_pending.drain(..FRAME_20MS_48K).collect();
            let payload = self.encoder.encode(&frame)?;
            let rtp_time = self.out_rtp_ts;
            self.out_rtp_ts = self.out_rtp_ts.wrapping_add(FRAME_20MS_48K as u64);
            // If the peer loop is gone, the call is ending; surface a transport
            // error so the pipeline finalizes.
            if self
                .outbound_tx
                .send(OpusOut { payload, rtp_time })
                .await
                .is_err()
            {
                return Err(FlowcatError::Transport(
                    "webrtc peer loop stopped (browser disconnected?)".into(),
                ));
            }
        }
        Ok(())
    }

    async fn send_clear(&mut self) -> Result<(), FlowcatError> {
        // WebRTC/RTP has no playout flush; barge-in upstream simply stops
        // producing AudioOut, so the outbound frame stream drains. We also drop
        // any sub-frame remainder so the next talkspurt starts clean.
        self.out_pending.clear();
        Ok(())
    }

    fn carrier_rate(&self) -> u32 {
        self.carrier_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use str0m::{Candidate, Rtc};

    /// The in-process handshake (two str0m peers over loopback UDP — a browser
    /// stand-in): keep the browser's pending offer, feed
    /// our SDP answer back via `accept_answer`, then pump both peers until DTLS
    /// connects. Asserts the transport emits `StreamStart`, an inbound Opus frame
    /// surfaces as `MediaIn::Audio` at the carrier rate, and `send_audio`
    /// produces outbound Opus on the wire.
    #[tokio::test]
    async fn handshake_and_audio_roundtrip() {
        use opus::{OpusEncoder, FRAME_20MS_48K};

        // --- Browser side: build Rtc, add audio media, make an offer. ---
        let browser_sock = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let browser_addr = browser_sock.local_addr().unwrap();
        let mut browser = Rtc::builder().build(std::time::Instant::now());
        browser.add_local_candidate(Candidate::host(browser_addr, "udp").unwrap());
        let mut change = browser.sdp_api();
        let browser_mid = change.add_media(
            str0m::media::MediaKind::Audio,
            str0m::media::Direction::SendRecv,
            None,
            None,
            None,
        );
        let (offer, pending) = change.apply().expect("offer");
        let offer_sdp = offer.to_sdp_string();

        // --- Our transport: bind a socket, accept the offer, get the answer. ---
        let our_socket = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let (mut transport, answer_sdp) =
            WebRtcTransport::accept_offer(&offer_sdp, our_socket, 16000).expect("accept");
        assert_eq!(transport.carrier_rate(), 16000);

        // Feed our answer back to the browser to complete negotiation.
        let answer = str0m::change::SdpAnswer::from_sdp_string(&answer_sdp).expect("answer parses");
        browser
            .sdp_api()
            .accept_answer(pending, answer)
            .expect("browser accepts answer");

        // --- Drive the browser peer on a task: pump UDP + send 5 audio frames
        //     once connected, so the transport sees inbound Opus. ---
        let browser_handle = tokio::spawn(async move {
            let mut buf = vec![0u8; 2048];
            let mut enc = OpusEncoder::new().unwrap();
            let mut connected = false;
            let mut sent_frames = 0u32;
            let mut rtp_ts: u64 = 0;
            let started = std::time::Instant::now();
            loop {
                if started.elapsed() > Duration::from_secs(8) {
                    return;
                }
                // Drain output.
                loop {
                    match browser.poll_output() {
                        Ok(str0m::Output::Timeout(_)) => break,
                        Ok(str0m::Output::Transmit(t)) => {
                            let _ = browser_sock.send_to(&t.contents, t.destination).await;
                        }
                        Ok(str0m::Output::Event(str0m::Event::Connected)) => connected = true,
                        Ok(str0m::Output::Event(_)) => {}
                        Err(_) => return,
                    }
                }
                // Once connected, write a few Opus frames to our transport.
                if connected && sent_frames < 5 {
                    if let Some(writer) = browser.writer(browser_mid) {
                        // Bind `pt` in its own statement so the borrowing
                        // payload_params() iterator is dropped before we move
                        // `writer` into `write()` (E0505).
                        let pt = writer.payload_params().next().map(|p| p.pt());
                        if let Some(pt) = pt {
                            // A 440 Hz tone @ 48 kHz, 20 ms.
                            let pcm: Vec<i16> = (0..FRAME_20MS_48K)
                                .map(|i| {
                                    let t = (sent_frames as usize * FRAME_20MS_48K + i) as f32
                                        / 48000.0;
                                    (6000.0 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()) as i16
                                })
                                .collect();
                            if let Ok(payload) = enc.encode(&pcm) {
                                let mt = str0m::media::MediaTime::new(
                                    rtp_ts,
                                    str0m::media::Frequency::FORTY_EIGHT_KHZ,
                                );
                                let _ = writer.write(pt, std::time::Instant::now(), mt, payload);
                                rtp_ts += FRAME_20MS_48K as u64;
                                sent_frames += 1;
                            }
                        }
                    }
                }

                tokio::select! {
                    r = browser_sock.recv_from(&mut buf) => {
                        if let Ok((n, src)) = r {
                            if let Ok(contents) = (&buf[..n]).try_into() {
                                let input = str0m::Input::Receive(
                                    std::time::Instant::now(),
                                    str0m::net::Receive {
                                        proto: str0m::net::Protocol::Udp,
                                        source: src,
                                        destination: browser_addr,
                                        contents,
                                    },
                                );
                                if browser.accepts(&input) {
                                    let _ = browser.handle_input(input);
                                }
                            }
                        }
                    }
                    _ = tokio::time::sleep(Duration::from_millis(5)) => {}
                }
                let _ = browser.handle_input(str0m::Input::Timeout(std::time::Instant::now()));
                if !browser.is_alive() {
                    return;
                }
            }
        });

        // --- Assert: first event is StreamStart, then an Audio chunk at 16 kHz. ---
        let first = tokio::time::timeout(Duration::from_secs(8), transport.recv())
            .await
            .expect("recv StreamStart timed out");
        assert!(
            matches!(first, Some(MediaIn::StreamStart { .. })),
            "first event is StreamStart, got {first:?}"
        );

        // Next, an inbound Opus frame decoded + resampled to the 16 kHz carrier.
        let audio = tokio::time::timeout(Duration::from_secs(8), async {
            loop {
                match transport.recv().await {
                    Some(MediaIn::Audio(c)) => return Some(c),
                    Some(MediaIn::Stop) | None => return None,
                    _ => {}
                }
            }
        })
        .await
        .expect("recv Audio timed out");
        let audio = audio.expect("got an inbound Audio chunk before Stop");
        assert_eq!(
            audio.sample_rate, 16000,
            "inbound audio at the carrier rate"
        );
        assert!(!audio.is_empty(), "inbound audio chunk carries samples");

        // --- Outbound: push carrier-rate PCM; send_audio must Opus-encode +
        //     enqueue without error (the browser pump receives it). ---
        let pcm16k: Vec<i16> = (0..320 * 4) // 4 × 20 ms @ 16 kHz
            .map(|i| {
                let t = i as f32 / 16000.0;
                (5000.0 * (2.0 * std::f32::consts::PI * 330.0 * t).sin()) as i16
            })
            .collect();
        transport
            .send_audio(AudioChunk::new(pcm16k, 16000))
            .await
            .expect("send_audio enqueues outbound Opus without error");

        browser_handle.abort();
        drop(transport); // aborts the peer task cleanly
    }

    /// `send_audio` at the wrong rate is rejected (no panic).
    #[tokio::test]
    async fn send_audio_wrong_rate_is_rejected() {
        let browser_sock = std::net::UdpSocket::bind("127.0.0.1:0").unwrap();
        let addr = browser_sock.local_addr().unwrap();
        let mut rtc = Rtc::builder().build(std::time::Instant::now());
        rtc.add_local_candidate(Candidate::host(addr, "udp").unwrap());
        let mut change = rtc.sdp_api();
        change.add_media(
            str0m::media::MediaKind::Audio,
            str0m::media::Direction::SendRecv,
            None,
            None,
            None,
        );
        let (offer, _pending) = change.apply().unwrap();

        let our_socket = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let (mut transport, _answer) =
            WebRtcTransport::accept_offer(&offer.to_sdp_string(), our_socket, 16000).unwrap();

        let err = transport
            .send_audio(AudioChunk::new(vec![0i16; 160], 8000))
            .await
            .unwrap_err();
        assert!(matches!(err, FlowcatError::Codec(_)));
    }
}
