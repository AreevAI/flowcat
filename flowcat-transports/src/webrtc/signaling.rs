// SPDX-License-Identifier: Apache-2.0
//
//! WebRTC signaling (SDP offer/answer + ICE) over `str0m` (sans-I/O),
//! driven on a tokio UDP socket.
//!
//! This is the **browser-compatible WebRTC peer** behind the
//! [`super::WebRtcTransport`]. It owns one [`str0m::Rtc`] instance plus the
//! single UDP socket all WebRTC traffic (STUN/DTLS/SRTP) is multiplexed over,
//! and runs the str0m event loop on a tokio task:
//!
//! 1. [`WebRtcPeer::accept_offer`] parses a **browser SDP offer**, builds the
//!    `Rtc`, adds the local host ICE candidate (our bound UDP addr), and
//!    produces the **SDP answer** (codecs / ICE-ufrag/pwd / DTLS fingerprint).
//! 2. [`WebRtcPeer::run`] drives the loop: `poll_output` → UDP `send_to` /
//!    sleep-until-timeout; inbound UDP → `handle_input`; surface inbound Opus
//!    [`PeerEvent`]s and accept outbound Opus frames over channels.
//!
//! ## SECURITY (WebRTC/ICE/DTLS-SRTP)
//!
//! The offer/answer + ICE path is the attack surface a remote browser controls.
//! Hardening here, called out for the reviewer:
//!
//! - **No panics on hostile input.** The offer is parsed via
//!   [`SdpOffer::from_sdp_string`] and accepted via `sdp_api().accept_offer` —
//!   both fallible; a malformed/oversized/empty offer returns
//!   [`FlowcatError::Protocol`], never panics. The offer string is length-capped
//!   ([`MAX_OFFER_BYTES`]) before parsing so a giant body can't blow memory.
//! - **DTLS-SRTP.** str0m generates a fresh self-signed DTLS certificate per
//!   `Rtc` and advertises its fingerprint in the SDP answer; the browser pins
//!   the peer to *our* fingerprint and we pin it to *theirs* (carried in the
//!   offer). `fingerprint_verification` is left **on** (str0m default) so a
//!   handshake whose cert doesn't match the SDP-advertised fingerprint is
//!   rejected — this is the MITM guard. We never disable it. The crypto provider
//!   is str0m's feature-flag default (aws-lc-rs).
//! - **ICE candidate trust.** We add exactly one *local* host candidate (our own
//!   bound socket addr). Remote candidates are conveyed inside the offer and
//!   handled by str0m's ICE agent; we do not blindly `add_remote_candidate` from
//!   any external trickle source. Inbound datagrams are demultiplexed by
//!   str0m's own `accepts()` so stray/spoofed packets that don't belong to this
//!   ICE session are ignored. The bind address is caller-chosen
//!   ([`WebRtcPeer::accept_offer`] takes the socket) so a deployment binds to a
//!   reachable interface, not a wildcard it didn't intend.
//! - **Bounded work.** Channels are bounded so a slow consumer applies
//!   backpressure rather than growing memory; an inbound media frame that can't
//!   be queued (consumer momentarily full) is dropped rather than stalling the
//!   ICE/DTLS servicing loop.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use str0m::change::SdpOffer;
use str0m::media::{Frequency, MediaKind, MediaTime, Mid};
use str0m::net::{Protocol, Receive};
use str0m::{Candidate, Event, IceConnectionState, Input, Output, Rtc};
use tokio::net::UdpSocket;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use flowcat_core::error::FlowcatError;

/// Largest SDP offer body we will parse. A browser offer is a few KB; this caps
/// a hostile oversized body before it reaches the parser. (Security.)
pub const MAX_OFFER_BYTES: usize = 64 * 1024;

/// UDP receive buffer for the multiplexed WebRTC socket (STUN/DTLS/SRTP). The
/// str0m examples use 2000; we keep DATAGRAM-MTU headroom.
const UDP_RECV_BUF: usize = 2048;

/// Bound on the inbound decoded-Opus-packet channel (frames). Generous (~1 s of
/// 20 ms frames) but bounded so a stalled consumer can't grow memory.
const INBOUND_DEPTH: usize = 64;
/// Bound on the outbound Opus-frame channel the transport pushes to.
const OUTBOUND_DEPTH: usize = 64;

/// An inbound event surfaced by the running [`WebRtcPeer`] loop.
#[derive(Debug, Clone)]
pub enum PeerEvent {
    /// ICE+DTLS established and an audio m-line is known — the media path is
    /// live. Carries a synthetic call id (the audio `Mid`) so the transport can
    /// emit `MediaIn::StreamStart`.
    Connected { call_id: String },
    /// One inbound **Opus packet** (a depayloaded RTP frame) from the browser,
    /// at the 48 kHz Opus clock. The transport decodes + resamples it.
    OpusIn(Vec<u8>),
    /// The peer disconnected / the session ended.
    Closed,
}

/// One outbound Opus packet to write to the browser (already encoded at 48 kHz).
#[derive(Debug, Clone)]
pub struct OpusOut {
    /// The compressed Opus payload (a full depayloaded frame str0m will RTP-ize).
    pub payload: Vec<u8>,
    /// The RTP timestamp at the 48 kHz Opus clock.
    pub rtp_time: u64,
}

/// A browser-compatible WebRTC peer: one [`Rtc`] + its multiplexed UDP socket.
///
/// Build with [`accept_offer`](Self::accept_offer) (which yields the SDP answer
/// to send back to the browser), then [`run`](Self::run) on a task. The
/// [`PeerChannels`] returned alongside bridge it to the
/// [`super::WebRtcTransport`].
pub struct WebRtcPeer {
    rtc: Rtc,
    socket: Arc<UdpSocket>,
    local_addr: SocketAddr,
    /// The audio media `Mid`, learned from the accepted offer (browsers send an
    /// audio m-line). Outbound audio is written to this mid.
    audio_mid: Option<Mid>,
    inbound_tx: mpsc::Sender<PeerEvent>,
    outbound_rx: mpsc::Receiver<OpusOut>,
    cancel: CancellationToken,
    /// Set when we've emitted `Connected` so it fires exactly once.
    announced: bool,
    /// Whether DTLS is up (we only announce `Connected` once both this and an
    /// audio mid are known).
    connected: bool,
}

/// The transport-facing channel ends for a [`WebRtcPeer`].
pub struct PeerChannels {
    /// Inbound events (Connected / OpusIn / Closed) the transport reads.
    pub inbound_rx: mpsc::Receiver<PeerEvent>,
    /// Outbound Opus frames the transport writes to be sent to the browser.
    pub outbound_tx: mpsc::Sender<OpusOut>,
    /// Cancels the peer loop (transport drop / call end).
    pub cancel: CancellationToken,
}

impl WebRtcPeer {
    /// Accept a browser SDP **offer** and produce the SDP **answer** string.
    ///
    /// - `offer_sdp` — the raw SDP offer text the browser POSTed. Length-capped
    ///   ([`MAX_OFFER_BYTES`]) and parsed defensively; a malformed/empty/oversized
    ///   offer yields [`FlowcatError::Protocol`] (never a panic).
    /// - `socket` — an already-bound UDP socket; its local addr becomes our host
    ///   ICE candidate. The caller chooses the bind interface (security).
    ///
    /// Returns the peer (drive it with [`run`](Self::run)) plus the answer SDP to
    /// hand back to the browser, and the [`PeerChannels`] for the transport.
    pub fn accept_offer(
        offer_sdp: &str,
        socket: UdpSocket,
    ) -> Result<(Self, String, PeerChannels), FlowcatError> {
        // SECURITY: cap before parse so a hostile oversized body can't blow up
        // the parser / memory.
        if offer_sdp.len() > MAX_OFFER_BYTES {
            return Err(FlowcatError::Protocol(format!(
                "SDP offer too large: {} bytes (max {MAX_OFFER_BYTES})",
                offer_sdp.len()
            )));
        }
        if offer_sdp.trim().is_empty() {
            return Err(FlowcatError::Protocol("empty SDP offer".into()));
        }

        // Parse defensively — `from_sdp_string` is fallible and never panics on
        // garbage. Map any error to a clean protocol error.
        let offer = SdpOffer::from_sdp_string(offer_sdp)
            .map_err(|e| FlowcatError::Protocol(format!("invalid SDP offer: {e}")))?;

        let local_addr = socket
            .local_addr()
            .map_err(|e| FlowcatError::Transport(format!("socket local_addr: {e}")))?;

        // Build the Rtc. str0m generates a fresh self-signed DTLS cert per Rtc,
        // resolves the crypto provider from feature flags, and leaves
        // `fingerprint_verification` ON (MITM guard) — we keep all defaults.
        let mut rtc = Rtc::new(Instant::now());

        // Our single local host candidate is the bound UDP addr.
        let candidate = Candidate::host(local_addr, "udp")
            .map_err(|e| FlowcatError::Transport(format!("host candidate: {e}")))?;
        rtc.add_local_candidate(candidate);

        // Produce the SDP answer. `accept_offer` validates the offer's media /
        // ICE / DTLS lines; an unacceptable offer is an error, not a panic.
        let answer = rtc
            .sdp_api()
            .accept_offer(offer)
            .map_err(|e| FlowcatError::Protocol(format!("offer not accepted: {e}")))?;
        let answer_sdp = answer.to_sdp_string();

        let socket = Arc::new(socket);
        let (inbound_tx, inbound_rx) = mpsc::channel::<PeerEvent>(INBOUND_DEPTH);
        let (outbound_tx, outbound_rx) = mpsc::channel::<OpusOut>(OUTBOUND_DEPTH);
        let cancel = CancellationToken::new();

        let peer = Self {
            rtc,
            socket,
            local_addr,
            audio_mid: None,
            inbound_tx,
            outbound_rx,
            cancel: cancel.clone(),
            announced: false,
            connected: false,
        };
        let channels = PeerChannels {
            inbound_rx,
            outbound_tx,
            cancel,
        };
        Ok((peer, answer_sdp, channels))
    }

    /// The bound local UDP address (the host ICE candidate). Useful for tests.
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Drive the str0m event loop until the peer disconnects or the loop is
    /// cancelled. Owns the UDP socket; this is the task body.
    pub async fn run(mut self) {
        let mut buf = vec![0u8; UDP_RECV_BUF];
        // Once the transport drops its `outbound_tx`, the recv() resolves to
        // `None` immediately and forever; track that so we stop selecting on it
        // (otherwise it would busy-loop). The session stays alive to drain
        // inbound media until the peer closes / we're cancelled.
        let mut outbound_open = true;

        loop {
            if !self.rtc.is_alive() {
                break;
            }

            // 1) Drain all immediate output (transmits + events) until str0m
            //    asks us to wait for a timeout.
            let timeout = match self.poll_out().await {
                Ok(t) => t,
                Err(()) => break,
            };
            let sleep_dur = timeout.saturating_duration_since(Instant::now());

            // 2) Wait for inbound UDP, an outbound audio frame, the str0m
            //    timeout, or cancellation.
            tokio::select! {
                _ = self.cancel.cancelled() => break,

                r = self.socket.recv_from(&mut buf) => match r {
                    Ok((n, source)) => self.handle_socket_input(&buf[..n], source),
                    Err(e) => {
                        tracing::debug!(error = %e, "webrtc UDP recv error; ending loop");
                        break;
                    }
                },

                maybe = self.outbound_rx.recv(), if outbound_open => match maybe {
                    Some(frame) => self.write_audio(frame),
                    None => outbound_open = false, // sender dropped; stop polling it
                },

                _ = tokio::time::sleep(sleep_dur) => {
                    // The scheduled timeout elapsed; the handle_input below drives
                    // str0m's timers forward.
                }
            }

            // 3) Always drive time forward so str0m's timers advance.
            if self
                .rtc
                .handle_input(Input::Timeout(Instant::now()))
                .is_err()
            {
                break;
            }
        }

        // Best-effort close notification to the transport.
        let _ = self.inbound_tx.try_send(PeerEvent::Closed);
    }

    /// Drain str0m output: send transmits on the socket, fold events into
    /// `PeerEvent`s, and return the next timeout. `Err(())` means the session is
    /// dead and the loop should stop.
    async fn poll_out(&mut self) -> Result<Instant, ()> {
        loop {
            if !self.rtc.is_alive() {
                return Err(());
            }
            match self.rtc.poll_output() {
                Ok(Output::Timeout(t)) => return Ok(t),
                Ok(Output::Transmit(t)) => {
                    if let Err(e) = self.socket.send_to(&t.contents, t.destination).await {
                        tracing::debug!(error = %e, "webrtc UDP send error");
                        return Err(());
                    }
                }
                Ok(Output::Event(ev)) => {
                    if self.handle_event(ev).is_break() {
                        return Err(());
                    }
                }
                Err(e) => {
                    tracing::debug!(error = %e, "webrtc poll_output error; disconnecting");
                    self.rtc.disconnect();
                    return Err(());
                }
            }
        }
    }

    /// Fold one str0m [`Event`] into a [`PeerEvent`] (or internal state).
    fn handle_event(&mut self, ev: Event) -> std::ops::ControlFlow<()> {
        use std::ops::ControlFlow::{Break, Continue};
        match ev {
            Event::IceConnectionStateChange(IceConnectionState::Disconnected) => {
                self.rtc.disconnect();
                return Break(());
            }
            Event::Connected => {
                self.connected = true;
                self.announce_connected();
            }
            Event::MediaAdded(m) => {
                if m.kind == MediaKind::Audio {
                    self.audio_mid = Some(m.mid);
                }
                self.announce_connected();
            }
            Event::MediaData(data) => {
                // Inbound depayloaded Opus frame from the browser; `data.data` is
                // an Arc<[u8]> — the full Opus packet.
                let payload = data.data.to_vec();
                if !payload.is_empty()
                    && self
                        .inbound_tx
                        .try_send(PeerEvent::OpusIn(payload))
                        .is_err()
                    && self.inbound_tx.is_closed()
                {
                    // Channel closed → the transport was dropped → end. (A
                    // momentary `Full` is *not* closed: we drop that one 20 ms
                    // frame rather than stall DTLS/STUN servicing.)
                    return Break(());
                }
            }
            _ => {}
        }
        Continue(())
    }

    /// Emit `Connected` exactly once, when both DTLS is up and an audio mid is
    /// known (so the transport has a stable call id).
    fn announce_connected(&mut self) {
        if self.announced || !self.connected {
            return;
        }
        let Some(mid) = self.audio_mid else { return };
        self.announced = true;
        let _ = self.inbound_tx.try_send(PeerEvent::Connected {
            call_id: mid.to_string(),
        });
    }

    /// Feed one inbound UDP datagram into str0m, demultiplexed by `accepts`.
    fn handle_socket_input(&mut self, data: &[u8], source: SocketAddr) {
        // Parse the datagram into str0m's network input. A datagram that doesn't
        // parse as STUN/DTLS/SRTP/RTP is silently ignored (hostile/stray input).
        let contents = match data.try_into() {
            Ok(c) => c,
            Err(_) => {
                tracing::trace!("webrtc: undecodable datagram dropped");
                return;
            }
        };
        let input = Input::Receive(
            Instant::now(),
            Receive {
                proto: Protocol::Udp,
                source,
                destination: self.local_addr,
                contents,
            },
        );
        // SECURITY: `accepts` is str0m's own demultiplexer — it checks the packet
        // belongs to *this* ICE/DTLS session. Packets that don't are dropped, so
        // a stray/spoofed datagram to our port can't inject media.
        if !self.rtc.accepts(&input) {
            tracing::trace!(%source, "webrtc: datagram not for this session; dropped");
            return;
        }
        if let Err(e) = self.rtc.handle_input(input) {
            tracing::debug!(error = %e, "webrtc handle_input error; disconnecting");
            self.rtc.disconnect();
        }
    }

    /// Write one outbound Opus frame to the browser via the audio media writer.
    fn write_audio(&mut self, frame: OpusOut) {
        let Some(mid) = self.audio_mid else {
            // No audio m-line negotiated yet — drop (shouldn't happen once
            // Connected, but never panic).
            return;
        };
        let Some(writer) = self.rtc.writer(mid) else {
            return;
        };
        // Use the negotiated Opus payload type for this writer.
        let Some(pt) = writer.payload_params().next().map(|p| p.pt()) else {
            tracing::trace!("webrtc: no payload params for audio writer; dropping frame");
            return;
        };
        let rtp_time = MediaTime::new(frame.rtp_time, Frequency::FORTY_EIGHT_KHZ);
        if let Err(e) = writer.write(pt, Instant::now(), rtp_time, frame.payload) {
            tracing::debug!(error = %e, "webrtc audio write error");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A realistic (minimal) browser-style SDP offer with one audio m-line
    /// negotiating Opus, plus the ICE + DTLS lines a real offer carries. This is
    /// close enough to a Chrome/Firefox offer for str0m to accept it.
    fn sample_browser_offer(host: &str) -> String {
        format!(
            "v=0\r\n\
             o=- 4611731400430051336 2 IN IP4 127.0.0.1\r\n\
             s=-\r\n\
             t=0 0\r\n\
             a=group:BUNDLE 0\r\n\
             a=msid-semantic: WMS\r\n\
             m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n\
             c=IN IP4 {host}\r\n\
             a=rtcp:9 IN IP4 0.0.0.0\r\n\
             a=ice-ufrag:F7gI\r\n\
             a=ice-pwd:x9cml/YzichV2+XlhiMu8g\r\n\
             a=ice-options:trickle\r\n\
             a=fingerprint:sha-256 \
             AB:CD:EF:01:23:45:67:89:AB:CD:EF:01:23:45:67:89:AB:CD:EF:01:23:45:67:89:AB:CD:EF:01:23:45:67:89\r\n\
             a=setup:actpass\r\n\
             a=mid:0\r\n\
             a=sendrecv\r\n\
             a=rtcp-mux\r\n\
             a=rtpmap:111 opus/48000/2\r\n\
             a=fmtp:111 minptime=10;useinbandfec=1\r\n"
        )
    }

    /// Offer→answer round-trip: a realistic browser offer is accepted and a
    /// valid SDP answer is produced with the expected codec/ICE/DTLS lines.
    #[tokio::test]
    async fn accept_offer_produces_valid_answer() {
        let socket = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let offer = sample_browser_offer("127.0.0.1");
        let (peer, answer, _ch) = WebRtcPeer::accept_offer(&offer, socket).expect("offer accepted");

        // We added our bound socket addr as the host candidate.
        assert_eq!(peer.local_addr().ip(), std::net::Ipv4Addr::LOCALHOST);

        // The answer must be a well-formed SDP carrying the negotiated audio
        // codec (Opus), our ICE credentials, and a DTLS fingerprint (the
        // MITM-guard the browser pins to).
        assert!(answer.starts_with("v=0"), "answer is SDP: {answer}");
        assert!(answer.contains("m=audio"), "answer has the audio m-line");
        let lower = answer.to_ascii_lowercase();
        assert!(lower.contains("opus"), "answer negotiates opus: {answer}");
        assert!(answer.contains("a=ice-ufrag:"), "answer carries ICE ufrag");
        assert!(answer.contains("a=ice-pwd:"), "answer carries ICE pwd");
        assert!(
            answer.contains("a=fingerprint:"),
            "answer carries the DTLS-SRTP fingerprint (MITM guard)"
        );
        // We answer a fixed DTLS role (browser offered actpass).
        assert!(
            answer.contains("a=setup:active") || answer.contains("a=setup:passive"),
            "answer fixes the DTLS role"
        );
    }

    /// Malformed / empty / oversized offers are rejected cleanly — no panic.
    #[tokio::test]
    async fn malformed_offers_are_rejected_without_panicking() {
        // Empty.
        let s = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        assert!(matches!(
            WebRtcPeer::accept_offer("", s),
            Err(FlowcatError::Protocol(_))
        ));

        // Whitespace-only.
        let s = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        assert!(matches!(
            WebRtcPeer::accept_offer("   \r\n  ", s),
            Err(FlowcatError::Protocol(_))
        ));

        // Garbage that is not SDP.
        let s = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        assert!(WebRtcPeer::accept_offer("this is not sdp at all", s).is_err());

        // Truncated SDP (header only, no media). Must not panic — Ok or Err both
        // acceptable; reaching the next line proves no panic.
        let s = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let _ = WebRtcPeer::accept_offer("v=0\r\no=- 0 0 IN IP4 0.0.0.0\r\n", s);

        // Oversized body is capped before parse.
        let s = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let big = "v=0\r\n".to_string() + &"a=x\r\n".repeat(MAX_OFFER_BYTES);
        assert!(matches!(
            WebRtcPeer::accept_offer(&big, s),
            Err(FlowcatError::Protocol(_))
        ));
    }

    /// A garbage UDP datagram fed to the input handler is dropped, not panicked,
    /// and does not tear down the session.
    #[tokio::test]
    async fn garbage_datagram_is_dropped_safely() {
        let socket = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let offer = sample_browser_offer("127.0.0.1");
        let (mut peer, _answer, _ch) =
            WebRtcPeer::accept_offer(&offer, socket).expect("offer accepted");
        let fake_source: SocketAddr = "127.0.0.1:5555".parse().unwrap();
        // A datagram that is neither STUN nor DTLS nor RTP.
        peer.handle_socket_input(&[0xFFu8, 0x00, 0x13, 0x37], fake_source);
        // Still alive — a stray datagram must not disconnect us.
        assert!(
            peer.rtc.is_alive(),
            "garbage datagram must not kill the session"
        );
    }
}
