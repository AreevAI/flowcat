// SPDX-License-Identifier: Apache-2.0
//
//! Hand-rolled RTP (RFC 3550) for G.711 telephony — packetize, depacketize,
//! and a small fixed jitter buffer.
//!
//! RTP is deliberately hand-rolled rather than pulled from a crate: for fixed
//! 20 ms G.711 it is *simple and fully deterministic*, and avoids a risky media
//! dependency (see SIP-DESIGN.md §2). The signaling stack (`rsipstack`) does not
//! touch any of this — RTP rides a plain `tokio::net::UdpSocket` owned by
//! [`SipTransport`](crate::sip::transport::SipTransport).
//!
//! ## Wire format (the 12-byte fixed header we emit/parse)
//!
//! ```text
//!  0                   1                   2                   3
//!  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |V=2|P|X|  CC   |M|     PT      |       sequence number         |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |                           timestamp                           |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |           synchronization source (SSRC) identifier            |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! ```
//!
//! We always emit V=2, P=0, X=0, CC=0, M=0 (no CSRCs, no extensions, no marker
//! bit — telephony G.711 has no use for them). On receive we parse the same
//! fields, **skip** any CSRC list (CC>0) and a header extension (X=1) if a peer
//! sends them, and hand back the payload. Sequence increments by 1 per packet;
//! timestamp increments by the sample count (160 for a 20 ms G.711 frame, since
//! the G.711 clock rate is 8000 Hz and one byte == one sample).

/// RTP version 2 (the only version RFC 3550 defines / we speak).
const RTP_VERSION: u8 = 2;

/// The fixed RTP header length in bytes (no CSRCs, no extension).
pub const RTP_HEADER_LEN: usize = 12;

/// G.711 payload type for μ-law (PCMU) — static RTP/AVP assignment (RFC 3551).
pub const PT_PCMU: u8 = 0;
/// G.711 payload type for A-law (PCMA) — static RTP/AVP assignment (RFC 3551).
pub const PT_PCMA: u8 = 8;

/// A parsed inbound RTP packet (the fields [`SipTransport`] cares about).
///
/// [`crate::sip::transport::SipTransport`] uses `seq` for the jitter buffer,
/// `payload_type` to choose μ-law vs A-law decode, and `payload` as the G.711
/// bytes. `timestamp`/`ssrc`/`marker` are parsed for completeness (and tests).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RtpPacket {
    /// Payload type (0 = PCMU, 8 = PCMA for our G.711-only path).
    pub payload_type: u8,
    /// 16-bit sequence number (wraps; used to reorder/dedupe in the jitter buffer).
    pub seq: u16,
    /// 32-bit media timestamp (G.711: +160 per 20 ms frame).
    pub timestamp: u32,
    /// Synchronization source identifier.
    pub ssrc: u32,
    /// The RTP marker bit (M). Unused for G.711 but parsed for completeness.
    pub marker: bool,
    /// The codec payload bytes (G.711 μ-law / A-law samples, one byte each).
    pub payload: Vec<u8>,
}

/// Errors from parsing a raw datagram as RTP.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RtpParseError {
    /// The datagram is shorter than the 12-byte fixed header.
    TooShort,
    /// The version field was not 2.
    BadVersion(u8),
    /// The header (incl. CSRCs / extension) claimed more bytes than the datagram has.
    Truncated,
}

impl std::fmt::Display for RtpParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RtpParseError::TooShort => write!(f, "RTP datagram shorter than 12-byte header"),
            RtpParseError::BadVersion(v) => write!(f, "RTP version {v} != 2"),
            RtpParseError::Truncated => write!(f, "RTP header claims more bytes than present"),
        }
    }
}

/// Monotonic RTP sender state: holds the SSRC and the running sequence /
/// timestamp, advancing them per outbound packet.
///
/// Seeded with a random SSRC and random initial seq/ts as RFC 3550 §5.1
/// recommends. One per outbound media leg.
#[derive(Debug, Clone)]
pub struct RtpSender {
    ssrc: u32,
    payload_type: u8,
    seq: u16,
    timestamp: u32,
}

impl RtpSender {
    /// New sender for `payload_type` (0/8) with a random SSRC + random initial
    /// seq/ts (RFC 3550 §5.1).
    pub fn new(payload_type: u8) -> Self {
        Self::with_seeds(
            payload_type,
            rand::random::<u32>(),
            rand::random::<u16>(),
            rand::random::<u32>(),
        )
    }

    /// New sender with explicit seeds — used by tests for deterministic output.
    pub fn with_seeds(payload_type: u8, ssrc: u32, seq: u16, timestamp: u32) -> Self {
        Self {
            ssrc,
            payload_type,
            seq,
            timestamp,
        }
    }

    /// The SSRC this sender stamps on every packet.
    pub fn ssrc(&self) -> u32 {
        self.ssrc
    }

    /// Packetize one G.711 frame: build the 12-byte header over `payload`, then
    /// advance seq (+1) and timestamp (+payload length, i.e. one tick/sample).
    ///
    /// Returns the full datagram (header + payload) ready for `UdpSocket::send`.
    pub fn packetize(&mut self, payload: &[u8]) -> Vec<u8> {
        let mut pkt = Vec::with_capacity(RTP_HEADER_LEN + payload.len());
        // Byte 0: V(2) P(1) X(1) CC(4) — V=2, everything else 0.
        pkt.push(RTP_VERSION << 6);
        // Byte 1: M(1) PT(7) — marker 0, payload type in low 7 bits.
        pkt.push(self.payload_type & 0x7f);
        // Bytes 2..4: sequence number (big-endian).
        pkt.extend_from_slice(&self.seq.to_be_bytes());
        // Bytes 4..8: timestamp (big-endian).
        pkt.extend_from_slice(&self.timestamp.to_be_bytes());
        // Bytes 8..12: SSRC (big-endian).
        pkt.extend_from_slice(&self.ssrc.to_be_bytes());
        pkt.extend_from_slice(payload);

        // Advance for the next packet. G.711 clock == sample rate, and one G.711
        // byte == one sample, so the timestamp step is exactly the payload len.
        self.seq = self.seq.wrapping_add(1);
        self.timestamp = self.timestamp.wrapping_add(payload.len() as u32);
        pkt
    }
}

/// Parse a raw datagram as an RTP packet, validating V=2 and skipping any
/// CSRC list / header extension so we always return the real payload.
pub fn depacketize(buf: &[u8]) -> Result<RtpPacket, RtpParseError> {
    if buf.len() < RTP_HEADER_LEN {
        return Err(RtpParseError::TooShort);
    }
    let b0 = buf[0];
    let version = b0 >> 6;
    if version != RTP_VERSION {
        return Err(RtpParseError::BadVersion(version));
    }
    let has_extension = (b0 & 0x10) != 0;
    let csrc_count = (b0 & 0x0f) as usize;

    let b1 = buf[1];
    let marker = (b1 & 0x80) != 0;
    let payload_type = b1 & 0x7f;
    let seq = u16::from_be_bytes([buf[2], buf[3]]);
    let timestamp = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);
    let ssrc = u32::from_be_bytes([buf[8], buf[9], buf[10], buf[11]]);

    // Skip the CSRC list (4 bytes each).
    let mut offset = RTP_HEADER_LEN + csrc_count * 4;
    if buf.len() < offset {
        return Err(RtpParseError::Truncated);
    }

    // Skip a header extension if present: 4-byte (profile + length-in-32bit-words)
    // prefix, then `length` words of extension data.
    if has_extension {
        if buf.len() < offset + 4 {
            return Err(RtpParseError::Truncated);
        }
        let ext_words = u16::from_be_bytes([buf[offset + 2], buf[offset + 3]]) as usize;
        offset += 4 + ext_words * 4;
        if buf.len() < offset {
            return Err(RtpParseError::Truncated);
        }
    }

    Ok(RtpPacket {
        payload_type,
        seq,
        timestamp,
        ssrc,
        marker,
        payload: buf[offset..].to_vec(),
    })
}

/// A tiny fixed-depth playout **jitter buffer** for inbound RTP.
///
/// Telephony G.711 is a constant 20 ms / 160-byte cadence, which makes reordering
/// trivial: we hold up to [`JITTER_DEPTH`] packets sorted by sequence number, and
/// only release a packet once the buffer is full (so a packet that arrived up to
/// `JITTER_DEPTH - 1` slots out of order gets a chance to be reordered ahead of
/// release).
///
/// Behaviour:
/// - **In order** → released in order, one out per one in once primed.
/// - **Reordered** within the window → re-sorted, released in sequence order.
/// - **Duplicate** seq (already buffered, or already released this far) → dropped.
/// - **Late** (seq below the last released, accounting for 16-bit wrap) → dropped.
///
/// ## Depth
///
/// [`JITTER_DEPTH`] = **2** frames (= 40 ms at 20 ms/frame). Tuned down from 4
/// (80 ms) to shave mouth-to-ear latency for the real-time voice agent: a single
/// SIP hop (agent ↔ carrier) almost never reorders by more than one packet, so a
/// 2-deep window still corrects the common adjacent swap while halving the fixed
/// playout delay. The buffer never grows past this; it is a bounded reorder
/// window, not an elastic backlog. Bump back toward 4 if a lossier/multi-hop
/// path shows audible reorder-drops.
pub const JITTER_DEPTH: usize = 2;

/// Bounded reorder buffer over RTP sequence numbers (see [`JITTER_DEPTH`]).
#[derive(Debug)]
pub struct JitterBuffer {
    /// Buffered packets, kept sorted ascending by 16-bit sequence number.
    buf: Vec<RtpPacket>,
    /// Max packets held before we start releasing (the reorder window).
    depth: usize,
    /// Sequence number of the last packet released, if any. Used to drop late
    /// and duplicate packets. `None` until the first release.
    last_released: Option<u16>,
    /// Whether `last_released` has ever been set (distinguishes "nothing released
    /// yet" so the very first packets aren't treated as late).
    primed: bool,
}

impl JitterBuffer {
    /// A jitter buffer with the default [`JITTER_DEPTH`].
    pub fn new() -> Self {
        Self::with_depth(JITTER_DEPTH)
    }

    /// A jitter buffer with an explicit reorder depth (>= 1).
    pub fn with_depth(depth: usize) -> Self {
        Self {
            buf: Vec::with_capacity(depth + 1),
            depth: depth.max(1),
            last_released: None,
            primed: false,
        }
    }

    /// Clear all buffered packets and playout state. Used when the RTP **SSRC
    /// changes mid-call** — the new stream's sequence numbers are unrelated to the
    /// old one, so without a reset the new (low) seqs would be dropped as "late"
    /// against the stale `last_released`, causing an audible dropout. Call
    /// [`drain`](Self::drain) first if the old stream's trailing audio must be kept.
    pub fn reset(&mut self) {
        self.buf.clear();
        self.last_released = None;
        self.primed = false;
    }

    /// Whether `seq` is "older" than `reference` on the 16-bit sequence circle.
    ///
    /// RFC 1982 serial-number arithmetic: treat the half-range (32768) as the
    /// split between "before" and "after" so wrap-around (…65535 → 0) is handled.
    fn seq_lt(seq: u16, reference: u16) -> bool {
        // (seq - reference) interpreted as a signed 16-bit delta < 0 ⇒ seq is behind.
        seq != reference && seq.wrapping_sub(reference) > 0x8000
    }

    /// Push one inbound packet. Returns any packets ready for playout (in
    /// sequence order); usually 0 or 1, but a burst that overfills the window
    /// can release more than one.
    ///
    /// Drops duplicates and late packets (see the type docs).
    pub fn push(&mut self, pkt: RtpPacket) -> Vec<RtpPacket> {
        // Drop late packets: at/below the last released sequence (incl. dup of it).
        if self.primed {
            let last = self.last_released.unwrap();
            if pkt.seq == last || Self::seq_lt(pkt.seq, last) {
                return Vec::new();
            }
        }
        // Drop duplicates already sitting in the buffer.
        if self.buf.iter().any(|p| p.seq == pkt.seq) {
            return Vec::new();
        }

        // Insert keeping the buffer sorted ascending by seq (relative to the
        // release point, so wrapped sequences sort correctly).
        let pos = self
            .buf
            .iter()
            .position(|p| Self::seq_lt(pkt.seq, p.seq))
            .unwrap_or(self.buf.len());
        self.buf.insert(pos, pkt);

        // Release from the front while we're over the reorder window.
        let mut out = Vec::new();
        while self.buf.len() > self.depth {
            let p = self.buf.remove(0);
            self.last_released = Some(p.seq);
            self.primed = true;
            out.push(p);
        }
        out
    }

    /// Drain everything still buffered, in sequence order (end-of-call flush).
    pub fn drain(&mut self) -> Vec<RtpPacket> {
        let out: Vec<RtpPacket> = self.buf.drain(..).collect();
        if let Some(last) = out.last() {
            self.last_released = Some(last.seq);
            self.primed = true;
        }
        out
    }
}

impl Default for JitterBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pkt(seq: u16, payload: &[u8]) -> RtpPacket {
        RtpPacket {
            payload_type: PT_PCMU,
            seq,
            timestamp: seq as u32 * 160,
            ssrc: 0xDEAD_BEEF,
            marker: false,
            payload: payload.to_vec(),
        }
    }

    // ---- packetize / depacketize round-trip -------------------------------

    #[test]
    fn packetize_writes_a_well_formed_12_byte_header() {
        let mut s = RtpSender::with_seeds(PT_PCMU, 0x1122_3344, 0x0010, 0x0000_0100);
        let payload: Vec<u8> = (0..160u16).map(|i| i as u8).collect();
        let dgram = s.packetize(&payload);

        assert_eq!(dgram.len(), RTP_HEADER_LEN + 160);
        // V=2, P=0, X=0, CC=0.
        assert_eq!(dgram[0], 0x80);
        // M=0, PT=0 (PCMU).
        assert_eq!(dgram[1], 0x00);
        // seq, ts, ssrc big-endian.
        assert_eq!(u16::from_be_bytes([dgram[2], dgram[3]]), 0x0010);
        assert_eq!(
            u32::from_be_bytes([dgram[4], dgram[5], dgram[6], dgram[7]]),
            0x0000_0100
        );
        assert_eq!(
            u32::from_be_bytes([dgram[8], dgram[9], dgram[10], dgram[11]]),
            0x1122_3344
        );
        assert_eq!(&dgram[RTP_HEADER_LEN..], &payload[..]);
    }

    #[test]
    fn packetize_advances_seq_by_one_and_ts_by_sample_count() {
        let mut s = RtpSender::with_seeds(PT_PCMA, 7, 100, 1000);
        let a = depacketize(&s.packetize(&[0u8; 160])).unwrap();
        let b = depacketize(&s.packetize(&[0u8; 160])).unwrap();
        let c = depacketize(&s.packetize(&[0u8; 80])).unwrap(); // a short (10ms) frame
        assert_eq!((a.seq, a.timestamp), (100, 1000));
        assert_eq!((b.seq, b.timestamp), (101, 1160)); // +1 seq, +160 ts
        assert_eq!((c.seq, c.timestamp), (102, 1320)); // +1 seq, +160 ts
                                                       // PT + SSRC carried through unchanged.
        assert_eq!(a.payload_type, PT_PCMA);
        assert_eq!(a.ssrc, 7);
    }

    #[test]
    fn packetize_then_depacketize_preserves_all_fields_and_payload() {
        let mut s = RtpSender::with_seeds(PT_PCMU, 0xCAFE_BABE, 65530, 4_000_000_000);
        let payload: Vec<u8> = (0..160u32).map(|i| (i * 7) as u8).collect();
        let dgram = s.packetize(&payload);
        let p = depacketize(&dgram).unwrap();
        assert_eq!(p.payload_type, PT_PCMU);
        assert_eq!(p.seq, 65530);
        assert_eq!(p.timestamp, 4_000_000_000);
        assert_eq!(p.ssrc, 0xCAFE_BABE);
        assert!(!p.marker);
        assert_eq!(p.payload, payload);
    }

    #[test]
    fn seq_and_timestamp_wrap_around() {
        // seq starts at u16::MAX, ts near u32::MAX → both must wrap, not panic.
        let mut s = RtpSender::with_seeds(PT_PCMU, 1, u16::MAX, u32::MAX - 100);
        let a = depacketize(&s.packetize(&[0u8; 160])).unwrap();
        let b = depacketize(&s.packetize(&[0u8; 160])).unwrap();
        assert_eq!(a.seq, u16::MAX);
        assert_eq!(b.seq, 0); // wrapped
        assert_eq!(a.timestamp, u32::MAX - 100);
        // (MAX-100) + 160 = MAX + 60; wrapping past 2^32 (= MAX+1) lands on 59.
        assert_eq!(b.timestamp, 59);
    }

    #[test]
    fn depacketize_rejects_short_and_bad_version() {
        assert_eq!(depacketize(&[0u8; 11]), Err(RtpParseError::TooShort));
        // Version 1 in the top two bits (0x40).
        let mut bad = vec![0x40u8];
        bad.extend_from_slice(&[0u8; 11]);
        assert_eq!(depacketize(&bad), Err(RtpParseError::BadVersion(1)));
    }

    #[test]
    fn depacketize_skips_csrc_list() {
        // V=2, CC=2 → two 4-byte CSRCs before the payload.
        let mut dgram = vec![0x82u8, PT_PCMU];
        dgram.extend_from_slice(&5u16.to_be_bytes()); // seq
        dgram.extend_from_slice(&0u32.to_be_bytes()); // ts
        dgram.extend_from_slice(&0u32.to_be_bytes()); // ssrc
        dgram.extend_from_slice(&[0u8; 8]); // 2 CSRCs
        dgram.extend_from_slice(&[0xAA, 0xBB]); // payload
        let p = depacketize(&dgram).unwrap();
        assert_eq!(p.seq, 5);
        assert_eq!(p.payload, vec![0xAA, 0xBB]);
    }

    #[test]
    fn depacketize_skips_header_extension() {
        // V=2, X=1 (0x90). One extension word after the 4-byte ext prefix.
        let mut dgram = vec![0x90u8, PT_PCMU];
        dgram.extend_from_slice(&9u16.to_be_bytes()); // seq
        dgram.extend_from_slice(&0u32.to_be_bytes()); // ts
        dgram.extend_from_slice(&0u32.to_be_bytes()); // ssrc
        dgram.extend_from_slice(&0xBEEFu16.to_be_bytes()); // ext profile
        dgram.extend_from_slice(&1u16.to_be_bytes()); // ext length = 1 word
        dgram.extend_from_slice(&[1, 2, 3, 4]); // the one ext word
        dgram.extend_from_slice(&[0x77]); // payload
        let p = depacketize(&dgram).unwrap();
        assert_eq!(p.seq, 9);
        assert_eq!(p.payload, vec![0x77]);
    }

    // ---- jitter buffer ----------------------------------------------------

    /// Helper: feed a sequence of packets, collect everything released in order.
    fn run(jb: &mut JitterBuffer, seqs: &[u16]) -> Vec<u16> {
        let mut out = Vec::new();
        for &s in seqs {
            for p in jb.push(pkt(s, &[s as u8])) {
                out.push(p.seq);
            }
        }
        out
    }

    // ── depacketize robustness (no panic on hostile/short input) ────────────────
    #[test]
    fn depacketize_exactly_12_bytes_is_empty_payload_no_panic() {
        let mut d = vec![0x80u8, PT_PCMU];
        d.extend_from_slice(&7u16.to_be_bytes());
        d.extend_from_slice(&0u32.to_be_bytes());
        d.extend_from_slice(&0u32.to_be_bytes());
        let p = depacketize(&d).unwrap();
        assert_eq!(p.seq, 7);
        assert!(p.payload.is_empty());
    }

    #[test]
    fn depacketize_rejects_truncated_csrc_list() {
        // CC=2 claims 8 CSRC bytes but the datagram stops after the 12-byte header.
        let mut d = vec![0x82u8, PT_PCMU];
        d.extend_from_slice(&1u16.to_be_bytes());
        d.extend_from_slice(&0u32.to_be_bytes());
        d.extend_from_slice(&0u32.to_be_bytes());
        assert_eq!(depacketize(&d), Err(RtpParseError::Truncated));
    }

    #[test]
    fn depacketize_rejects_truncated_header_extension() {
        // X=1, ext length claims 4 words but none follow.
        let mut d = vec![0x90u8, PT_PCMU];
        d.extend_from_slice(&2u16.to_be_bytes());
        d.extend_from_slice(&0u32.to_be_bytes());
        d.extend_from_slice(&0u32.to_be_bytes());
        d.extend_from_slice(&0xBEEFu16.to_be_bytes());
        d.extend_from_slice(&4u16.to_be_bytes());
        assert_eq!(depacketize(&d), Err(RtpParseError::Truncated));
    }

    #[test]
    fn depacketize_parses_marker_bit_set() {
        let mut d = vec![0x80u8, 0x80 | PT_PCMU];
        d.extend_from_slice(&3u16.to_be_bytes());
        d.extend_from_slice(&0u32.to_be_bytes());
        d.extend_from_slice(&0u32.to_be_bytes());
        d.push(0xAA);
        let p = depacketize(&d).unwrap();
        assert!(p.marker, "marker bit should parse as true");
        assert_eq!(p.payload_type, PT_PCMU);
        assert_eq!(p.payload, vec![0xAA]);
    }

    #[test]
    fn depacketize_does_not_panic_on_arbitrary_short_buffers() {
        for n in 0..RTP_HEADER_LEN {
            assert_eq!(depacketize(&vec![0xFFu8; n]), Err(RtpParseError::TooShort));
        }
    }

    // ── sender seq (u16) + timestamp (u32) wrap independently ───────────────────
    #[test]
    fn sender_seq_and_ts_wrap_independently() {
        let mut s = RtpSender::with_seeds(PT_PCMU, 1, u16::MAX - 1, 0);
        let seqs: Vec<u16> = (0..4)
            .map(|_| depacketize(&s.packetize(&[0u8; 160])).unwrap().seq)
            .collect();
        assert_eq!(seqs, vec![u16::MAX - 1, u16::MAX, 0, 1]);

        let mut s2 = RtpSender::with_seeds(PT_PCMU, 1, 0, u32::MAX - 160);
        assert_eq!(
            depacketize(&s2.packetize(&[0u8; 160])).unwrap().timestamp,
            u32::MAX - 160
        );
        assert_eq!(
            depacketize(&s2.packetize(&[0u8; 160])).unwrap().timestamp,
            u32::MAX
        );
        assert_eq!(
            depacketize(&s2.packetize(&[0u8; 160])).unwrap().timestamp,
            159
        ); // wrapped
    }

    // ── jitter: depth edges + lost-packet handling ──────────────────────────────
    #[test]
    fn jitter_depth_one_releases_immediately_in_order() {
        let mut jb = JitterBuffer::with_depth(1);
        assert_eq!(run(&mut jb, &[1, 2, 3, 4]), vec![1, 2, 3]);
        let tail: Vec<u16> = jb.drain().into_iter().map(|p| p.seq).collect();
        assert_eq!(tail, vec![4]);
    }

    #[test]
    fn jitter_with_depth_zero_clamps_to_one() {
        let mut jb = JitterBuffer::with_depth(0);
        assert_eq!(run(&mut jb, &[5, 6]), vec![5]);
    }

    #[test]
    fn jitter_lost_packet_does_not_stall_and_late_arrival_is_dropped() {
        // 3 is lost: 1,2,4,5,6 arrive. Window drains in order; a late 3 is dropped.
        let mut jb = JitterBuffer::with_depth(2);
        assert_eq!(run(&mut jb, &[1, 2, 4, 5, 6]), vec![1, 2, 4]);
        assert_eq!(jb.push(pkt(3, &[3])), vec![], "late lost packet dropped");
    }

    // ── SSRC-change fix: reset() clears state so a fresh stream starts clean ────
    #[test]
    fn jitter_reset_clears_state_for_a_fresh_stream() {
        let mut jb = JitterBuffer::with_depth(2);
        let _ = run(&mut jb, &[40000, 40001, 40002, 40003]); // primes + leaves 40002/40003 buffered
        jb.reset(); // what rx_loop does on an SSRC change
                    // Buffered old-stream packets are gone and last_released is cleared, so a
                    // brand-new stream (new SSRC) primes from scratch and releases in order —
                    // releasing exactly the new seqs proves the old 40002/40003 didn't leak.
        assert_eq!(run(&mut jb, &[100, 101, 102, 103]), vec![100, 101]);
    }

    #[test]
    fn jitter_in_order_releases_in_order_after_priming() {
        let mut jb = JitterBuffer::with_depth(4);
        // 8 in-order packets through a depth-4 window: first 4 stay buffered,
        // then one out per one in → seqs 1..=4 released.
        let released = run(&mut jb, &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(released, vec![1, 2, 3, 4]);
        // Draining the window yields the held tail in order.
        let tail: Vec<u16> = jb.drain().into_iter().map(|p| p.seq).collect();
        assert_eq!(tail, vec![5, 6, 7, 8]);
    }

    #[test]
    fn jitter_reorders_within_window() {
        let mut jb = JitterBuffer::with_depth(4);
        // 3 and 4 swapped on the wire; window re-sorts them before release.
        let released = run(&mut jb, &[1, 2, 4, 3, 5, 6, 7]);
        assert_eq!(released, vec![1, 2, 3]); // 3 emitted before 4 despite arriving later
        let tail: Vec<u16> = jb.drain().into_iter().map(|p| p.seq).collect();
        assert_eq!(tail, vec![4, 5, 6, 7]);
    }

    #[test]
    fn jitter_drops_duplicates() {
        let mut jb = JitterBuffer::with_depth(4);
        // Duplicate 2 (still in the window) is dropped; output has no repeat.
        let released = run(&mut jb, &[1, 2, 2, 3, 4, 5, 6]);
        assert_eq!(released, vec![1, 2]);
        let tail: Vec<u16> = jb.drain().into_iter().map(|p| p.seq).collect();
        assert_eq!(tail, vec![3, 4, 5, 6]);
    }

    #[test]
    fn jitter_drops_late_packet_below_release_point() {
        let mut jb = JitterBuffer::with_depth(2);
        // Prime + release up through some seqs, then a stale low seq arrives late.
        let released = run(&mut jb, &[10, 11, 12, 13, 14]);
        assert_eq!(released, vec![10, 11, 12]); // depth 2 → release once over 2
                                                // 11 is now below last_released (12) → dropped, releases nothing.
        assert_eq!(jb.push(pkt(11, &[0])), vec![]);
        // A fresh in-window late-ish dup of an already-released seq also drops.
        assert_eq!(jb.push(pkt(12, &[0])), vec![]);
    }

    #[test]
    fn jitter_handles_seq_wraparound() {
        let mut jb = JitterBuffer::with_depth(2);
        // Sequence wraps 65534, 65535, 0, 1 — must stay in order across the wrap.
        let released = run(&mut jb, &[65534, 65535, 0, 1, 2]);
        assert_eq!(released, vec![65534, 65535, 0]);
        let tail: Vec<u16> = jb.drain().into_iter().map(|p| p.seq).collect();
        assert_eq!(tail, vec![1, 2]);
    }

    #[test]
    fn jitter_reordered_across_wrap_is_sorted() {
        let mut jb = JitterBuffer::with_depth(3);
        // 0 arrives before 65535 (reordered across the wrap); window sorts them.
        let released = run(&mut jb, &[65534, 0, 65535, 1, 2, 3]);
        assert_eq!(released, vec![65534, 65535, 0]);
        let tail: Vec<u16> = jb.drain().into_iter().map(|p| p.seq).collect();
        assert_eq!(tail, vec![1, 2, 3]);
    }
}
