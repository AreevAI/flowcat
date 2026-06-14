// SPDX-License-Identifier: Apache-2.0
//
//! Minimal SDP (RFC 4566) for G.711 telephony — just enough to offer/answer a
//! single `audio` stream over RTP/AVP with PCMU/PCMA.
//!
//! Hand-rolled (no `sdp-rs` dep) because our needs are tiny and fixed: one media
//! line, two possible codecs, ptime 20, sendrecv (see SIP-DESIGN.md §2). We build
//! offers/answers and parse a peer's offer/answer to extract its media IP, port,
//! and chosen codec.
//!
//! Build output shape (one media stream, IPv4):
//! ```text
//! v=0
//! o=- 0 0 IN IP4 <ip>
//! s=flowcat
//! c=IN IP4 <ip>
//! t=0 0
//! m=audio <port> RTP/AVP 0 8      (offer: both; answer: the one chosen)
//! a=rtpmap:0 PCMU/8000
//! a=rtpmap:8 PCMA/8000
//! a=ptime:20
//! a=sendrecv
//! ```

use std::net::Ipv4Addr;

use super::rtp::{PT_PCMA, PT_PCMU};

/// The G.711 codec negotiated for a stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum G711Codec {
    /// μ-law, RTP payload type 0.
    Pcmu,
    /// A-law, RTP payload type 8.
    Pcma,
}

impl G711Codec {
    /// The static RTP/AVP payload type number (0 = PCMU, 8 = PCMA).
    pub fn payload_type(self) -> u8 {
        match self {
            G711Codec::Pcmu => PT_PCMU,
            G711Codec::Pcma => PT_PCMA,
        }
    }

    /// The `a=rtpmap` encoding name (`PCMU` / `PCMA`).
    pub fn rtpmap_name(self) -> &'static str {
        match self {
            G711Codec::Pcmu => "PCMU",
            G711Codec::Pcma => "PCMA",
        }
    }

    /// Map a static payload-type number to a G.711 codec, if it is one.
    pub fn from_payload_type(pt: u8) -> Option<Self> {
        match pt {
            PT_PCMU => Some(G711Codec::Pcmu),
            PT_PCMA => Some(G711Codec::Pcma),
            _ => None,
        }
    }
}

/// The audio media parameters parsed from / built into an SDP body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SdpMedia {
    /// Connection IPv4 address the peer receives RTP on (`c=` / media-level).
    pub ip: Ipv4Addr,
    /// UDP port the peer receives RTP on (`m=audio <port> …`).
    pub port: u16,
    /// The chosen / preferred G.711 codec.
    pub codec: G711Codec,
}

/// Build an SDP **offer** advertising both PCMU and PCMA at `ip:port`.
///
/// PCMU is listed first (our preference). ptime 20, sendrecv.
pub fn build_offer(ip: Ipv4Addr, port: u16) -> String {
    // m= line lists both PTs (0 8); rtpmap for each.
    build_sdp(ip, port, &[G711Codec::Pcmu, G711Codec::Pcma])
}

/// Build an SDP **answer** committing to exactly one `codec` at `ip:port`.
pub fn build_answer(ip: Ipv4Addr, port: u16, codec: G711Codec) -> String {
    build_sdp(ip, port, &[codec])
}

/// Shared SDP body builder for an audio/RTP-AVP stream listing `codecs` (in
/// order) on the `m=` line, with an `a=rtpmap` per codec, ptime 20, sendrecv.
fn build_sdp(ip: Ipv4Addr, port: u16, codecs: &[G711Codec]) -> String {
    let pts: Vec<String> = codecs
        .iter()
        .map(|c| c.payload_type().to_string())
        .collect();
    let mut s = String::new();
    s.push_str("v=0\r\n");
    s.push_str(&format!("o=- 0 0 IN IP4 {ip}\r\n"));
    s.push_str("s=flowcat\r\n");
    s.push_str(&format!("c=IN IP4 {ip}\r\n"));
    s.push_str("t=0 0\r\n");
    s.push_str(&format!("m=audio {port} RTP/AVP {}\r\n", pts.join(" ")));
    for c in codecs {
        s.push_str(&format!(
            "a=rtpmap:{} {}/8000\r\n",
            c.payload_type(),
            c.rtpmap_name()
        ));
    }
    s.push_str("a=ptime:20\r\n");
    s.push_str("a=sendrecv\r\n");
    s
}

/// Errors parsing a peer's SDP body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SdpParseError {
    /// No `m=audio` line was found.
    NoAudioMedia,
    /// No usable connection (`c=IN IP4 …`) address was found.
    NoConnection,
    /// The `m=audio` line had no port / an unparseable port.
    BadPort,
    /// The peer offered no G.711 (PCMU/PCMA) payload type we support.
    NoSupportedCodec,
}

impl std::fmt::Display for SdpParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SdpParseError::NoAudioMedia => write!(f, "SDP has no m=audio line"),
            SdpParseError::NoConnection => write!(f, "SDP has no c=IN IP4 connection address"),
            SdpParseError::BadPort => write!(f, "SDP m=audio line has no/invalid port"),
            SdpParseError::NoSupportedCodec => write!(f, "SDP offers no PCMU/PCMA codec"),
        }
    }
}

/// Parse a peer's SDP, returning the audio IP, port, and the **first** G.711
/// codec listed on the `m=audio` line that we support.
///
/// Codec selection follows the order the peer lists payload types on the `m=`
/// line (its preference). For an offer that lists `0 8` we therefore pick PCMU;
/// `8 0` would pick PCMA. A media-level `c=` line overrides a session-level one.
pub fn parse(sdp: &str) -> Result<SdpMedia, SdpParseError> {
    let mut session_ip: Option<Ipv4Addr> = None;
    let mut media_ip: Option<Ipv4Addr> = None;
    let mut port: Option<u16> = None;
    let mut media_pts: Vec<u8> = Vec::new();
    let mut in_audio = false;
    let mut seen_audio = false;

    for line in sdp.lines() {
        let line = line.trim_end_matches('\r');
        if let Some(rest) = line.strip_prefix("c=") {
            // c=IN IP4 <addr>. Before the first m= → session-level; after → the
            // current media's connection.
            if let Some(addr) = parse_connection_ipv4(rest) {
                if in_audio {
                    media_ip = Some(addr);
                } else {
                    session_ip = Some(addr);
                }
            }
        } else if let Some(rest) = line.strip_prefix("m=") {
            if let Some(audio) = rest.strip_prefix("audio ") {
                if seen_audio {
                    // We commit to exactly one audio stream. A second m=audio
                    // (RFC 4566 allows multiple) must NOT clobber the first
                    // stream's port or merge its payload types — ignore it, and
                    // leave audio scope so a trailing c= isn't mis-attributed to it.
                    in_audio = false;
                    continue;
                }
                // First (and for us only) audio media line.
                in_audio = true;
                seen_audio = true;
                let mut parts = audio.split_whitespace();
                port = parts.next().and_then(|p| p.parse::<u16>().ok());
                // Skip the transport token ("RTP/AVP"); the rest are payload types.
                let _transport = parts.next();
                for pt in parts {
                    if let Ok(n) = pt.parse::<u8>() {
                        media_pts.push(n);
                    }
                }
            } else {
                // A different (e.g. video) media section starts — leave audio scope.
                in_audio = false;
            }
        }
    }

    if !seen_audio {
        return Err(SdpParseError::NoAudioMedia);
    }
    let ip = media_ip.or(session_ip).ok_or(SdpParseError::NoConnection)?;
    let port = port.ok_or(SdpParseError::BadPort)?;
    if port == 0 {
        // Port 0 means the stream is declined / inactive — no media to send.
        return Err(SdpParseError::BadPort);
    }
    // Pick the first PCMU/PCMA in the peer's listed order (their preference).
    let codec = media_pts
        .iter()
        .find_map(|&pt| G711Codec::from_payload_type(pt))
        .ok_or(SdpParseError::NoSupportedCodec)?;

    Ok(SdpMedia { ip, port, codec })
}

/// Parse the IPv4 address out of a `c=` line body of the form `IN IP4 <addr>`.
fn parse_connection_ipv4(rest: &str) -> Option<Ipv4Addr> {
    let mut parts = rest.split_whitespace();
    match (parts.next(), parts.next(), parts.next()) {
        (Some("IN"), Some("IP4"), Some(addr)) => {
            // Strip any TTL/count suffix (e.g. "224.2.1.1/127") for multicast.
            addr.split('/').next()?.parse().ok()
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ip(a: u8, b: u8, c: u8, d: u8) -> Ipv4Addr {
        Ipv4Addr::new(a, b, c, d)
    }

    // ── symmetric-RTP: answer must carry the node's public IP + bound port ──────
    #[test]
    fn answer_carries_public_ip_port_sendrecv_ptime_and_rtpmap() {
        let pub_ip = ip(34, 124, 207, 59);
        let sdp = build_answer(pub_ip, 41234, G711Codec::Pcmu);
        assert!(
            sdp.contains("c=IN IP4 34.124.207.59\r\n"),
            "answer c= must be the public IP"
        );
        assert!(
            sdp.contains("o=- 0 0 IN IP4 34.124.207.59\r\n"),
            "o= origin must be the public IP"
        );
        assert!(
            sdp.contains("m=audio 41234 RTP/AVP 0\r\n"),
            "m= must carry the bound RTP port + PCMU only"
        );
        assert!(sdp.contains("a=rtpmap:0 PCMU/8000\r\n"));
        assert!(sdp.contains("a=ptime:20\r\n"));
        assert!(sdp.contains("a=sendrecv\r\n"));
        assert!(
            !sdp.contains("PCMA"),
            "answer must commit to exactly one codec"
        );
    }

    // ── codec selection from a realistic carrier offer (PCMU + DTMF) ────────────
    #[test]
    fn parse_zadarma_style_offer_selects_pcmu() {
        let sdp = "v=0\r\no=Zadarma 12345 1 IN IP4 185.45.152.42\r\ns=Zadarma\r\n\
                   c=IN IP4 185.45.152.42\r\nt=0 0\r\nm=audio 18722 RTP/AVP 0 101\r\n\
                   a=rtpmap:0 PCMU/8000\r\na=rtpmap:101 telephone-event/8000\r\n\
                   a=fmtp:101 0-16\r\na=ptime:20\r\na=sendrecv\r\n";
        let m = parse(sdp).unwrap();
        assert_eq!(m.codec, G711Codec::Pcmu);
        assert_eq!(m.ip, ip(185, 45, 152, 42));
        assert_eq!(m.port, 18722);
    }

    // ── an offer with NO G.711 must be rejected, not silently mis-selected ──────
    #[test]
    fn parse_rejects_offer_without_g711() {
        let sdp = "v=0\r\nc=IN IP4 9.9.9.9\r\nm=audio 5000 RTP/AVP 111 101\r\n\
                   a=rtpmap:111 opus/48000/2\r\na=rtpmap:101 telephone-event/8000\r\n";
        assert_eq!(parse(sdp), Err(SdpParseError::NoSupportedCodec));
    }

    #[test]
    fn parse_rejects_dtmf_only_offer() {
        let sdp = "v=0\r\nc=IN IP4 1.2.3.4\r\nm=audio 5000 RTP/AVP 101\r\n\
                   a=rtpmap:101 telephone-event/8000\r\n";
        assert_eq!(parse(sdp), Err(SdpParseError::NoSupportedCodec));
    }

    // ── line-ending + IPv6 robustness ───────────────────────────────────────────
    #[test]
    fn parse_tolerates_lf_only_line_endings() {
        let sdp = "v=0\nc=IN IP4 7.7.7.7\nm=audio 5000 RTP/AVP 0\na=rtpmap:0 PCMU/8000\n";
        let m = parse(sdp).unwrap();
        assert_eq!(m.ip, ip(7, 7, 7, 7));
        assert_eq!(m.port, 5000);
        assert_eq!(m.codec, G711Codec::Pcmu);
    }

    #[test]
    fn parse_ipv6_only_connection_yields_no_connection() {
        let sdp = "v=0\r\nc=IN IP6 2001:db8::1\r\nm=audio 5000 RTP/AVP 0\r\n";
        assert_eq!(parse(sdp), Err(SdpParseError::NoConnection));
    }

    #[test]
    fn parse_prefers_ipv4_media_connection_over_ipv6_session() {
        let sdp = "v=0\r\nc=IN IP6 2001:db8::1\r\nt=0 0\r\nm=audio 9000 RTP/AVP 0\r\n\
                   c=IN IP4 172.16.5.5\r\na=rtpmap:0 PCMU/8000\r\n";
        let m = parse(sdp).unwrap();
        assert_eq!(m.ip, ip(172, 16, 5, 5));
    }

    // ── hostile/malformed bodies must return Err, never panic ───────────────────
    #[test]
    fn parse_malformed_inputs_do_not_panic() {
        assert!(parse("").is_err());
        assert!(parse("this is not sdp at all\r\n\r\n").is_err());
        assert!(parse("c=IN IP4 1.1.1.1\r\nm=audio\r\n").is_err());
        assert!(parse("c=IN IP4 1.1.1.1\r\nm=audio abc RTP/AVP 0\r\n").is_err());
        assert!(parse("c=IN IP4 1.1.1.1\r\nm=audio 5000 RTP/AVP\r\n").is_err());
        assert!(parse("c=IN IP4 1.1.1.1\r\nm=audio 70000 RTP/AVP 0\r\n").is_err());
        assert!(parse("m=\r\n").is_err());
        assert!(parse("c=IN IP4\r\nm=audio 5000 RTP/AVP 0\r\n").is_err());
    }

    // ── regression: a 2nd m=audio must NOT clobber the first stream ─────────────
    #[test]
    fn parse_uses_first_audio_section_only() {
        let sdp = "v=0\r\nc=IN IP4 10.0.0.1\r\nm=audio 8000 RTP/AVP 0\r\n\
                   a=rtpmap:0 PCMU/8000\r\nm=audio 9000 RTP/AVP 8\r\n\
                   a=rtpmap:8 PCMA/8000\r\n";
        let m = parse(sdp).unwrap();
        assert_eq!(
            m.port, 8000,
            "second m=audio must not clobber the first stream's port"
        );
        assert_eq!(
            m.codec,
            G711Codec::Pcmu,
            "PTs from the second m=audio must not leak in"
        );
    }

    #[test]
    fn offer_lists_both_codecs_with_ptime_and_sendrecv() {
        let sdp = build_offer(ip(10, 0, 0, 5), 40000);
        assert!(sdp.contains("m=audio 40000 RTP/AVP 0 8\r\n"));
        assert!(sdp.contains("a=rtpmap:0 PCMU/8000\r\n"));
        assert!(sdp.contains("a=rtpmap:8 PCMA/8000\r\n"));
        assert!(sdp.contains("c=IN IP4 10.0.0.5\r\n"));
        assert!(sdp.contains("a=ptime:20\r\n"));
        assert!(sdp.contains("a=sendrecv\r\n"));
    }

    #[test]
    fn answer_commits_to_one_codec() {
        let sdp = build_answer(ip(192, 168, 1, 9), 5004, G711Codec::Pcma);
        assert!(sdp.contains("m=audio 5004 RTP/AVP 8\r\n"));
        assert!(sdp.contains("a=rtpmap:8 PCMA/8000\r\n"));
        assert!(!sdp.contains("PCMU"));
    }

    #[test]
    fn build_then_parse_offer_round_trips_pcmu() {
        // Our own offer lists 0 first → parse picks PCMU.
        let sdp = build_offer(ip(203, 0, 113, 7), 12345);
        let m = parse(&sdp).unwrap();
        assert_eq!(m.ip, ip(203, 0, 113, 7));
        assert_eq!(m.port, 12345);
        assert_eq!(m.codec, G711Codec::Pcmu);
        assert_eq!(m.codec.payload_type(), 0);
    }

    #[test]
    fn build_then_parse_answer_round_trips_pcma() {
        let sdp = build_answer(ip(8, 8, 4, 4), 6000, G711Codec::Pcma);
        let m = parse(&sdp).unwrap();
        assert_eq!(m.ip, ip(8, 8, 4, 4));
        assert_eq!(m.port, 6000);
        assert_eq!(m.codec, G711Codec::Pcma);
    }

    #[test]
    fn parse_picks_first_listed_payload_type_pcma_when_peer_prefers_it() {
        // Peer lists 8 before 0 → its preference is PCMA.
        let sdp = "v=0\r\n\
                   o=- 0 0 IN IP4 1.2.3.4\r\n\
                   s=-\r\n\
                   c=IN IP4 1.2.3.4\r\n\
                   t=0 0\r\n\
                   m=audio 7000 RTP/AVP 8 0\r\n\
                   a=rtpmap:8 PCMA/8000\r\n\
                   a=rtpmap:0 PCMU/8000\r\n";
        let m = parse(sdp).unwrap();
        assert_eq!(m.codec, G711Codec::Pcma);
        assert_eq!(m.port, 7000);
    }

    #[test]
    fn parse_skips_unsupported_codecs_and_finds_g711() {
        // Telephony-event (101) + opus (111) listed before G.711 PCMU (0).
        let sdp = "v=0\r\n\
                   c=IN IP4 9.9.9.9\r\n\
                   m=audio 5555 RTP/AVP 111 101 0\r\n\
                   a=rtpmap:111 opus/48000/2\r\n\
                   a=rtpmap:101 telephone-event/8000\r\n\
                   a=rtpmap:0 PCMU/8000\r\n";
        let m = parse(sdp).unwrap();
        assert_eq!(m.codec, G711Codec::Pcmu);
    }

    #[test]
    fn media_level_connection_overrides_session_level() {
        let sdp = "v=0\r\n\
                   o=- 0 0 IN IP4 10.0.0.1\r\n\
                   c=IN IP4 10.0.0.1\r\n\
                   t=0 0\r\n\
                   m=audio 9000 RTP/AVP 0\r\n\
                   c=IN IP4 172.16.5.5\r\n\
                   a=rtpmap:0 PCMU/8000\r\n";
        let m = parse(sdp).unwrap();
        // The media-level c= (after m=audio) wins over the session-level one.
        assert_eq!(m.ip, ip(172, 16, 5, 5));
    }

    #[test]
    fn parse_errors_on_missing_pieces() {
        // No m=audio at all.
        assert_eq!(
            parse("v=0\r\nc=IN IP4 1.1.1.1\r\n"),
            Err(SdpParseError::NoAudioMedia)
        );
        // m=audio but no connection line anywhere.
        assert_eq!(
            parse("v=0\r\nm=audio 5000 RTP/AVP 0\r\n"),
            Err(SdpParseError::NoConnection)
        );
        // Port 0 = declined stream.
        assert_eq!(
            parse("v=0\r\nc=IN IP4 1.1.1.1\r\nm=audio 0 RTP/AVP 0\r\n"),
            Err(SdpParseError::BadPort)
        );
        // Only an unsupported codec offered.
        assert_eq!(
            parse("v=0\r\nc=IN IP4 1.1.1.1\r\nm=audio 5000 RTP/AVP 111\r\n"),
            Err(SdpParseError::NoSupportedCodec)
        );
    }
}
