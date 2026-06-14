// SPDX-License-Identifier: Apache-2.0
//
//! Native SIP/RTP: Flowcat speaks SIP directly (no FreeSWITCH).
//!
//! This module makes a `flowcat-core` process a SIP user agent on a trunk:
//! REGISTER (with digest auth + periodic refresh), accept inbound INVITEs, and
//! originate outbound INVITEs — then carry the call's audio over hand-rolled
//! RTP, surfaced through the same [`MediaTransport`](crate::transport::MediaTransport)
//! seam the WS/Plivo path uses. See SIP-DESIGN.md §2.
//!
//! Split:
//! - **Signaling** ([`agent`]) is delegated to `rsipstack` (REGISTER/INVITE/ACK/
//!   BYE transactions + dialogs, digest auth). rsipstack re-exports the `rsip`
//!   message/header types as `rsipstack::sip`.
//! - **RTP** ([`rtp`]) and **SDP** ([`sdp`]) are hand-rolled (deterministic,
//!   fully unit-tested, no extra media dep) over a plain `tokio::net::UdpSocket`.
//! - [`transport::SipTransport`] bridges RTP ↔ decoded `MediaIn::Audio` for one
//!   dialog, implementing [`MediaTransport`](crate::transport::MediaTransport).

pub mod agent;
pub mod rtp;
pub mod sdp;
pub mod transport;

pub use agent::{
    InboundInvite, SipAgent, SipConfig, DEFAULT_RTP_PORT_BASE, DEFAULT_RTP_PORT_TRIES,
};
pub use rtp::{depacketize, JitterBuffer, RtpPacket, RtpSender, JITTER_DEPTH};
pub use sdp::{G711Codec, SdpMedia};
pub use transport::SipTransport;
