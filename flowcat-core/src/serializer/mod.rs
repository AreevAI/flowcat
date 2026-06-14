// SPDX-License-Identifier: Apache-2.0
//
//! Per-carrier WebSocket framing. **Pure** (no I/O): turns inbound frames into
//! normalized [`SerIn`] events and encodes outbound audio/clears into [`WsOut`].
//!
//! See DESIGN.md "Trait contracts". The one concrete serializer is `plivo`
//! (Plivo `<Stream>` JSON+base64). SIP/RTP carriers do not use a WS-media
//! serializer — they ride native SIP/RTP ([`crate::sip`]), so the former
//! FreeSWITCH `gateway` serializer was removed (see SIP-DESIGN.md §4).

pub mod plivo;

use crate::types::{AudioChunk, SerIn, WsIn, WsOut};

pub use plivo::PlivoSerializer;

/// Carrier-specific WS framing. Implementations are pure functions over frames
/// (no sockets, no async) so they are trivially unit-testable.
pub trait MediaSerializer: Send {
    /// Normalize one inbound frame into a [`SerIn`] event.
    fn on_message(&mut self, msg: &WsIn) -> SerIn;

    /// Encode an outbound audio chunk into a frame to send back to the carrier.
    fn encode_audio(&self, chunk: &AudioChunk) -> WsOut;

    /// Encode a "clear/flush queued audio" control frame for barge-in, if the
    /// carrier supports one.
    fn encode_clear(&self) -> Option<WsOut>;

    /// The carrier-side audio sample rate (e.g. 8000 for telephony μ-law).
    fn carrier_rate(&self) -> u32;
}
