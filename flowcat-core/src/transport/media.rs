// SPDX-License-Identifier: Apache-2.0
//
//! The `MediaTransport` trait — Flowcat's transport-agnostic *media* seam.
//!
//! Where [`MediaSocket`](crate::transport::MediaSocket) is a raw WS frame pipe,
//! `MediaTransport` is one level up: it speaks **decoded audio + lifecycle
//! events**, so the [S2S pipeline](crate::pipeline::build_s2s_task) does not care
//! whether the audio arrived as carrier WebSocket frames (Plivo) or RTP (native
//! SIP). See
//! SIP-DESIGN.md §1 "The transport seam".
//!
//! Two implementations plug in here:
//! - [`WsCarrierTransport`](crate::transport::WsCarrierTransport) adapts the WS
//!   path (`MediaSocket` + `MediaSerializer`) — used for Plivo.
//! - the native-SIP `SipTransport` (separate module, not yet built) will decode
//!   RTP into the same `MediaIn` shape.

use async_trait::async_trait;

use crate::error::FlowcatError;
use crate::types::AudioChunk;

/// An inbound media-lifecycle event, normalized across every transport.
///
/// The carrier-/protocol-specific framing (Plivo `start`/`media`/`stop`, an SDP
/// answer, an RTP packet, a BYE) is collapsed into these three cases by the
/// concrete [`MediaTransport`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaIn {
    /// The media stream started; carries the carrier-side call identifier.
    StreamStart {
        /// Carrier-side call id (Plivo `callId`, SIP `Call-ID`, …).
        call_id: String,
    },
    /// Decoded inbound audio from the caller, at the transport's `carrier_rate`.
    Audio(AudioChunk),
    /// The call ended (carrier stop, peer hangup / BYE, socket close).
    Stop,
}

/// A transport-agnostic bidirectional media channel for one call.
///
/// Driven by the [S2S pipeline](crate::pipeline::build_s2s_task): `recv` yields the
/// next inbound [`MediaIn`]; `send_audio` plays bot audio back to the caller; `send_clear`
/// flushes buffered playback on barge-in; `carrier_rate` is the sample rate that
/// inbound audio arrives at and outbound audio must be supplied at.
#[async_trait]
pub trait MediaTransport: Send {
    /// Next inbound event: the call started, a chunk of caller audio (at
    /// [`carrier_rate`](MediaTransport::carrier_rate)), or the call ended.
    /// `None` once the transport is closed/exhausted.
    async fn recv(&mut self) -> Option<MediaIn>;

    /// Play bot audio (at [`carrier_rate`](MediaTransport::carrier_rate)) to the
    /// caller.
    async fn send_audio(&mut self, chunk: AudioChunk) -> Result<(), FlowcatError>;

    /// Barge-in: flush any buffered playback. No-op where unsupported (e.g. RTP
    /// has no flush — barge-in there is simply "stop sending").
    async fn send_clear(&mut self) -> Result<(), FlowcatError>;

    /// The carrier-side audio sample rate (e.g. 8000 for telephony G.711).
    /// Inbound [`MediaIn::Audio`] is at this rate, and outbound
    /// [`send_audio`](MediaTransport::send_audio) chunks must be too.
    fn carrier_rate(&self) -> u32;
}
