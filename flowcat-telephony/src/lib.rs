// SPDX-License-Identifier: Apache-2.0
//
//! # flowcat-telephony
//!
//! Per-carrier WebSocket **`FrameSerializer`s** + DTMF for the Flowcat runtime:
//! Twilio, Telnyx, Plivo, Exotel, Vonage, Genesys, Asterisk, Cloudonix, Vobiz.
//!
//! Each serializer impls the frozen
//! [`MediaSerializer`](flowcat_core::MediaSerializer) trait (pure framing, no
//! I/O). The DTMF frames (`InputDtmf`/`OutputDtmf`/`KeypadEntry`) are already in
//! the core [`Frame`](flowcat_core::Frame) enum; this crate adds the RFC2833
//! (always-on) and in-band Goertzel (`dtmf-inband`) detectors. flowcat-core never
//! depends back on this crate.
//!
//! Each carrier is behind its own (dep-less) feature; `default = ["plivo"]`
//! mirrors the live carrier today. Carrier serializers are implemented in their
//! declared modules — no new `mod` decls or dep lines are added. SECURITY:
//! serializer signatures + DTMF need a security-review sign-off.

/// Twilio `<Stream>` serializer. Behind `twilio`.
#[cfg(feature = "twilio")]
pub mod twilio;

/// Telnyx media serializer. Behind `telnyx`.
#[cfg(feature = "telnyx")]
pub mod telnyx;

/// Plivo `<Stream>` serializer (ported from `flowcat_core::serializer::plivo`).
/// Behind `plivo` (on by default).
#[cfg(feature = "plivo")]
pub mod plivo;

/// Exotel serializer. Behind `exotel`.
#[cfg(feature = "exotel")]
pub mod exotel;

/// Vonage serializer. Behind `vonage`.
#[cfg(feature = "vonage")]
pub mod vonage;

/// Genesys serializer. Behind `genesys`.
#[cfg(feature = "genesys")]
pub mod genesys;

/// Asterisk (ARI/AudioSocket) serializer. Behind `asterisk`.
#[cfg(feature = "asterisk")]
pub mod asterisk;

/// Cloudonix serializer. Behind `cloudonix`.
#[cfg(feature = "cloudonix")]
pub mod cloudonix;

/// Vobiz serializer. Behind `vobiz`.
#[cfg(feature = "vobiz")]
pub mod vobiz;

/// DTMF detection/encoding (RFC2833 always; in-band Goertzel behind
/// `dtmf-inband`).
pub mod dtmf;

// ---- Convenience re-exports. ----

#[cfg(feature = "twilio")]
pub use twilio::TwilioSerializer;

#[cfg(feature = "telnyx")]
pub use telnyx::{Encoding as TelnyxEncoding, TelnyxSerializer};

#[cfg(feature = "plivo")]
pub use plivo::{plivo_answer_xml, PlivoSerializer};

#[cfg(feature = "exotel")]
pub use exotel::ExotelSerializer;

#[cfg(feature = "vonage")]
pub use vonage::VonageSerializer;

#[cfg(feature = "genesys")]
pub use genesys::GenesysSerializer;

#[cfg(feature = "asterisk")]
pub use asterisk::AsteriskSerializer;

#[cfg(feature = "cloudonix")]
pub use cloudonix::CloudonixSerializer;

#[cfg(feature = "vobiz")]
pub use vobiz::VobizSerializer;

pub use dtmf::{DtmfSymbol, Rfc2833Receiver, Rfc2833Sender, TelephoneEvent};

#[cfg(feature = "dtmf-inband")]
pub use dtmf::InbandDtmfDetector;
