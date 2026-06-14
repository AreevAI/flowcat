// SPDX-License-Identifier: Apache-2.0
//
//! Transport layer.
//!
//! Two seams live here:
//! - [`MediaSocket`] — a raw bidirectional WS text/binary frame pipe, with a
//!   tungstenite-backed [`WsMediaTransport`] implementation.
//! - [`MediaTransport`] — the transport-agnostic *decoded-audio* seam the
//!   pipeline drives (WS frames or RTP look the same to it). The WS path is
//!   adapted into it by [`WsCarrierTransport`].

pub mod carrier;
pub mod media;
pub mod socket;
pub mod ws_media;

pub use carrier::WsCarrierTransport;
pub use media::{MediaIn, MediaTransport};
pub use socket::MediaSocket;
pub use ws_media::WsMediaTransport;
