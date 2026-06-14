// SPDX-License-Identifier: Apache-2.0
//
//! Opus encode/decode for the WebRTC transport.
//!
//! Bridges WebRTC Opus payloads ↔ the pipeline's PCM
//! [`AudioChunk`](flowcat_core::types::AudioChunk)s via `audiopus` (libopus).
//! Behind `webrtc-str0m`.
//!
//! ## Rates
//!
//! Browser WebRTC negotiates Opus at a **48 kHz** clock (the RTP timebase is
//! always 48 kHz for Opus regardless of the internal coded bandwidth). The
//! Flowcat pipeline works in mono PCM tagged with its own sample rate and
//! resamples to/from the carrier rate via [`flowcat_core::codec::Resampler`].
//!
//! This module is deliberately the *codec only* — it does **no** resampling.
//! It encodes/decodes at a single Opus rate (48 kHz mono, the WebRTC default)
//! and the [`super`] transport composes a [`Resampler`](flowcat_core::codec::Resampler)
//! on each side to bridge 48 kHz ↔ the pipeline's `carrier_rate`. Keeping the
//! two concerns separate keeps each unit-testable.
//!
//! ## Frame sizes
//!
//! Opus only accepts a frame whose sample count is one of the legal Opus frame
//! durations (2.5/5/10/20/40/60 ms). At 48 kHz mono those are 120/240/480/960/
//! 1920/2880 samples. WebRTC browsers emit **20 ms** frames (960 samples @
//! 48 kHz mono) by default; [`OpusEncoder::encode`] therefore expects exactly a
//! legal frame and returns an error (never panics) on an illegal length.

use audiopus::coder::{Decoder, Encoder};
use audiopus::{Application, Channels, SampleRate};

use flowcat_core::error::FlowcatError;

/// The Opus clock / PCM rate WebRTC uses (48 kHz mono).
pub const OPUS_RATE: u32 = 48_000;

/// Samples in a 20 ms Opus frame at 48 kHz mono — the browser default and the
/// frame size [`OpusEncoder::encode`] is sized for.
pub const FRAME_20MS_48K: usize = 960;

/// The largest legal Opus frame at 48 kHz mono (60 ms = 2880 samples). Decode
/// output buffers are sized to this so any legal browser frame fits.
pub const MAX_FRAME_48K: usize = 2880;

/// Max bytes in one compressed Opus packet we will accept / emit. Opus packets
/// are small (a 20 ms VoIP frame is typically < 200 bytes); 4000 is libopus's
/// own recommended max-packet bound and bounds the encode output buffer.
pub const MAX_PACKET_BYTES: usize = 4000;

/// An Opus encoder: 48 kHz mono PCM → compressed Opus payload (the RTP frame
/// the WebRTC peer writes).
pub struct OpusEncoder {
    inner: Encoder,
}

impl OpusEncoder {
    /// Build a VoIP-tuned mono 48 kHz Opus encoder.
    pub fn new() -> Result<Self, FlowcatError> {
        let inner = Encoder::new(SampleRate::Hz48000, Channels::Mono, Application::Voip)
            .map_err(|e| FlowcatError::Codec(format!("opus encoder init: {e}")))?;
        Ok(Self { inner })
    }

    /// Encode one mono 48 kHz PCM frame to an Opus packet.
    ///
    /// `pcm.len()` must be a legal Opus frame length at 48 kHz mono
    /// (120/240/480/960/1920/2880 samples). Returns the compressed bytes.
    /// Never panics: an illegal length or a libopus error surfaces as
    /// [`FlowcatError::Codec`].
    pub fn encode(&mut self, pcm: &[i16]) -> Result<Vec<u8>, FlowcatError> {
        if !is_legal_opus_frame(pcm.len()) {
            return Err(FlowcatError::Codec(format!(
                "opus encode: {} samples is not a legal 48 kHz mono Opus frame",
                pcm.len()
            )));
        }
        let mut out = vec![0u8; MAX_PACKET_BYTES];
        let n = self
            .inner
            .encode(pcm, &mut out)
            .map_err(|e| FlowcatError::Codec(format!("opus encode: {e}")))?;
        out.truncate(n);
        Ok(out)
    }
}

/// An Opus decoder: a compressed Opus payload (an inbound WebRTC RTP frame) →
/// 48 kHz mono PCM.
pub struct OpusDecoder {
    inner: Decoder,
}

impl OpusDecoder {
    /// Build a mono 48 kHz Opus decoder.
    pub fn new() -> Result<Self, FlowcatError> {
        let inner = Decoder::new(SampleRate::Hz48000, Channels::Mono)
            .map_err(|e| FlowcatError::Codec(format!("opus decoder init: {e}")))?;
        Ok(Self { inner })
    }

    /// Decode one Opus packet to mono 48 kHz PCM.
    ///
    /// Robust to hostile input: an empty packet, a truncated/garbage payload, or
    /// any libopus error is returned as [`FlowcatError::Codec`] — it never
    /// panics. The output buffer is sized to the largest legal frame so a valid
    /// 60 ms frame always fits.
    pub fn decode(&mut self, packet: &[u8]) -> Result<Vec<i16>, FlowcatError> {
        if packet.is_empty() {
            return Err(FlowcatError::Codec("opus decode: empty packet".into()));
        }
        let mut out = vec![0i16; MAX_FRAME_48K];
        let n = self
            .inner
            .decode(Some(packet), &mut out[..], false)
            .map_err(|e| FlowcatError::Codec(format!("opus decode: {e}")))?;
        out.truncate(n);
        Ok(out)
    }

    /// Decode a *lost* frame (packet loss concealment) — feed `None`, producing
    /// `frame_samples` of concealed PCM. Used when the jitter buffer reports a
    /// gap. `frame_samples` must be a legal 48 kHz frame length.
    pub fn decode_lost(&mut self, frame_samples: usize) -> Result<Vec<i16>, FlowcatError> {
        if !is_legal_opus_frame(frame_samples) {
            return Err(FlowcatError::Codec(format!(
                "opus PLC: {frame_samples} is not a legal 48 kHz mono Opus frame"
            )));
        }
        let mut out = vec![0i16; frame_samples];
        let n = self
            .inner
            .decode(None::<&[u8]>, &mut out[..], false)
            .map_err(|e| FlowcatError::Codec(format!("opus PLC: {e}")))?;
        out.truncate(n);
        Ok(out)
    }
}

/// Whether `n` mono samples is a legal Opus frame at 48 kHz
/// (2.5/5/10/20/40/60 ms).
fn is_legal_opus_frame(n: usize) -> bool {
    matches!(n, 120 | 240 | 480 | 960 | 1920 | 2880)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 440 Hz sine at 48 kHz, `n` samples.
    fn sine(n: usize) -> Vec<i16> {
        (0..n)
            .map(|i| {
                let t = i as f32 / OPUS_RATE as f32;
                (8000.0 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()) as i16
            })
            .collect()
    }

    #[test]
    fn encode_decode_loopback_preserves_frame_size_and_rough_fidelity() {
        let mut enc = OpusEncoder::new().unwrap();
        let mut dec = OpusDecoder::new().unwrap();

        // Encode a sequence of 20 ms frames; Opus is stateful so we prime a few
        // frames before measuring fidelity (the first frame carries codec
        // lookahead / is unreliable to compare).
        let mut last_in = Vec::new();
        let mut last_out = Vec::new();
        for _ in 0..5 {
            let pcm = sine(FRAME_20MS_48K);
            let packet = enc.encode(&pcm).unwrap();
            assert!(!packet.is_empty(), "opus produced a non-empty packet");
            assert!(packet.len() <= MAX_PACKET_BYTES);
            let decoded = dec.decode(&packet).unwrap();
            assert_eq!(
                decoded.len(),
                FRAME_20MS_48K,
                "decoded frame is the same 960-sample (20 ms) size"
            );
            last_in = pcm;
            last_out = decoded;
        }

        // Round-trip fidelity (after warm-up): Opus is lossy + has algorithmic
        // delay, so we don't compare sample-for-sample. Instead we assert the
        // decoded signal has comparable energy to the input (RMS within a
        // generous tolerance) — proving it's the same tone, not silence/garbage.
        let rms = |s: &[i16]| -> f64 {
            let sumsq: f64 = s.iter().map(|&v| (v as f64) * (v as f64)).sum();
            (sumsq / s.len().max(1) as f64).sqrt()
        };
        let rin = rms(&last_in);
        let rout = rms(&last_out);
        assert!(rin > 100.0, "input has real energy");
        assert!(
            rout > rin * 0.3 && rout < rin * 3.0,
            "decoded RMS {rout:.0} within 0.3x..3x of input RMS {rin:.0} (same tone survived)"
        );
    }

    #[test]
    fn encode_rejects_illegal_frame_length_without_panicking() {
        let mut enc = OpusEncoder::new().unwrap();
        // 500 samples is not a legal 48 kHz Opus frame.
        let err = enc.encode(&vec![0i16; 500]).unwrap_err();
        assert!(matches!(err, FlowcatError::Codec(_)));
    }

    #[test]
    fn decode_handles_hostile_input_without_panicking() {
        let mut dec = OpusDecoder::new().unwrap();
        // Empty packet → error, not panic.
        assert!(dec.decode(&[]).is_err());
        // Random garbage bytes → must not panic (libopus rejects or yields
        // something; either way no panic).
        let garbage = vec![0xFFu8; 7];
        let _ = dec.decode(&garbage);
        // A clearly invalid TOC-only single byte.
        let _ = dec.decode(&[0x00u8]);
    }

    #[test]
    fn decode_lost_produces_concealment_of_requested_size() {
        let mut enc = OpusEncoder::new().unwrap();
        let mut dec = OpusDecoder::new().unwrap();
        // Prime the decoder with one real frame so PLC has state to extrapolate.
        let pkt = enc.encode(&sine(FRAME_20MS_48K)).unwrap();
        let _ = dec.decode(&pkt).unwrap();
        let concealed = dec.decode_lost(FRAME_20MS_48K).unwrap();
        assert_eq!(concealed.len(), FRAME_20MS_48K);
        // Illegal PLC length is rejected, not panicked.
        assert!(dec.decode_lost(123).is_err());
    }

    #[test]
    fn legal_frame_lengths_are_exactly_the_opus_set() {
        for n in [120usize, 240, 480, 960, 1920, 2880] {
            assert!(is_legal_opus_frame(n), "{n} should be legal");
        }
        for n in [0usize, 1, 159, 160, 320, 500, 961, 2881] {
            assert!(!is_legal_opus_frame(n), "{n} should be illegal");
        }
    }
}
