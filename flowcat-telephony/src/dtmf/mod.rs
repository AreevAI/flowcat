// SPDX-License-Identifier: Apache-2.0
//
//! DTMF detection/encoding bridged onto the core [`Frame`] enum.
//!
//! Emits/consumes the DTMF frames already in the core
//! [`Frame`](flowcat_core::Frame) enum (`InputDtmf`/`OutputDtmf`/`KeypadEntry`).
//!
//! - [`rfc2833`] — RFC 2833 / RFC 4733 `telephone-event` (out-of-band) DTMF;
//!   always available.
//! - [`goertzel`] — in-band Goertzel tone detection; behind `dtmf-inband`.
//!
//! The processors here are deliberately **pure, synchronous, channel-free** state
//! machines: feed them wire data / [`Frame`]s, get [`Frame`]s back. They contain
//! no sockets and no `async`, so the live pipeline can drive them from a
//! `FrameProcessor::process_frame` arm, and they are exhaustively
//! fixture-testable here without any transport.

pub mod rfc2833;

#[cfg(feature = "dtmf-inband")]
pub mod goertzel;

use flowcat_core::{Frame, KeypadEntry};

pub use rfc2833::{DtmfSymbol, TelephoneEvent};

/// Receives RFC 4733 `telephone-event` RTP payloads and emits one
/// [`Frame::InputDtmf`] per completed key-press.
///
/// RFC 4733 sends an event as **many** packets (one per RTP frame for the tone's
/// duration), with the `E` (end) bit set on the final packet(s). This receiver
/// debounces that into a single logical press: it fires exactly once, on the
/// first packet bearing the end bit for the current event, and suppresses the
/// duplicate end retransmissions RFC 4733 mandates.
///
/// A–D events (codes 12–15) have no 12-key [`KeypadEntry`]; they are recognized
/// but do **not** produce a core `Frame::InputDtmf` (the symbol is still available
/// via [`Rfc2833Receiver::last_symbol`] for callers that want the full grid).
#[derive(Debug, Default)]
pub struct Rfc2833Receiver {
    /// The event code of the in-progress press, if any (used to dedup).
    active: Option<u8>,
    /// Whether the end packet for the active event has already fired.
    fired_end: bool,
    /// The last fully-recognized symbol (covers A–D, unlike `Frame::InputDtmf`).
    last_symbol: Option<DtmfSymbol>,
}

impl Rfc2833Receiver {
    /// Create an idle receiver.
    pub fn new() -> Self {
        Self::default()
    }

    /// The most recent fully-recognized DTMF symbol (including A–D).
    pub fn last_symbol(&self) -> Option<DtmfSymbol> {
        self.last_symbol
    }

    /// Feed one `telephone-event` payload. Returns `Some(Frame::InputDtmf(..))`
    /// exactly once per key-press (on its end packet), or `None` for start/repeat
    /// packets, malformed payloads, or A–D keys. Never panics.
    pub fn on_payload(&mut self, payload: &[u8]) -> Option<Frame> {
        let ev = rfc2833::decode_event(payload)?;
        let code = ev.symbol.event_code();

        // A new event begins (or a different key than the active one).
        if self.active != Some(code) {
            self.active = Some(code);
            self.fired_end = false;
        }

        if !ev.end {
            // Mid-tone packet: record nothing yet.
            return None;
        }

        // End packet. RFC 4733 retransmits the end packet up to 3×; fire once.
        if self.fired_end {
            return None;
        }
        self.fired_end = true;
        self.last_symbol = Some(ev.symbol);

        // Map to the 12-key core frame; A–D have no core representation.
        ev.symbol.to_keypad_entry().map(Frame::InputDtmf)
    }

    /// Reset the receiver between calls.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Encodes a [`Frame::OutputDtmf`] into the RFC 4733 `telephone-event` payload
/// **sequence** a carrier expects: for each key, a run of `repeats` tone packets
/// with rising `duration`, then an end packet (the `E` bit set) — mirroring how a
/// sender spaces a DTMF digit across RTP frames.
#[derive(Debug, Clone, Copy)]
pub struct Rfc2833Sender {
    /// Tone packets emitted before the end packet (RFC 4733 sends one per RTP
    /// frame for the key's hold time; 3 is a reasonable fixture default).
    pub tone_packets: u16,
    /// Timestamp ticks added to `duration` per packet (160 ticks = 20 ms @ 8 kHz).
    pub tick_step: u16,
    /// Tone volume in −dBm0 (0 = loudest).
    pub volume: u8,
}

impl Default for Rfc2833Sender {
    fn default() -> Self {
        Self {
            tone_packets: 3,
            tick_step: 160,
            volume: 10,
        }
    }
}

impl Rfc2833Sender {
    /// Encode the payload sequence for one keypad key.
    pub fn encode_key(&self, key: KeypadEntry) -> Vec<[u8; 4]> {
        self.encode_symbol(DtmfSymbol::from_keypad_entry(key))
    }

    /// Encode the payload sequence for any DTMF symbol (including A–D).
    pub fn encode_symbol(&self, symbol: DtmfSymbol) -> Vec<[u8; 4]> {
        let mut out = Vec::with_capacity(self.tone_packets as usize + 1);
        let mut duration: u16 = 0;
        for _ in 0..self.tone_packets {
            duration = duration.saturating_add(self.tick_step);
            out.push(rfc2833::encode_event(symbol, false, self.volume, duration));
        }
        // Final end packet.
        duration = duration.saturating_add(self.tick_step);
        out.push(rfc2833::encode_event(symbol, true, self.volume, duration));
        out
    }

    /// Encode a whole [`Frame::OutputDtmf`] into a flat payload stream. Returns an
    /// empty vec for any non-`OutputDtmf` frame.
    pub fn encode_frame(&self, frame: &Frame) -> Vec<[u8; 4]> {
        match frame {
            Frame::OutputDtmf(keys) => keys.iter().flat_map(|k| self.encode_key(*k)).collect(),
            _ => Vec::new(),
        }
    }
}

/// In-band DTMF detector: feeds blocks of mono PCM, debounces repeated detections
/// of the same tone, and emits one [`Frame::InputDtmf`] per distinct key-press.
///
/// Only available with the `dtmf-inband` feature (it pulls the Goertzel DSP).
#[cfg(feature = "dtmf-inband")]
#[derive(Debug, Default)]
pub struct InbandDtmfDetector {
    sample_rate: u32,
    cfg: goertzel::DetectorConfig,
    /// The symbol currently being held (debounce: don't re-fire while held).
    held: Option<DtmfSymbol>,
    last_symbol: Option<DtmfSymbol>,
}

#[cfg(feature = "dtmf-inband")]
impl InbandDtmfDetector {
    /// Create a detector at the given sample rate with default thresholds.
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            cfg: goertzel::DetectorConfig::default(),
            held: None,
            last_symbol: None,
        }
    }

    /// Create a detector with custom thresholds.
    pub fn with_config(sample_rate: u32, cfg: goertzel::DetectorConfig) -> Self {
        Self {
            sample_rate,
            cfg,
            held: None,
            last_symbol: None,
        }
    }

    /// The most recent detected symbol (including A–D).
    pub fn last_symbol(&self) -> Option<DtmfSymbol> {
        self.last_symbol
    }

    /// Feed one PCM block. Emits `Some(Frame::InputDtmf(..))` on the **rising
    /// edge** of a new tone (so a tone spread over several blocks fires once), and
    /// `None` while the same tone is held, on silence, or for A–D keys. Never
    /// panics.
    pub fn on_audio(&mut self, pcm: &[i16]) -> Option<Frame> {
        let detected = goertzel::detect(pcm, self.sample_rate, &self.cfg);
        match detected {
            Some(sym) => {
                if self.held == Some(sym) {
                    // Same tone still held — already reported.
                    return None;
                }
                self.held = Some(sym);
                self.last_symbol = Some(sym);
                sym.to_keypad_entry().map(Frame::InputDtmf)
            }
            None => {
                // Tone released; the next detection is a fresh press.
                self.held = None;
                None
            }
        }
    }

    /// Reset detector state between calls.
    pub fn reset(&mut self) {
        self.held = None;
        self.last_symbol = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Frame` has no `PartialEq` (it carries `Arc<dyn CustomFrame>`), so unwrap
    /// the keypad entry out of a `Frame::InputDtmf` for comparison.
    fn dtmf_key(frame: Option<Frame>) -> KeypadEntry {
        match frame {
            Some(Frame::InputDtmf(k)) => k,
            other => panic!("expected Frame::InputDtmf, got {other:?}"),
        }
    }

    #[test]
    fn receiver_fires_once_on_end_packet() {
        let mut rx = Rfc2833Receiver::new();
        // Two mid-tone packets → no frame yet.
        let p_start = rfc2833::encode_event(DtmfSymbol::Digit(5), false, 10, 160);
        assert!(rx.on_payload(&p_start).is_none());
        assert!(rx.on_payload(&p_start).is_none());
        // End packet → exactly one frame.
        let p_end = rfc2833::encode_event(DtmfSymbol::Digit(5), true, 10, 480);
        assert_eq!(dtmf_key(rx.on_payload(&p_end)), KeypadEntry::Five);
        // Retransmitted end packet → suppressed.
        assert!(rx.on_payload(&p_end).is_none());
        assert_eq!(rx.last_symbol(), Some(DtmfSymbol::Digit(5)));
    }

    #[test]
    fn receiver_handles_back_to_back_distinct_keys() {
        let mut rx = Rfc2833Receiver::new();
        let one_end = rfc2833::encode_event(DtmfSymbol::Digit(1), true, 10, 320);
        let two_end = rfc2833::encode_event(DtmfSymbol::Digit(2), true, 10, 320);
        assert_eq!(dtmf_key(rx.on_payload(&one_end)), KeypadEntry::One);
        assert_eq!(dtmf_key(rx.on_payload(&two_end)), KeypadEntry::Two);
    }

    #[test]
    fn receiver_recognizes_letters_without_core_frame() {
        let mut rx = Rfc2833Receiver::new();
        let a_end = rfc2833::encode_event(DtmfSymbol::Letter('A'), true, 10, 320);
        // No core Frame for A, but the symbol is recorded.
        assert!(rx.on_payload(&a_end).is_none());
        assert_eq!(rx.last_symbol(), Some(DtmfSymbol::Letter('A')));
    }

    #[test]
    fn receiver_ignores_malformed_payloads() {
        let mut rx = Rfc2833Receiver::new();
        assert!(rx.on_payload(&[]).is_none());
        assert!(rx.on_payload(&[1, 2, 3]).is_none());
        assert!(rx.on_payload(&[99, 0, 0, 0]).is_none());
    }

    #[test]
    fn sender_round_trips_through_receiver_for_all_12() {
        let sender = Rfc2833Sender::default();
        use KeypadEntry::*;
        for key in [
            Zero, One, Two, Three, Four, Five, Six, Seven, Eight, Nine, Star, Pound,
        ] {
            let payloads = sender.encode_key(key);
            assert_eq!(payloads.len(), sender.tone_packets as usize + 1);
            // Feed the whole sequence to a fresh receiver: exactly one frame.
            let mut rx = Rfc2833Receiver::new();
            let mut frames: Vec<Frame> = payloads.iter().filter_map(|p| rx.on_payload(p)).collect();
            assert_eq!(frames.len(), 1, "key {key:?} should yield one frame");
            assert_eq!(dtmf_key(frames.pop()), key);
        }
    }

    #[test]
    fn sender_encodes_output_dtmf_frame() {
        let sender = Rfc2833Sender::default();
        let frame = Frame::OutputDtmf(vec![KeypadEntry::One, KeypadEntry::Pound]);
        let payloads = sender.encode_frame(&frame);
        // Two keys × (tone_packets + 1) end packet each.
        assert_eq!(payloads.len(), 2 * (sender.tone_packets as usize + 1));
        // A non-OutputDtmf frame yields nothing.
        assert!(sender.encode_frame(&Frame::Stop).is_empty());
    }

    #[test]
    fn sender_encodes_letters() {
        let sender = Rfc2833Sender::default();
        let seq = sender.encode_symbol(DtmfSymbol::Letter('D'));
        // The last packet must carry the end bit + event code 15.
        let last = seq.last().unwrap();
        assert_eq!(last[0], 15);
        assert_eq!(last[1] & 0x80, 0x80);
    }

    #[cfg(feature = "dtmf-inband")]
    #[test]
    fn inband_detector_fires_once_per_press() {
        let sr = 8000u32;
        let mut det = InbandDtmfDetector::new(sr);
        // Build a 50 ms dual-tone for '7' (852 Hz + 1209 Hz).
        let n = (sr * 50 / 1000) as usize;
        let block: Vec<i16> = (0..n)
            .map(|i| {
                let t = i as f64 / sr as f64;
                let a = (2.0 * std::f64::consts::PI * 852.0 * t).sin();
                let b = (2.0 * std::f64::consts::PI * 1209.0 * t).sin();
                ((a + b) * 0.4 * i16::MAX as f64) as i16
            })
            .collect();
        // Rising edge fires.
        assert_eq!(dtmf_key(det.on_audio(&block)), KeypadEntry::Seven);
        // Held tone does not re-fire.
        assert!(det.on_audio(&block).is_none());
        // Silence releases.
        let silence = vec![0i16; n];
        assert!(det.on_audio(&silence).is_none());
        // The same tone again is a new press.
        assert_eq!(dtmf_key(det.on_audio(&block)), KeypadEntry::Seven);
        assert_eq!(det.last_symbol(), Some(DtmfSymbol::Digit(7)));
    }

    #[cfg(feature = "dtmf-inband")]
    #[test]
    fn inband_detector_ignores_silence_and_noise() {
        let mut det = InbandDtmfDetector::new(8000);
        assert!(det.on_audio(&vec![0i16; 400]).is_none());
        assert!(det.on_audio(&[]).is_none());
    }
}
