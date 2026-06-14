// SPDX-License-Identifier: Apache-2.0
//
//! RFC 2833 / RFC 4733 `telephone-event` DTMF payload codec.
//!
//! Out-of-band DTMF rides its own RTP payload type as a 4-byte `telephone-event`
//! payload (RFC 4733 §2.3):
//!
//! ```text
//!  0                   1                   2                   3
//!  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! |     event     |E|R| volume    |          duration             |
//! +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//! ```
//!
//! - `event` (8 bits): the DTMF code, 0–15 (0–9, `*`=10, `#`=11, A–D=12–15).
//! - `E` (end): set on the final packet of an event.
//! - `R`: reserved (must be 0).
//! - `volume` (6 bits): tone power, 0 (loudest) … 63, expressed as −dBm0.
//! - `duration` (16 bits, big-endian): event length so far, in timestamp ticks
//!   (8 kHz for telephony → 8 ticks/ms).
//!
//! This module is **always available** (no feature gate, no extra dep): it only
//! builds/parses the 4-byte payload — it does not touch RTP headers or sockets.
//! Decoders are defensive: a short/garbage payload yields `None`, never a panic.

use flowcat_core::KeypadEntry;

/// A DTMF keypad symbol covering the **full 16-symbol** telephone-event grid,
/// including the A–D tones the 12-key core [`KeypadEntry`] omits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DtmfSymbol {
    /// `0`–`9`.
    Digit(u8),
    /// `*`.
    Star,
    /// `#`.
    Pound,
    /// `A`–`D` (event codes 12–15); not representable in core `KeypadEntry`.
    Letter(char),
}

impl DtmfSymbol {
    /// The RFC 4733 event code (0–15) for this symbol.
    pub fn event_code(self) -> u8 {
        match self {
            DtmfSymbol::Digit(d) => d, // 0..=9
            DtmfSymbol::Star => 10,
            DtmfSymbol::Pound => 11,
            DtmfSymbol::Letter('A') => 12,
            DtmfSymbol::Letter('B') => 13,
            DtmfSymbol::Letter('C') => 14,
            DtmfSymbol::Letter('D') => 15,
            // Any out-of-range Letter is clamped to 'A' defensively; constructors
            // below never produce one.
            DtmfSymbol::Letter(_) => 12,
        }
    }

    /// Build a symbol from an RFC 4733 event code (0–15). Returns `None` for any
    /// code outside the DTMF grid.
    pub fn from_event_code(code: u8) -> Option<Self> {
        match code {
            0..=9 => Some(DtmfSymbol::Digit(code)),
            10 => Some(DtmfSymbol::Star),
            11 => Some(DtmfSymbol::Pound),
            12 => Some(DtmfSymbol::Letter('A')),
            13 => Some(DtmfSymbol::Letter('B')),
            14 => Some(DtmfSymbol::Letter('C')),
            15 => Some(DtmfSymbol::Letter('D')),
            _ => None,
        }
    }

    /// The keypad character for this symbol (`'0'..'9'`, `'*'`, `'#'`, `'A'..'D'`).
    pub fn to_char(self) -> char {
        match self {
            DtmfSymbol::Digit(d) => (b'0' + d) as char,
            DtmfSymbol::Star => '*',
            DtmfSymbol::Pound => '#',
            DtmfSymbol::Letter(c) => c,
        }
    }

    /// Parse a symbol from a keypad character. Returns `None` for anything that is
    /// not a DTMF key.
    pub fn from_char(c: char) -> Option<Self> {
        match c {
            '0'..='9' => Some(DtmfSymbol::Digit(c as u8 - b'0')),
            '*' => Some(DtmfSymbol::Star),
            '#' => Some(DtmfSymbol::Pound),
            'A'..='D' => Some(DtmfSymbol::Letter(c)),
            'a'..='d' => Some(DtmfSymbol::Letter(c.to_ascii_uppercase())),
            _ => None,
        }
    }

    /// Convert to the 12-key core [`KeypadEntry`]. A–D have no core
    /// representation and yield `None`.
    pub fn to_keypad_entry(self) -> Option<KeypadEntry> {
        Some(match self {
            DtmfSymbol::Digit(0) => KeypadEntry::Zero,
            DtmfSymbol::Digit(1) => KeypadEntry::One,
            DtmfSymbol::Digit(2) => KeypadEntry::Two,
            DtmfSymbol::Digit(3) => KeypadEntry::Three,
            DtmfSymbol::Digit(4) => KeypadEntry::Four,
            DtmfSymbol::Digit(5) => KeypadEntry::Five,
            DtmfSymbol::Digit(6) => KeypadEntry::Six,
            DtmfSymbol::Digit(7) => KeypadEntry::Seven,
            DtmfSymbol::Digit(8) => KeypadEntry::Eight,
            DtmfSymbol::Digit(9) => KeypadEntry::Nine,
            DtmfSymbol::Star => KeypadEntry::Star,
            DtmfSymbol::Pound => KeypadEntry::Pound,
            DtmfSymbol::Digit(_) | DtmfSymbol::Letter(_) => return None,
        })
    }

    /// Build a symbol from the 12-key core [`KeypadEntry`].
    pub fn from_keypad_entry(k: KeypadEntry) -> Self {
        match k {
            KeypadEntry::Zero => DtmfSymbol::Digit(0),
            KeypadEntry::One => DtmfSymbol::Digit(1),
            KeypadEntry::Two => DtmfSymbol::Digit(2),
            KeypadEntry::Three => DtmfSymbol::Digit(3),
            KeypadEntry::Four => DtmfSymbol::Digit(4),
            KeypadEntry::Five => DtmfSymbol::Digit(5),
            KeypadEntry::Six => DtmfSymbol::Digit(6),
            KeypadEntry::Seven => DtmfSymbol::Digit(7),
            KeypadEntry::Eight => DtmfSymbol::Digit(8),
            KeypadEntry::Nine => DtmfSymbol::Digit(9),
            KeypadEntry::Star => DtmfSymbol::Star,
            KeypadEntry::Pound => DtmfSymbol::Pound,
        }
    }
}

/// A decoded RFC 4733 `telephone-event` payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TelephoneEvent {
    /// The DTMF symbol.
    pub symbol: DtmfSymbol,
    /// The `E` (end) bit — set on the final packet of the event.
    pub end: bool,
    /// Tone volume in −dBm0 (0 = loudest … 63).
    pub volume: u8,
    /// Cumulative event duration in RTP timestamp ticks.
    pub duration: u16,
}

/// Encode a `telephone-event` 4-byte payload.
///
/// `volume` is masked to 6 bits; `duration` is written big-endian. The reserved
/// `R` bit is always 0 (RFC 4733 §2.5.1).
pub fn encode_event(symbol: DtmfSymbol, end: bool, volume: u8, duration: u16) -> [u8; 4] {
    let event = symbol.event_code();
    let mut byte1 = volume & 0x3F; // low 6 bits = volume
    if end {
        byte1 |= 0x80; // E bit
    }
    let dur = duration.to_be_bytes();
    [event, byte1, dur[0], dur[1]]
}

/// Decode a `telephone-event` payload. Returns `None` if the payload is not 4
/// bytes or the event code is outside the DTMF grid (0–15) — never panics.
pub fn decode_event(payload: &[u8]) -> Option<TelephoneEvent> {
    if payload.len() != 4 {
        return None;
    }
    let symbol = DtmfSymbol::from_event_code(payload[0])?;
    let end = payload[1] & 0x80 != 0;
    let volume = payload[1] & 0x3F;
    let duration = u16::from_be_bytes([payload[2], payload[3]]);
    Some(TelephoneEvent {
        symbol,
        end,
        volume,
        duration,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 16 DTMF symbols in event-code order.
    fn all_symbols() -> Vec<DtmfSymbol> {
        let mut v: Vec<DtmfSymbol> = (0u8..=9).map(DtmfSymbol::Digit).collect();
        v.push(DtmfSymbol::Star);
        v.push(DtmfSymbol::Pound);
        v.extend(['A', 'B', 'C', 'D'].map(DtmfSymbol::Letter));
        v
    }

    #[test]
    fn all_16_symbols_encode_decode_round_trip() {
        for (i, sym) in all_symbols().into_iter().enumerate() {
            assert_eq!(sym.event_code() as usize, i, "event code ordering");
            let payload = encode_event(sym, true, 10, 1600);
            let ev = decode_event(&payload).expect("valid payload decodes");
            assert_eq!(ev.symbol, sym);
            assert!(ev.end);
            assert_eq!(ev.volume, 10);
            assert_eq!(ev.duration, 1600);
        }
    }

    #[test]
    fn char_round_trip_for_all_16() {
        for c in "0123456789*#ABCD".chars() {
            let sym = DtmfSymbol::from_char(c).unwrap();
            assert_eq!(sym.to_char(), c);
        }
        // Lowercase letters normalize to uppercase.
        assert_eq!(DtmfSymbol::from_char('a'), Some(DtmfSymbol::Letter('A')));
        // Non-DTMF chars reject.
        assert_eq!(DtmfSymbol::from_char('E'), None);
        assert_eq!(DtmfSymbol::from_char('z'), None);
    }

    #[test]
    fn end_bit_and_duration_encode_correctly() {
        // Not-end, volume 0, duration 0x0140 = 320.
        let p = encode_event(DtmfSymbol::Digit(5), false, 0, 0x0140);
        assert_eq!(p, [5, 0x00, 0x01, 0x40]);
        let ev = decode_event(&p).unwrap();
        assert!(!ev.end);
        assert_eq!(ev.duration, 320);

        // End bit set with volume 63.
        let p2 = encode_event(DtmfSymbol::Pound, true, 63, 800);
        assert_eq!(p2[0], 11);
        assert_eq!(p2[1], 0x80 | 0x3F);
        assert_eq!(decode_event(&p2).unwrap().symbol, DtmfSymbol::Pound);
    }

    #[test]
    fn volume_is_masked_to_six_bits() {
        // volume 0xFF must not leak into the E/R bits.
        let p = encode_event(DtmfSymbol::Digit(1), false, 0xFF, 0);
        assert_eq!(p[1] & 0x80, 0, "E bit must stay clear");
        assert_eq!(p[1] & 0x3F, 0x3F);
        assert_eq!(decode_event(&p).unwrap().volume, 0x3F);
    }

    #[test]
    fn decode_rejects_malformed_payloads() {
        assert_eq!(decode_event(&[]), None);
        assert_eq!(decode_event(&[1, 2, 3]), None); // too short
        assert_eq!(decode_event(&[1, 2, 3, 4, 5]), None); // too long
        assert_eq!(decode_event(&[16, 0, 0, 0]), None); // event code 16 = out of grid
        assert_eq!(decode_event(&[255, 0, 0, 0]), None);
    }

    #[test]
    fn keypad_entry_round_trip_for_12() {
        use flowcat_core::KeypadEntry::*;
        for k in [
            Zero, One, Two, Three, Four, Five, Six, Seven, Eight, Nine, Star, Pound,
        ] {
            let sym = DtmfSymbol::from_keypad_entry(k);
            assert_eq!(sym.to_keypad_entry(), Some(k));
        }
        // A–D have no core KeypadEntry.
        for c in ['A', 'B', 'C', 'D'] {
            assert_eq!(DtmfSymbol::Letter(c).to_keypad_entry(), None);
        }
    }
}
