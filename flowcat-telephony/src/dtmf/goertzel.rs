// SPDX-License-Identifier: Apache-2.0
//
//! In-band DTMF detection via the **Goertzel algorithm** (behind `dtmf-inband`).
//!
//! DTMF encodes each key as the sum of one **low-group** and one **high-group**
//! sine tone:
//!
//! ```text
//!          1209  1336  1477  1633 Hz
//!   697 Hz    1     2     3     A
//!   770 Hz    4     5     6     B
//!   852 Hz    7     8     9     C
//!   941 Hz    *     0     #     D
//! ```
//!
//! The Goertzel algorithm evaluates the DFT energy at a *single* target frequency
//! far more cheaply than a full FFT. We run eight detectors (one per DTMF
//! frequency) over a block of mono PCM and pick the strongest low/high pair,
//! validating it against an absolute energy floor and a "twist" ratio so noise and
//! speech do not register as digits.
//!
//! Pure DSP — `f64` math only, no extra dep. Detection never panics on empty or
//! garbage input (it returns `None`).

use crate::dtmf::rfc2833::DtmfSymbol;

/// The four DTMF low-group frequencies (Hz).
const LOW_FREQS: [f64; 4] = [697.0, 770.0, 852.0, 941.0];
/// The four DTMF high-group frequencies (Hz).
const HIGH_FREQS: [f64; 4] = [1209.0, 1336.0, 1477.0, 1633.0];

/// The DTMF grid indexed `[low_row][high_col]`.
const GRID: [[char; 4]; 4] = [
    ['1', '2', '3', 'A'],
    ['4', '5', '6', 'B'],
    ['7', '8', '9', 'C'],
    ['*', '0', '#', 'D'],
];

/// Tunable detection thresholds.
#[derive(Debug, Clone, Copy)]
pub struct DetectorConfig {
    /// Minimum normalized energy (per sample) for a tone to count as present.
    /// Energy is `goertzel_power / (n^2 * 0.25)` so a full-scale pure tone ≈ 1.0.
    pub energy_floor: f64,
    /// Maximum allowed ratio between the dominant and the next-strongest tone
    /// **within the same group** — guards against broadband (speech) energy
    /// where several bins are comparably hot.
    pub group_dominance: f64,
    /// Maximum allowed "twist": |10·log10(low/high)| dB. Real DTMF keeps the two
    /// tones within roughly ±8 dB.
    pub max_twist_db: f64,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            energy_floor: 0.0035,
            group_dominance: 0.5,
            max_twist_db: 8.0,
        }
    }
}

/// One Goertzel single-frequency power evaluation over `samples` at `sample_rate`.
///
/// Returns the squared magnitude of the DFT bin nearest `target_freq`.
fn goertzel_power(samples: &[i16], sample_rate: u32, target_freq: f64) -> f64 {
    let n = samples.len();
    if n == 0 || sample_rate == 0 {
        return 0.0;
    }
    let k = (0.5 + (n as f64 * target_freq) / sample_rate as f64).floor();
    let omega = (2.0 * std::f64::consts::PI * k) / n as f64;
    let coeff = 2.0 * omega.cos();
    let mut s_prev = 0.0f64;
    let mut s_prev2 = 0.0f64;
    for &x in samples {
        let s = x as f64 + coeff * s_prev - s_prev2;
        s_prev2 = s_prev;
        s_prev = s;
    }
    // |X(k)|^2 = s_prev^2 + s_prev2^2 - coeff * s_prev * s_prev2
    s_prev * s_prev + s_prev2 * s_prev2 - coeff * s_prev * s_prev2
}

/// Detect a single DTMF symbol in a block of mono 16-bit PCM.
///
/// Returns the detected [`DtmfSymbol`] (covering all 16 keys, including A–D) or
/// `None` if the block is silence, noise, speech, or too short to evaluate the
/// low tones reliably.
pub fn detect(samples: &[i16], sample_rate: u32, cfg: &DetectorConfig) -> Option<DtmfSymbol> {
    let n = samples.len();
    // Need enough samples to resolve ~697 Hz: at least ~1.5 cycles. At 8 kHz a
    // standard ≥40 ms (≥320-sample) block is comfortable; bail on tiny blocks.
    if n < 64 || sample_rate == 0 {
        return None;
    }

    // Normalizer so a full-scale pure tone yields ≈ 1.0 regardless of block size.
    let norm = (n as f64).powi(2) * 0.25 * (i16::MAX as f64).powi(2);
    if norm == 0.0 {
        return None;
    }

    let low: Vec<f64> = LOW_FREQS
        .iter()
        .map(|&f| goertzel_power(samples, sample_rate, f) / norm)
        .collect();
    let high: Vec<f64> = HIGH_FREQS
        .iter()
        .map(|&f| goertzel_power(samples, sample_rate, f) / norm)
        .collect();

    let (low_idx, low_e) = argmax(&low)?;
    let (high_idx, high_e) = argmax(&high)?;

    // Both groups must clear the absolute energy floor.
    if low_e < cfg.energy_floor || high_e < cfg.energy_floor {
        return None;
    }

    // The winning bin must dominate its group (rejects broadband speech).
    if second_max(&low, low_idx) > low_e * cfg.group_dominance
        || second_max(&high, high_idx) > high_e * cfg.group_dominance
    {
        return None;
    }

    // Twist: the two tones must be within ±max_twist_db of each other.
    let twist_db = 10.0 * (low_e / high_e).log10();
    if twist_db.abs() > cfg.max_twist_db {
        return None;
    }

    DtmfSymbol::from_char(GRID[low_idx][high_idx])
}

/// Index + value of the maximum element.
fn argmax(v: &[f64]) -> Option<(usize, f64)> {
    v.iter()
        .copied()
        .enumerate()
        .filter(|(_, x)| x.is_finite())
        .fold(None, |acc, (i, x)| match acc {
            Some((_, best)) if x <= best => acc,
            _ => Some((i, x)),
        })
}

/// The largest value in `v` excluding index `skip`.
fn second_max(v: &[f64], skip: usize) -> f64 {
    v.iter()
        .copied()
        .enumerate()
        .filter(|&(i, x)| i != skip && x.is_finite())
        .map(|(_, x)| x)
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 8000;

    /// Synthesize a DTMF dual-tone block of `ms` milliseconds at full-ish scale.
    fn dual_tone(low: f64, high: f64, ms: u32) -> Vec<i16> {
        let n = (SR * ms / 1000) as usize;
        (0..n)
            .map(|i| {
                let t = i as f64 / SR as f64;
                let a = (2.0 * std::f64::consts::PI * low * t).sin();
                let b = (2.0 * std::f64::consts::PI * high * t).sin();
                // 0.4 amplitude each so the sum stays well inside i16 range.
                ((a + b) * 0.4 * i16::MAX as f64) as i16
            })
            .collect()
    }

    fn freqs_for(c: char) -> (f64, f64) {
        for (r, row) in GRID.iter().enumerate() {
            for (col, &k) in row.iter().enumerate() {
                if k == c {
                    return (LOW_FREQS[r], HIGH_FREQS[col]);
                }
            }
        }
        panic!("not a DTMF key: {c}");
    }

    #[test]
    fn detects_every_one_of_the_16_digits() {
        let cfg = DetectorConfig::default();
        for c in "0123456789*#ABCD".chars() {
            let (lo, hi) = freqs_for(c);
            let block = dual_tone(lo, hi, 50);
            let got = detect(&block, SR, &cfg);
            assert_eq!(
                got.map(DtmfSymbol::to_char),
                Some(c),
                "digit {c} not detected (got {got:?})"
            );
        }
    }

    #[test]
    fn rejects_silence() {
        let cfg = DetectorConfig::default();
        let silence = vec![0i16; 400];
        assert_eq!(detect(&silence, SR, &cfg), None);
    }

    #[test]
    fn rejects_white_noise() {
        let cfg = DetectorConfig::default();
        // Deterministic pseudo-random noise (LCG) so the test is stable.
        let mut state: u32 = 0x1234_5678;
        let noise: Vec<i16> = (0..400)
            .map(|_| {
                state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((state >> 16) as i16) / 4
            })
            .collect();
        assert_eq!(detect(&noise, SR, &cfg), None);
    }

    #[test]
    fn rejects_single_tone() {
        let cfg = DetectorConfig::default();
        // A lone 697 Hz tone (no high-group partner) is not a valid DTMF key.
        let n = 400usize;
        let single: Vec<i16> = (0..n)
            .map(|i| {
                let t = i as f64 / SR as f64;
                ((2.0 * std::f64::consts::PI * 697.0 * t).sin() * 0.8 * i16::MAX as f64) as i16
            })
            .collect();
        assert_eq!(detect(&single, SR, &cfg), None);
    }

    #[test]
    fn rejects_too_short_block() {
        let cfg = DetectorConfig::default();
        let (lo, hi) = freqs_for('1');
        let tiny = dual_tone(lo, hi, 5); // 40 samples
        assert!(tiny.len() < 64);
        assert_eq!(detect(&tiny, SR, &cfg), None);
    }

    #[test]
    fn empty_and_zero_rate_do_not_panic() {
        let cfg = DetectorConfig::default();
        assert_eq!(detect(&[], SR, &cfg), None);
        let (lo, hi) = freqs_for('5');
        let block = dual_tone(lo, hi, 50);
        assert_eq!(detect(&block, 0, &cfg), None);
        assert_eq!(goertzel_power(&[], SR, 697.0), 0.0);
    }
}
