// SPDX-License-Identifier: Apache-2.0
//
//! Audio codec helpers: G.711 (μ-law / A-law) ↔ PCM16 and sample-rate conversion.
//!
//! Telephony is G.711 μ-law @ 8 kHz; Gemini Live wants 16 kHz PCM in and emits
//! 24 kHz PCM out (see DESIGN.md "Audio path"). G.711 via `audio-codec-algorithms`,
//! resampling via `rubato`.
//!
//! ## Streaming resampler design
//!
//! Telephony delivers audio in small, frequently *variable*-size chunks (≈20 ms:
//! 160 samples @ 8 kHz, 320 @ 16 kHz, 480 @ 24 kHz — and gateways may coalesce or
//! split frames). `rubato`'s sinc resamplers come in two flavours:
//!
//! - [`rubato::SincFixedIn`] consumes a **fixed** number of input frames per
//!   `process` call and produces a variable number of output frames.
//! - [`rubato::SincFixedOut`] produces a fixed number of output frames and pulls
//!   a variable number of input frames.
//!
//! A real-time media path naturally has a *fixed input cadence* (whatever the
//! carrier hands us) and an unconstrained output, so [`SincFixedIn`] is the right
//! base. But `SincFixedIn` still requires *exactly* `chunk_size` samples on every
//! call, which an arbitrary 20 ms frame will not satisfy. We therefore decouple
//! the carrier's chunk size from the resampler's block size with a small internal
//! **carry buffer**:
//!
//! 1. Append the incoming samples to `in_buf`.
//! 2. While `in_buf` holds at least one full `BLOCK` of samples, feed exactly
//!    `BLOCK` frames through the resampler (re-using pre-allocated scratch I/O
//!    buffers) and append the result to the output.
//! 3. Keep the `< BLOCK` remainder in `in_buf` for the next call.
//!
//! This yields **continuous, gap-free** resampling across an arbitrary stream of
//! variable chunks: every input sample is consumed exactly once and in order, the
//! resampler's internal filter state is preserved across calls (so block
//! boundaries are seamless), and the resampler plus its scratch buffers are
//! allocated **once** in [`Resampler::new`] — there is no per-call reallocation.
//! A pass-through (`from_rate == to_rate`) skips `rubato` entirely.

use rubato::{
    Resampler as RubatoResampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};

use crate::error::FlowcatError;
use crate::types::AudioChunk;

// ---------------------------------------------------------------------------
// G.711 (μ-law / A-law) <-> PCM16
// ---------------------------------------------------------------------------

/// Decode a G.711 μ-law byte stream into 16-bit PCM samples.
pub fn ulaw_to_pcm16(ulaw: &[u8]) -> Vec<i16> {
    ulaw.iter()
        .map(|&b| audio_codec_algorithms::decode_ulaw(b))
        .collect()
}

/// Encode 16-bit PCM samples into a G.711 μ-law byte stream.
pub fn pcm16_to_ulaw(pcm: &[i16]) -> Vec<u8> {
    pcm.iter()
        .map(|&s| audio_codec_algorithms::encode_ulaw(s))
        .collect()
}

/// Decode a G.711 A-law byte stream into 16-bit PCM samples.
///
/// Provided alongside the μ-law path because a SIP trunk may negotiate A-law
/// (PCMA) instead of μ-law (PCMU) (see `sip::sdp`); same `audio-codec-algorithms`
/// backend.
pub fn alaw_to_pcm16(alaw: &[u8]) -> Vec<i16> {
    alaw.iter()
        .map(|&b| audio_codec_algorithms::decode_alaw(b))
        .collect()
}

/// Encode 16-bit PCM samples into a G.711 A-law byte stream.
pub fn pcm16_to_alaw(pcm: &[i16]) -> Vec<u8> {
    pcm.iter()
        .map(|&s| audio_codec_algorithms::encode_alaw(s))
        .collect()
}

// ---------------------------------------------------------------------------
// Sample-rate conversion (streaming)
// ---------------------------------------------------------------------------

/// Fixed resampler input block size (frames per `rubato` call).
///
/// Independent of the carrier's frame size — the carry buffer feeds the resampler
/// in `BLOCK`-sized units regardless of how the input is chunked. 256 keeps the
/// sinc filter cheap while staying well above any single 20 ms telephony frame.
const BLOCK: usize = 256;

/// A mono sample-rate converter (e.g. 8k→16k, 24k→8k) backed by `rubato`.
///
/// Holds resampler state across calls so streaming chunks convert correctly
/// (see the module docs for the carry-buffer streaming design).
pub struct Resampler {
    /// Source sample rate in Hz.
    pub from_rate: u32,
    /// Destination sample rate in Hz.
    pub to_rate: u32,

    /// The underlying `rubato` resampler. `None` when `from_rate == to_rate`
    /// (pass-through — no resampling work to do).
    inner: Option<SincFixedIn<f32>>,
    /// Carry buffer of not-yet-resampled input samples (`< BLOCK` between calls).
    in_buf: Vec<i16>,
    /// Pre-allocated single-channel input scratch (`BLOCK` f32 samples) — avoids
    /// per-call allocation when handing data to `rubato`.
    scratch_in: Vec<f32>,
    /// Pre-allocated output scratch sized to the resampler's max output frames.
    scratch_out: Vec<Vec<f32>>,
}

impl Resampler {
    /// Create a resampler converting `from_rate` → `to_rate` (mono).
    ///
    /// Both rates must be non-zero. When the rates are equal the resampler is a
    /// cheap pass-through. The `rubato` resampler and all scratch buffers are
    /// allocated here, once, so [`process`](Self::process) never reallocates.
    pub fn new(from_rate: u32, to_rate: u32) -> Result<Self, FlowcatError> {
        if from_rate == 0 || to_rate == 0 {
            return Err(FlowcatError::Codec(format!(
                "invalid sample rate: from={from_rate} to={to_rate} (must be > 0)"
            )));
        }

        if from_rate == to_rate {
            return Ok(Self {
                from_rate,
                to_rate,
                inner: None,
                in_buf: Vec::new(),
                scratch_in: Vec::new(),
                scratch_out: Vec::new(),
            });
        }

        let ratio = to_rate as f64 / from_rate as f64;
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            oversampling_factor: 256,
            interpolation: SincInterpolationType::Cubic,
            window: WindowFunction::BlackmanHarris2,
        };

        // FixedIn: fixed BLOCK frames in, variable frames out — matches the
        // carrier's fixed input cadence. `max_resample_ratio_relative = 1.0`
        // because our ratio is constant for the life of the call.
        let inner = SincFixedIn::<f32>::new(ratio, 1.0, params, BLOCK, 1)
            .map_err(|e| FlowcatError::Codec(format!("rubato init {from_rate}->{to_rate}: {e}")))?;

        let scratch_out = inner.output_buffer_allocate(true);
        let scratch_in = Vec::with_capacity(BLOCK);

        Ok(Self {
            from_rate,
            to_rate,
            inner: Some(inner),
            in_buf: Vec::new(),
            scratch_in,
            scratch_out,
        })
    }

    /// Resample one mono PCM chunk from `from_rate` to `to_rate`.
    ///
    /// Accepts a chunk of **any** length (including empty); buffers a sub-block
    /// remainder internally and returns the samples produced so far. The returned
    /// [`AudioChunk`] is tagged at `to_rate`. Across a sequence of calls the
    /// concatenation of outputs is a continuous resampling of the concatenated
    /// inputs.
    pub fn process(&mut self, input: &AudioChunk) -> Result<AudioChunk, FlowcatError> {
        if input.sample_rate != self.from_rate {
            return Err(FlowcatError::Codec(format!(
                "resampler expected input at {} Hz, got {} Hz",
                self.from_rate, input.sample_rate
            )));
        }

        // Pass-through: rates match, no work.
        let resampler = match self.inner.as_mut() {
            None => return Ok(AudioChunk::new(input.pcm.clone(), self.to_rate)),
            Some(r) => r,
        };

        self.in_buf.extend_from_slice(&input.pcm);

        let mut out: Vec<i16> = Vec::new();
        let mut consumed = 0usize;
        while self.in_buf.len() - consumed >= BLOCK {
            let block = &self.in_buf[consumed..consumed + BLOCK];

            // Fill the pre-allocated f32 input scratch (i16 -> f32 in [-1, 1)).
            self.scratch_in.clear();
            self.scratch_in
                .extend(block.iter().map(|&s| s as f32 / 32768.0));

            let (_in_frames, out_frames) = resampler
                .process_into_buffer(
                    std::slice::from_ref(&self.scratch_in),
                    &mut self.scratch_out,
                    None,
                )
                .map_err(|e| FlowcatError::Codec(format!("rubato process: {e}")))?;

            out.extend(
                self.scratch_out[0][..out_frames]
                    .iter()
                    .map(|&s| f32_to_i16(s)),
            );
            consumed += BLOCK;
        }

        // Drop the consumed prefix, keep the `< BLOCK` remainder for next time.
        if consumed > 0 {
            self.in_buf.drain(..consumed);
        }

        Ok(AudioChunk::new(out, self.to_rate))
    }

    /// Flush any buffered remainder (< one `BLOCK`) at end-of-stream by
    /// zero-padding the final partial block, returning the tail output.
    ///
    /// Optional — the streaming `process` path never needs this mid-call, but a
    /// caller that wants every last sample (e.g. before finalizing a recording)
    /// can drain the resampler here. Returns an empty chunk for a pass-through or
    /// an already-empty buffer.
    pub fn flush(&mut self) -> Result<AudioChunk, FlowcatError> {
        let resampler = match self.inner.as_mut() {
            None => return Ok(AudioChunk::new(Vec::new(), self.to_rate)),
            Some(r) => r,
        };
        if self.in_buf.is_empty() {
            return Ok(AudioChunk::new(Vec::new(), self.to_rate));
        }

        self.scratch_in.clear();
        self.scratch_in
            .extend(self.in_buf.iter().map(|&s| s as f32 / 32768.0));
        self.scratch_in.resize(BLOCK, 0.0); // zero-pad the final partial block
        self.in_buf.clear();

        let (_in_frames, out_frames) = resampler
            .process_into_buffer(
                std::slice::from_ref(&self.scratch_in),
                &mut self.scratch_out,
                None,
            )
            .map_err(|e| FlowcatError::Codec(format!("rubato flush: {e}")))?;

        let out: Vec<i16> = self.scratch_out[0][..out_frames]
            .iter()
            .map(|&s| f32_to_i16(s))
            .collect();
        Ok(AudioChunk::new(out, self.to_rate))
    }
}

/// Convert a resampled f32 sample back to i16 with clamping (rubato can overshoot
/// slightly past ±1.0 around transients).
#[inline]
fn f32_to_i16(s: f32) -> i16 {
    (s * 32768.0).round().clamp(-32768.0, 32767.0) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Representative samples spanning the i16 range, including the rails.
    fn representative_samples() -> Vec<i16> {
        vec![
            0, 1, -1, 100, -100, 1000, -1000, 8000, -8000, 16000, -16000, 32000, -32000, 32767,
            -32768, 255, -256, 4095, -4096,
        ]
    }

    // ── G.711 absolute reference values (ITU-T G.191) ───────────────────────────
    // Pins companding to the standard so a codec-dependency swap or a mistaken
    // hand-roll that garbles/chops audio fails LOUDLY — the round-trip tests above
    // are self-referential and would NOT catch a wrong-companding regression.
    #[test]
    fn ulaw_decodes_to_g711_reference_values() {
        assert_eq!(
            ulaw_to_pcm16(&[0x00]),
            vec![-32124],
            "0x00 = negative full-scale"
        );
        assert_eq!(
            ulaw_to_pcm16(&[0x80]),
            vec![32124],
            "0x80 = positive full-scale"
        );
        assert_eq!(ulaw_to_pcm16(&[0xFF]), vec![0], "0xFF = -0 (silence)");
        assert_eq!(
            ulaw_to_pcm16(&[0x7F]),
            vec![0],
            "0x7F = +0 (the other zero)"
        );
    }

    #[test]
    fn ulaw_silence_round_trips_to_canonical_zero_code() {
        assert_eq!(
            pcm16_to_ulaw(&[0]),
            vec![0xFF],
            "encode(0) = canonical -0 code"
        );
        assert_eq!(ulaw_to_pcm16(&[0xFF]), vec![0], "0xFF decodes to silence");
    }

    #[test]
    fn ulaw_full_scale_encode_does_not_panic_or_wrap() {
        let enc = pcm16_to_ulaw(&[i16::MAX, i16::MIN]);
        let dec = ulaw_to_pcm16(&enc);
        assert!(
            dec[0] > 30000,
            "i16::MAX -> near +full-scale, got {}",
            dec[0]
        );
        assert!(
            dec[1] < -30000,
            "i16::MIN -> near -full-scale, got {}",
            dec[1]
        );
    }

    #[test]
    fn alaw_decodes_to_g711_reference_values() {
        assert_eq!(alaw_to_pcm16(&[0x00]), vec![-5504]);
        assert_eq!(alaw_to_pcm16(&[0xFF]), vec![848]);
        // A-law has no exact-zero code; 0x55/0xD5 are the smallest-magnitude
        // ("silence") codewords, decoding to -8 / +8 (verified against the
        // G.191 table in audio-codec-algorithms 0.8.0).
        assert_eq!(alaw_to_pcm16(&[0x55]), vec![-8]);
        assert_eq!(alaw_to_pcm16(&[0xD5]), vec![8]);
    }

    #[test]
    fn empty_lengths_are_preserved_both_directions() {
        assert!(ulaw_to_pcm16(&[]).is_empty());
        assert!(pcm16_to_ulaw(&[]).is_empty());
        assert!(alaw_to_pcm16(&[]).is_empty());
        assert!(pcm16_to_alaw(&[]).is_empty());
    }

    // ── Resampler robustness: empty, sub-BLOCK, the 16k→8k sibling rate, clamp ──
    #[test]
    fn resample_empty_chunk_does_not_panic() {
        let mut rs = Resampler::new(8000, 16000).unwrap();
        let out = rs.process(&AudioChunk::new(vec![], 8000)).unwrap();
        assert_eq!(out.sample_rate, 16000);
        assert!(out.is_empty(), "empty in -> empty out");
        assert!(rs.flush().unwrap().is_empty());
    }

    #[test]
    fn resample_sub_block_chunk_buffers_without_panic() {
        let mut rs = Resampler::new(8000, 16000).unwrap();
        let out = rs.process(&AudioChunk::new(sine(8000, 37), 8000)).unwrap();
        assert!(out.is_empty(), "37 (< BLOCK) samples are fully buffered");
        let tail = rs.flush().unwrap();
        assert_eq!(tail.sample_rate, 16000);
        assert!(tail.len() < BLOCK * 2);
    }

    #[test]
    fn resample_16k_to_8k_roughly_halves() {
        let mut rs = Resampler::new(16000, 8000).unwrap();
        let total_in = 3200usize; // 200 ms @ 16k
        let out = rs
            .process(&AudioChunk::new(sine(16000, total_in), 16000))
            .unwrap();
        let total_out = out.len() + rs.flush().unwrap().len();
        assert_eq!(out.sample_rate, 8000);
        let expected = total_in / 2;
        assert!(
            ((expected - BLOCK)..=(expected + BLOCK)).contains(&total_out),
            "16k->8k expected ~{expected}, got {total_out}"
        );
    }

    #[test]
    fn f32_to_i16_clamps_overshoot_without_wrapping() {
        assert_eq!(f32_to_i16(2.0), i16::MAX);
        assert_eq!(f32_to_i16(-2.0), i16::MIN);
        assert_eq!(f32_to_i16(0.0), 0);
        assert_eq!(f32_to_i16(1.0), i16::MAX); // +1.0 -> 32768 clamps to 32767
        assert_eq!(f32_to_i16(-1.0), i16::MIN);
    }

    #[test]
    fn ulaw_decode_then_pcm_is_stable_through_reencode() {
        // μ-law is a *fixed 8-bit code*. Re-encoding a decoded sample is stable
        // **at the PCM level** for every codeword: decode→encode→decode yields the
        // exact same PCM value as the first decode. (The codeword itself is not
        // perfectly idempotent for *one* value — G.711 has two zero codes, 0x7F
        // "+0" and 0xFF "-0", both decoding to PCM 0; re-encoding 0 picks the
        // canonical -0 code. That is correct companding, not a wrapper bug — so we
        // assert the meaningful PCM-stable property, which holds for all 256 codes.)
        for code in 0u8..=255 {
            let pcm1 = ulaw_to_pcm16(&[code]);
            let reencoded = pcm16_to_ulaw(&pcm1);
            let pcm2 = ulaw_to_pcm16(&reencoded);
            assert_eq!(
                pcm2, pcm1,
                "μ-law code {code} not PCM-stable: {pcm1:?} -> {reencoded:?} -> {pcm2:?}"
            );
        }
    }

    #[test]
    fn pcm_ulaw_roundtrip_within_g711_quantization() {
        // PCM -> μ-law -> PCM is lossy 8-bit companding. μ-law is logarithmic, so
        // the meaningful bound is *relative*: away from zero the round-trip error
        // is a small fraction of the magnitude. For small magnitudes the absolute
        // step is tiny, and at the very rails the topmost reconstruction level sits
        // ~2% below full-scale — all expected. We assert both bounds.
        let pcm = representative_samples();
        let ulaw = pcm16_to_ulaw(&pcm);
        let back = ulaw_to_pcm16(&ulaw);
        assert_eq!(back.len(), pcm.len());
        for (orig, rt) in pcm.iter().zip(back.iter()) {
            let o = *orig as i32;
            let err = (o - *rt as i32).abs();
            // Small absolute floor for near-zero samples, OR <= ~3% relative for
            // larger magnitudes (μ-law's near-constant relative quantization).
            let rel_ok = err as f64 <= (o.unsigned_abs() as f64) * 0.03;
            assert!(
                err <= 8 || rel_ok,
                "μ-law quantization error out of bounds: orig={orig} roundtrip={rt} err={err}"
            );
        }
    }

    #[test]
    fn alaw_decode_then_pcm_is_stable_through_reencode() {
        for code in 0u8..=255 {
            let pcm1 = alaw_to_pcm16(&[code]);
            let reencoded = pcm16_to_alaw(&pcm1);
            let pcm2 = alaw_to_pcm16(&reencoded);
            assert_eq!(pcm2, pcm1, "A-law code {code} not PCM-stable");
        }
    }

    /// A 440 Hz-ish sine generator at a given rate, `n` samples.
    fn sine(rate: u32, n: usize) -> Vec<i16> {
        (0..n)
            .map(|i| {
                let t = i as f32 / rate as f32;
                (16000.0 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()) as i16
            })
            .collect()
    }

    #[test]
    fn rate_passthrough_returns_input_unchanged() {
        let mut rs = Resampler::new(8000, 8000).unwrap();
        let chunk = AudioChunk::new(sine(8000, 160), 8000);
        let out = rs.process(&chunk).unwrap();
        assert_eq!(out.sample_rate, 8000);
        assert_eq!(out.pcm, chunk.pcm);
    }

    #[test]
    fn resample_8k_to_16k_roughly_doubles_over_a_stream() {
        // Feed several variable-size 8 kHz chunks (telephony cadence) and confirm
        // the *cumulative* output is ~2x the cumulative input — proving the
        // streaming carry-buffer is continuous, not a single-chunk fluke.
        let mut rs = Resampler::new(8000, 16000).unwrap();
        let chunk_sizes = [160usize, 160, 80, 320, 200, 160, 240]; // varied, incl. < BLOCK
        let total_in: usize = chunk_sizes.iter().sum();

        let mut total_out = 0usize;
        let mut phase = 0usize;
        for &n in &chunk_sizes {
            let samples: Vec<i16> = sine(8000, phase + n)[phase..].to_vec();
            phase += n;
            let out = rs.process(&AudioChunk::new(samples, 8000)).unwrap();
            assert_eq!(out.sample_rate, 16000);
            total_out += out.len();
        }
        total_out += rs.flush().unwrap().len();

        let expected = total_in * 2;
        let lo = expected.saturating_sub(BLOCK * 2);
        let hi = expected + BLOCK * 2;
        assert!(
            (lo..=hi).contains(&total_out),
            "8k->16k expected ~{expected} samples (in={total_in}), got {total_out}"
        );
    }

    #[test]
    fn resample_24k_to_8k_roughly_thirds_over_a_stream() {
        // 24 kHz Gemini-out -> 8 kHz carrier: cumulative output ~= input / 3,
        // checked over a multi-chunk stream with varied (incl. sub-BLOCK) sizes.
        let mut rs = Resampler::new(24000, 8000).unwrap();
        let chunk_sizes = [480usize, 480, 240, 960, 300, 480]; // 20ms @24k = 480
        let total_in: usize = chunk_sizes.iter().sum();

        let mut total_out = 0usize;
        let mut phase = 0usize;
        for &n in &chunk_sizes {
            let samples: Vec<i16> = sine(24000, phase + n)[phase..].to_vec();
            phase += n;
            let out = rs.process(&AudioChunk::new(samples, 24000)).unwrap();
            assert_eq!(out.sample_rate, 8000);
            total_out += out.len();
        }
        total_out += rs.flush().unwrap().len();

        let expected = total_in / 3;
        let lo = expected.saturating_sub(BLOCK);
        let hi = expected + BLOCK;
        assert!(
            (lo..=hi).contains(&total_out),
            "24k->8k expected ~{expected} samples (in={total_in}), got {total_out}"
        );
    }

    #[test]
    fn resample_8k_to_24k_roughly_triples() {
        let mut rs = Resampler::new(8000, 24000).unwrap();
        let total_in = 1600usize; // 200 ms
        let out = rs
            .process(&AudioChunk::new(sine(8000, total_in), 8000))
            .unwrap();
        let tail = rs.flush().unwrap().len();
        let total_out = out.len() + tail;
        let expected = total_in * 3;
        assert!(
            ((expected - BLOCK * 3)..=(expected + BLOCK * 3)).contains(&total_out),
            "8k->24k expected ~{expected}, got {total_out}"
        );
    }

    #[test]
    fn rejects_zero_rate() {
        assert!(Resampler::new(0, 16000).is_err());
        assert!(Resampler::new(8000, 0).is_err());
    }

    #[test]
    fn rejects_mismatched_input_rate() {
        let mut rs = Resampler::new(8000, 16000).unwrap();
        let wrong = AudioChunk::new(vec![0i16; 160], 16000);
        assert!(rs.process(&wrong).is_err());
    }
}
