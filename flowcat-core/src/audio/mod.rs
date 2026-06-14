// SPDX-License-Identifier: Apache-2.0
//
//! Audio intelligence + call recording (PROCESSOR-DESIGN §6.1).
//!
//! This module is the home for the **audio-intelligence** processors that sit on
//! the hot media path inside the cascaded pipeline — VAD, end-of-turn, noise
//! filters, and mixers — alongside the existing call **recorder**. The analyzer
//! *traits* (`VadAnalyzer`/`TurnAnalyzer`) are frozen in [`crate::service`]; the
//! ONNX (`ort`) impls compile only under the `vad-ort` feature so flowcat-core
//! stays dependency-light by default.
//!
//! Module map:
//!
//! - [`vad`] — `VadAnalyzer` impls (Silero behind `vad-ort`, a mock always).
//! - [`turn`] — `TurnAnalyzer` impls (Smart-Turn behind `vad-ort`, a mock always).
//! - [`strategy`] — turn start/stop/mute + interruption strategies (pure logic).
//! - [`filter`] — RNNoise/Koala audio-filter processors (Krisp deferred).
//! - [`mixer`] — audio mixers.
//!
//! ## Call recording (existing)
//!
//! Tap both legs (caller + bot), mono-mix, render to WAV bytes.
//!
//! The recorder accumulates audio from each leg and produces a single mono WAV
//! (via `hound`) suitable for upload to the object store (see DESIGN.md "Audio path").
//!
//! The two legs typically arrive at different sample rates (caller ≈ 8 kHz μ-law
//! decoded, bot ≈ 24 kHz from Gemini Live), so each leg is buffered with its own
//! rate and resampled to the recorder's common `sample_rate` only at render time.
//! Mixing is a plain per-sample sum clamped back into `i16`.

pub mod filter;
pub mod mixer;
pub mod strategy;
pub mod turn;
pub mod vad;

// ---- Audio-intelligence re-exports (ergonomic flat names) ----
#[cfg(feature = "filter-rnnoise")]
pub use filter::RnnoiseFilter;
pub use filter::{AudioFilter, AudioFilterProcessor, IdentityFilter};
pub use mixer::{AudioMixer, MixerProcessor, SilenceMixer, SoundfileMixer};
pub use strategy::{
    AlwaysInterrupt, AlwaysMute, ExternalTurnStart, FirstSpeechMute, FunctionCallMute,
    InterruptionStrategy, MinWordsInterrupt, MinWordsTurnStart, MuteStrategy,
    MuteUntilFirstBotComplete, SpeechTimeoutStop, TurnAnalyzerStop, TurnStartDecision,
    TurnStartStrategy, TurnStopDecision, TurnStopStrategy, VadTurnStart,
};
#[cfg(feature = "vad-ort")]
pub use turn::SmartTurn;
pub use turn::{MockTurn, TurnProcessor, TurnSilenceTracker};
#[cfg(feature = "vad-ort")]
pub use vad::SileroVad;
pub use vad::{window_volume, MockVad, VadProcessor, VadStateMachine};

use std::collections::HashMap;
use std::io::Cursor;
use std::time::{Duration, Instant};

use hound::{SampleFormat, WavSpec, WavWriter};

use crate::codec::Resampler;
use crate::error::FlowcatError;
use crate::types::AudioChunk;

/// A chunk placed on the call's wall-clock timeline (`offset_s` = seconds from
/// call start). The placement is already advanced past the leg's prior audio so
/// same-leg chunks never overlap; render silence-pads the gap up to it.
struct PlacedChunk {
    offset_s: f64,
    chunk: AudioChunk,
}

/// Accumulates inbound (caller) and outbound (bot) audio on a shared wall-clock
/// timeline and mixes them down to a single mono WAV byte buffer.
///
/// Each leg records only its *active* chunks, but every chunk is stamped with its
/// arrival time relative to call start, so render silence-pads the gaps and the
/// two legs stay aligned — a caller turn at t=2s and a bot turn at t=5s land where
/// they actually happened instead of both starting at sample 0 (which made the
/// two voices play over each other in the recording).
pub struct AudioRecorder {
    /// Output WAV sample rate in Hz.
    pub sample_rate: u32,
    /// Buffered caller-leg chunks (kept at their source rate; resampled at render).
    inbound: Vec<PlacedChunk>,
    /// Buffered bot-leg chunks (kept at their source rate; resampled at render).
    outbound: Vec<PlacedChunk>,
    /// Call-start reference for the live [`push_inbound`](Self::push_inbound) /
    /// [`push_outbound`](Self::push_outbound) taps; set lazily on the first push.
    start: Option<Instant>,
    /// Running end-of-audio per leg (seconds): keeps same-leg chunks contiguous
    /// and makes [`duration_seconds`](Self::duration_seconds) O(1).
    inbound_end_s: f64,
    outbound_end_s: f64,
}

impl AudioRecorder {
    /// Create a recorder that renders mixed audio at `sample_rate` Hz.
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            inbound: Vec::new(),
            outbound: Vec::new(),
            start: None,
            inbound_end_s: 0.0,
            outbound_end_s: 0.0,
        }
    }

    /// Elapsed time since the first push (call start), set lazily on first use.
    fn tick(&mut self) -> Duration {
        match self.start {
            Some(s) => s.elapsed(),
            None => {
                self.start = Some(Instant::now());
                Duration::ZERO
            }
        }
    }

    /// Append a chunk from the inbound (caller) leg, stamped at the live wall clock.
    ///
    /// Stored at its own rate; resample + silence-pad happen in
    /// [`render_wav`](Self::render_wav). Empty chunks are dropped.
    pub fn push_inbound(&mut self, chunk: &AudioChunk) {
        let at = self.tick();
        self.push_inbound_at(at, chunk);
    }

    /// Append a chunk from the outbound (bot) leg, stamped at the live wall clock.
    pub fn push_outbound(&mut self, chunk: &AudioChunk) {
        let at = self.tick();
        self.push_outbound_at(at, chunk);
    }

    /// Append an inbound chunk at an explicit timeline offset from call start.
    ///
    /// Deterministic entry point (no wall clock): the live taps go through
    /// [`push_inbound`](Self::push_inbound); tests place chunks at known offsets.
    pub fn push_inbound_at(&mut self, offset: Duration, chunk: &AudioChunk) {
        if let Some(p) = place(offset, chunk, self.inbound_end_s) {
            self.inbound_end_s = p.offset_s + chunk_duration_s(chunk);
            self.inbound.push(p);
        }
    }

    /// Append an outbound chunk at an explicit timeline offset from call start.
    pub fn push_outbound_at(&mut self, offset: Duration, chunk: &AudioChunk) {
        if let Some(p) = place(offset, chunk, self.outbound_end_s) {
            self.outbound_end_s = p.offset_s + chunk_duration_s(chunk);
            self.outbound.push(p);
        }
    }

    /// Wall-clock call duration in seconds — the end of the later leg's last audio.
    /// Reported back to the control plane as `usage_metrics.duration_seconds`.
    pub fn duration_seconds(&self) -> f64 {
        self.inbound_end_s.max(self.outbound_end_s)
    }

    /// Mix both legs to mono and render a complete WAV file as bytes.
    ///
    /// Each leg is resampled to `self.sample_rate` and laid out on the shared
    /// timeline (silence-padding the gaps between turns); the two streams are
    /// summed sample-by-sample (zero-padding the shorter leg to the longer), each
    /// sum is clamped into `i16`, and the result is written as a mono 16-bit PCM
    /// WAV via `hound`.
    pub fn render_wav(&self) -> Result<Vec<u8>, FlowcatError> {
        let caller = render_leg(&self.inbound, self.sample_rate)?;
        let bot = render_leg(&self.outbound, self.sample_rate)?;

        let n = caller.len().max(bot.len());
        let mut mixed = Vec::with_capacity(n);
        for i in 0..n {
            let a = *caller.get(i).unwrap_or(&0) as i32;
            let b = *bot.get(i).unwrap_or(&0) as i32;
            mixed.push((a + b).clamp(i16::MIN as i32, i16::MAX as i32) as i16);
        }

        let spec = WavSpec {
            channels: 1,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };

        // Write into an owned byte buffer. `Cursor<&mut Vec<u8>>` is `Write +
        // Seek` (which hound requires), and crucially leaves us owning `buf`
        // after `finalize()` consumes the writer.
        let mut buf: Vec<u8> = Vec::new();
        {
            let mut writer = WavWriter::new(Cursor::new(&mut buf), spec)
                .map_err(|e| FlowcatError::Codec(format!("wav header: {e}")))?;
            for &s in &mixed {
                writer
                    .write_sample(s)
                    .map_err(|e| FlowcatError::Codec(format!("wav sample: {e}")))?;
            }
            writer
                .finalize()
                .map_err(|e| FlowcatError::Codec(format!("wav finalize: {e}")))?;
        }
        Ok(buf)
    }
}

/// Real-time duration of a chunk in seconds (rate-independent: samples ÷ rate).
fn chunk_duration_s(chunk: &AudioChunk) -> f64 {
    if chunk.sample_rate == 0 {
        0.0
    } else {
        chunk.pcm.len() as f64 / chunk.sample_rate as f64
    }
}

/// Resolve a chunk's timeline placement: never before the leg's prior audio (so
/// same-leg chunks stay contiguous and never self-overlap), else at its arrival
/// offset. Returns `None` for empty chunks (dropped, as the recorder always has).
fn place(offset: Duration, chunk: &AudioChunk, leg_end_s: f64) -> Option<PlacedChunk> {
    if chunk.is_empty() {
        return None;
    }
    Some(PlacedChunk {
        offset_s: offset.as_secs_f64().max(leg_end_s),
        chunk: chunk.clone(),
    })
}

/// Resample one leg to `target_rate` and lay each chunk on the timeline, padding
/// silence up to each chunk's offset so the gaps between turns are preserved.
///
/// Reuses one [`Resampler`] per distinct source rate (so a steady single-rate leg
/// builds exactly one resampler and preserves its filter state across chunks);
/// flushes each resampler's tail at the end so no samples are dropped. Chunks
/// already at `target_rate` are copied straight through.
fn render_leg(chunks: &[PlacedChunk], target_rate: u32) -> Result<Vec<i16>, FlowcatError> {
    let mut out: Vec<i16> = Vec::new();
    let mut resamplers: HashMap<u32, Resampler> = HashMap::new();

    for pc in chunks {
        // Silence-pad the output up to this chunk's wall-clock placement.
        let target = (pc.offset_s * target_rate as f64).round() as usize;
        if out.len() < target {
            out.resize(target, 0);
        }

        if pc.chunk.sample_rate == target_rate {
            out.extend_from_slice(&pc.chunk.pcm);
            continue;
        }
        let rs = match resamplers.get_mut(&pc.chunk.sample_rate) {
            Some(rs) => rs,
            None => {
                let rs = Resampler::new(pc.chunk.sample_rate, target_rate)?;
                resamplers.entry(pc.chunk.sample_rate).or_insert(rs)
            }
        };
        let res = rs.process(&pc.chunk)?;
        out.extend_from_slice(&res.pcm);
    }

    // Drain each resampler's buffered remainder.
    for rs in resamplers.values_mut() {
        let tail = rs.flush()?;
        out.extend_from_slice(&tail.pcm);
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hound::WavReader;

    fn tone(rate: u32, n: usize, freq: f32, amp: f32) -> AudioChunk {
        let pcm = (0..n)
            .map(|i| {
                let t = i as f32 / rate as f32;
                (amp * (2.0 * std::f32::consts::PI * freq * t).sin()) as i16
            })
            .collect();
        AudioChunk::new(pcm, rate)
    }

    /// Parse WAV bytes back with hound and return (spec, samples).
    fn parse_wav(bytes: &[u8]) -> (WavSpec, Vec<i16>) {
        let mut reader = WavReader::new(Cursor::new(bytes)).expect("valid WAV header");
        let spec = reader.spec();
        let samples: Vec<i16> = reader
            .samples::<i16>()
            .map(|s| s.expect("readable sample"))
            .collect();
        (spec, samples)
    }

    #[test]
    fn renders_valid_wav_header_at_recorder_rate() {
        let mut rec = AudioRecorder::new(8000);
        // Both legs already at the recorder rate -> no resampling, exact counts.
        rec.push_inbound_at(Duration::ZERO, &tone(8000, 800, 440.0, 8000.0));
        rec.push_outbound_at(Duration::ZERO, &tone(8000, 800, 660.0, 8000.0));

        let bytes = rec.render_wav().unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");

        let (spec, samples) = parse_wav(&bytes);
        assert_eq!(spec.channels, 1, "mono");
        assert_eq!(spec.sample_rate, 8000);
        assert_eq!(spec.bits_per_sample, 16);
        assert_eq!(spec.sample_format, SampleFormat::Int);
        assert_eq!(samples.len(), 800, "same-rate legs mix 1:1");
    }

    #[test]
    fn mixes_two_legs_by_summation() {
        let mut rec = AudioRecorder::new(8000);
        // Two constant DC-ish offsets so the mix is predictable: sum then clamp.
        rec.push_inbound_at(Duration::ZERO, &AudioChunk::new(vec![1000i16; 100], 8000));
        rec.push_outbound_at(Duration::ZERO, &AudioChunk::new(vec![2000i16; 100], 8000));

        let (_, samples) = parse_wav(&rec.render_wav().unwrap());
        assert_eq!(samples.len(), 100);
        assert!(samples.iter().all(|&s| s == 3000), "1000 + 2000 = 3000");
    }

    #[test]
    fn mix_clamps_instead_of_wrapping() {
        let mut rec = AudioRecorder::new(8000);
        rec.push_inbound_at(Duration::ZERO, &AudioChunk::new(vec![30000i16; 10], 8000));
        rec.push_outbound_at(Duration::ZERO, &AudioChunk::new(vec![30000i16; 10], 8000));

        let (_, samples) = parse_wav(&rec.render_wav().unwrap());
        // 60000 must saturate to i16::MAX, never wrap to a negative value.
        assert!(samples.iter().all(|&s| s == i16::MAX));
    }

    #[test]
    fn longer_leg_determines_length_shorter_zero_padded() {
        let mut rec = AudioRecorder::new(8000);
        rec.push_inbound_at(Duration::ZERO, &AudioChunk::new(vec![500i16; 200], 8000));
        rec.push_outbound_at(Duration::ZERO, &AudioChunk::new(vec![500i16; 50], 8000));

        let (_, samples) = parse_wav(&rec.render_wav().unwrap());
        assert_eq!(samples.len(), 200, "length = longer leg");
        // Overlap region summed; tail is caller-only (bot zero-padded).
        assert_eq!(samples[0], 1000);
        assert_eq!(samples[199], 500);
    }

    #[test]
    fn resamples_legs_at_different_rates_to_common_rate() {
        // Caller at 8 kHz, bot at 24 kHz, recorder at 16 kHz: both legs are
        // resampled before mixing and the WAV is still valid & mono @ 16 kHz.
        let mut rec = AudioRecorder::new(16000);
        rec.push_inbound_at(Duration::ZERO, &tone(8000, 1600, 440.0, 8000.0)); // 200 ms
        rec.push_outbound_at(Duration::ZERO, &tone(24000, 4800, 660.0, 8000.0)); // 200 ms

        let bytes = rec.render_wav().unwrap();
        let (spec, samples) = parse_wav(&bytes);
        assert_eq!(spec.sample_rate, 16000);
        assert_eq!(spec.channels, 1);
        // ~200 ms @ 16 kHz ≈ 3200 samples (resampler delay/edge tolerance).
        assert!(
            (3000..=3400).contains(&samples.len()),
            "expected ~3200 samples @16k, got {}",
            samples.len()
        );
    }

    #[test]
    fn empty_recorder_renders_valid_empty_wav() {
        let rec = AudioRecorder::new(8000);
        let bytes = rec.render_wav().unwrap();
        let (spec, samples) = parse_wav(&bytes);
        assert_eq!(spec.sample_rate, 8000);
        assert_eq!(samples.len(), 0);
    }

    #[test]
    fn empty_chunks_are_ignored() {
        let mut rec = AudioRecorder::new(8000);
        rec.push_inbound_at(Duration::ZERO, &AudioChunk::new(vec![], 8000));
        rec.push_outbound_at(Duration::ZERO, &AudioChunk::new(vec![], 8000));
        let (_, samples) = parse_wav(&rec.render_wav().unwrap());
        assert_eq!(samples.len(), 0);
    }

    #[test]
    fn sequential_turns_do_not_overlap_on_the_timeline() {
        // Regression: legs used to be summed from sample 0, so a user turn and a
        // later bot turn played simultaneously. With timeline placement the bot
        // turn lands at its real offset and the two never sum together.
        let mut rec = AudioRecorder::new(8000);
        rec.push_inbound_at(Duration::ZERO, &AudioChunk::new(vec![1000i16; 100], 8000));
        rec.push_outbound_at(
            Duration::from_millis(500),
            &AudioChunk::new(vec![2000i16; 100], 8000),
        );

        let (_, samples) = parse_wav(&rec.render_wav().unwrap());
        // Bot lands 0.5 s in (4000 samples @ 8 kHz) and runs 100 samples.
        assert_eq!(samples.len(), 4100, "timeline spans to the bot turn");
        assert_eq!(samples[50], 1000, "user turn, bot silent");
        assert_eq!(samples[4050], 2000, "bot turn, user silent");
        assert!(
            !samples.contains(&3000),
            "sequential turns must not overlap (no summed 1000+2000)"
        );
    }

    #[test]
    fn duration_seconds_spans_to_last_audio() {
        let mut rec = AudioRecorder::new(8000);
        // Caller: 1.0 s at t=0.  Bot: 0.5 s starting at t=2.0 s → ends at 2.5 s.
        rec.push_inbound_at(Duration::ZERO, &AudioChunk::new(vec![1i16; 8000], 8000));
        rec.push_outbound_at(
            Duration::from_secs(2),
            &AudioChunk::new(vec![1i16; 4000], 8000),
        );
        assert!(
            (rec.duration_seconds() - 2.5).abs() < 1e-6,
            "duration spans to the later leg's last audio, got {}",
            rec.duration_seconds()
        );
    }
}
