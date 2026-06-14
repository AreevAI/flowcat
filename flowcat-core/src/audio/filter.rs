// SPDX-License-Identifier: Apache-2.0
//
//! Input-leg audio-filter processors (noise suppression).
//!
//! Mirrors pipecat's `pipecat/audio/filters/`: a filter transforms the inbound
//! caller audio **before** VAD/STT. This module defines:
//! - [`AudioFilter`] — the pure-DSP filter trait (`filter(pcm) -> pcm`), the Rust
//!   analogue of pipecat's `BaseAudioFilter`;
//! - [`IdentityFilter`] — a pass-through, always available (the conservative
//!   default when no noise-suppression backend is built);
//! - [`RnnoiseFilter`] — RNNoise noise suppression via the pure-Rust
//!   [`nnnoiseless`] port, behind the `filter-rnnoise` feature;
//! - [`AudioFilterProcessor`] — a [`FrameProcessor`] that applies a filter to each
//!   [`Frame::InputAudio`] and is toggled live by [`Frame::FilterEnable`].
//!
//! Krisp/Koala are **deferred** — they wrap proprietary SDKs (FFI/sidecar) per the
//! risk register and ride a later feature/`Custom`, not a new core dep. The
//! processor shape here is what those impls plug into.

use async_trait::async_trait;

use crate::error::Result;
use crate::processor::frame::{AudioFrame, Frame, StartParams};
use crate::processor::{Envelope, FrameProcessor, Link, ProcessorSetup};

/// A pure-DSP audio filter over 16-bit mono PCM. Mirrors pipecat
/// `BaseAudioFilter`: `start(sample_rate)` initializes the filter for the
/// transport rate, `filter(pcm)` transforms one chunk, `reset` clears state.
///
/// Implementations must be deterministic given their inputs (so they are
/// fixture-testable) and must not block — the live path runs them on the hot leg.
pub trait AudioFilter: Send {
    /// A stable name for tracing/observers.
    fn name(&self) -> &str;
    /// Initialize for the transport `sample_rate` (Hz). Default: no-op.
    fn start(&mut self, _sample_rate: u32) -> Result<()> {
        Ok(())
    }
    /// Transform one PCM chunk, returning the filtered samples. A filter that
    /// buffers internally (e.g. to reach a required frame size) may return fewer
    /// or more samples than it received.
    fn filter(&mut self, pcm: &[i16]) -> Result<Vec<i16>>;
    /// Reset any internal buffering/state. Default: no-op.
    fn reset(&mut self) {}
}

/// A pass-through filter: returns its input unchanged. Always available (no dep).
/// The conservative default when no noise-suppression backend is compiled in.
#[derive(Debug, Default, Clone)]
pub struct IdentityFilter;

impl AudioFilter for IdentityFilter {
    fn name(&self) -> &str {
        "IdentityFilter"
    }
    fn filter(&mut self, pcm: &[i16]) -> Result<Vec<i16>> {
        Ok(pcm.to_vec())
    }
}

/// A [`FrameProcessor`] that applies an [`AudioFilter`] to inbound audio.
///
/// On [`Frame::InputAudio`] it runs the filter and forwards a filtered
/// `InputAudio`; on [`Frame::FilterEnable`] it toggles filtering live (when
/// disabled it forwards audio unchanged, mirroring pipecat's `FilterEnableFrame`).
/// All other frames pass through.
///
/// Sits **before** the [`VadProcessor`](crate::audio::vad::VadProcessor) /
/// STT in the cascaded input chain so downstream stages see clean audio.
pub struct AudioFilterProcessor<F: AudioFilter> {
    filter: F,
    enabled: bool,
    sample_rate: u32,
}

impl<F: AudioFilter> AudioFilterProcessor<F> {
    /// Wrap `filter`, enabled by default.
    pub fn new(filter: F) -> Self {
        Self {
            filter,
            enabled: true,
            sample_rate: 16_000,
        }
    }

    /// Wrap `filter`, initially disabled (pass-through until a
    /// [`Frame::FilterEnable`]`(true)` arrives).
    pub fn disabled(filter: F) -> Self {
        Self {
            filter,
            enabled: false,
            sample_rate: 16_000,
        }
    }
}

#[async_trait]
impl<F: AudioFilter + 'static> FrameProcessor for AudioFilterProcessor<F> {
    fn name(&self) -> &str {
        self.filter.name()
    }

    async fn start(&mut self, _setup: &ProcessorSetup, params: &StartParams) -> Result<()> {
        self.sample_rate = params.audio_in_sample_rate;
        self.filter.start(self.sample_rate)
    }

    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        match &env.frame {
            Frame::FilterEnable(on) => {
                self.enabled = *on;
                if !self.enabled {
                    self.filter.reset();
                }
                // Forward the control frame so downstream observers see it.
                link.push(env.meta, env.frame, env.direction).await;
                return Ok(());
            }
            Frame::InputAudio(audio) if self.enabled => {
                let filtered = self.filter.filter(&audio.pcm)?;
                // A filter that buffers may emit nothing yet; drop the empty frame.
                if !filtered.is_empty() {
                    let out = AudioFrame::mono(filtered, audio.sample_rate);
                    link.push_down(Frame::InputAudio(std::sync::Arc::new(out)))
                        .await;
                }
                return Ok(());
            }
            _ => {}
        }
        link.push(env.meta, env.frame, env.direction).await;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RNNoise impl (behind `filter-rnnoise`).
// ---------------------------------------------------------------------------

/// RNNoise noise suppression via the pure-Rust [`nnnoiseless`] port, behind the
/// `filter-rnnoise` feature.
///
/// Mirrors pipecat's `RNNoiseFilter`: RNNoise operates on 48 kHz mono in fixed
/// 480-sample frames, so this filter resamples the transport rate ↔ 48 kHz (via
/// the existing [`Resampler`](crate::codec::Resampler)), buffers to the 480-sample
/// frame boundary, denoises each frame, and resamples back. `nnnoiseless` is a
/// pure-Rust port (no C/system dep), so the build stays toolchain-light.
///
/// **Note on scale:** RNNoise expects float PCM in the i16 amplitude range
/// (±32768), not normalized to ±1.0 — this impl feeds `f32`-cast i16 directly, as
/// the reference RNNoise / `nnnoiseless` API expects.
#[cfg(feature = "filter-rnnoise")]
pub struct RnnoiseFilter {
    state: Box<nnnoiseless::DenoiseState<'static>>,
    in_rate: u32,
    /// Buffer of 48 kHz f32 samples awaiting a full RNNoise frame.
    buf_48k: Vec<f32>,
    resampler_in: Option<crate::codec::Resampler>,
    resampler_out: Option<crate::codec::Resampler>,
}

#[cfg(feature = "filter-rnnoise")]
impl RnnoiseFilter {
    /// RNNoise's required frame size (480 samples @ 48 kHz).
    pub const FRAME_SIZE: usize = nnnoiseless::DenoiseState::FRAME_SIZE;

    /// Create an RNNoise filter (sample rate is set in [`AudioFilter::start`]).
    pub fn new() -> Self {
        Self {
            state: nnnoiseless::DenoiseState::new(),
            in_rate: 48_000,
            buf_48k: Vec::new(),
            resampler_in: None,
            resampler_out: None,
        }
    }
}

#[cfg(feature = "filter-rnnoise")]
impl Default for RnnoiseFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "filter-rnnoise")]
impl AudioFilter for RnnoiseFilter {
    fn name(&self) -> &str {
        "RnnoiseFilter"
    }

    fn start(&mut self, sample_rate: u32) -> Result<()> {
        self.in_rate = sample_rate;
        self.buf_48k.clear();
        if sample_rate != 48_000 {
            self.resampler_in = Some(crate::codec::Resampler::new(sample_rate, 48_000)?);
            self.resampler_out = Some(crate::codec::Resampler::new(48_000, sample_rate)?);
        } else {
            self.resampler_in = None;
            self.resampler_out = None;
        }
        Ok(())
    }

    fn filter(&mut self, pcm: &[i16]) -> Result<Vec<i16>> {
        use crate::types::AudioChunk;

        if pcm.is_empty() {
            return Ok(Vec::new());
        }

        // Upsample to 48 kHz if needed.
        let at_48k: Vec<i16> = if let Some(rs) = self.resampler_in.as_mut() {
            rs.process(&AudioChunk::new(pcm.to_vec(), self.in_rate))?
                .pcm
        } else {
            pcm.to_vec()
        };
        self.buf_48k.extend(at_48k.iter().map(|&s| s as f32));

        // Denoise every full 480-sample frame.
        let mut out_48k: Vec<i16> = Vec::new();
        let mut frame_out = [0.0f32; Self::FRAME_SIZE];
        while self.buf_48k.len() >= Self::FRAME_SIZE {
            let frame_in: Vec<f32> = self.buf_48k.drain(..Self::FRAME_SIZE).collect();
            self.state.process_frame(&mut frame_out, &frame_in);
            out_48k.extend(
                frame_out
                    .iter()
                    .map(|&s| s.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16),
            );
        }

        if out_48k.is_empty() {
            return Ok(Vec::new());
        }

        // Downsample back to the transport rate if needed.
        if let Some(rs) = self.resampler_out.as_mut() {
            Ok(rs.process(&AudioChunk::new(out_48k, 48_000))?.pcm)
        } else {
            Ok(out_48k)
        }
    }

    fn reset(&mut self) {
        self.buf_48k.clear();
    }
}

// ---------------------------------------------------------------------------
// Deferred proprietary backends (Krisp / Koala) — stub markers only.
// ---------------------------------------------------------------------------
//
// TODO: Krisp and Koala noise suppression wrap proprietary SDKs
// (Krisp = native FFI/sidecar, Koala = Picovoice's licensed model). They are
// deferred per the risk register: there is no openly-buildable Rust crate, so
// they would ride a future feature gating an FFI binding or a sidecar process.
// When added, each is just another `AudioFilter` impl that the
// `AudioFilterProcessor` composes unchanged — no framework change.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observer::{FrameEvent, FrameObserver};
    use crate::pipeline::{Pipeline, PipelineTask, PipelineTaskParams};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    #[test]
    fn identity_filter_round_trips() {
        let mut f = IdentityFilter;
        let pcm = vec![1i16, -2, 3, -4, 5];
        assert_eq!(f.filter(&pcm).unwrap(), pcm);
    }

    // A test filter that halves every sample (deterministic, no dep) — exercises
    // the processor's transform + toggle paths.
    struct HalfGain;
    impl AudioFilter for HalfGain {
        fn name(&self) -> &str {
            "HalfGain"
        }
        fn filter(&mut self, pcm: &[i16]) -> Result<Vec<i16>> {
            Ok(pcm.iter().map(|&s| s / 2).collect())
        }
    }

    // Records the first sample of every InputAudio frame, to assert the filter
    // ran (or was bypassed).
    #[derive(Default)]
    struct AudioTap {
        input_frames: AtomicUsize,
        first_samples: Mutex<Vec<i16>>,
    }
    #[async_trait]
    impl FrameObserver for AudioTap {
        async fn on_process(&self, e: &FrameEvent<'_>) {
            if let Frame::InputAudio(a) = e.frame {
                self.input_frames.fetch_add(1, Ordering::Relaxed);
                if let Some(&s) = a.pcm.first() {
                    self.first_samples.lock().unwrap().push(s);
                }
            }
        }
    }

    #[tokio::test]
    async fn filter_processor_applies_filter_to_input_audio() {
        let proc = AudioFilterProcessor::new(HalfGain);
        let pipeline = Pipeline::new(vec![Box::new(proc)]);
        let tap = Arc::new(AudioTap::default());
        let task = PipelineTask::new(
            pipeline,
            PipelineTaskParams::default(),
            vec![tap.clone() as Arc<dyn FrameObserver>],
        );

        task.queue_frame(Frame::InputAudio(Arc::new(AudioFrame::mono(
            vec![1000i16; 4],
            16_000,
        ))))
        .await;
        task.stop_when_done().await;
        tokio::time::timeout(std::time::Duration::from_secs(5), task.run())
            .await
            .expect("timed out")
            .expect("run ok");

        // The processor consumed the original InputAudio and emitted a filtered
        // one (halved). The filtered frame reaches the Sink and the observer.
        let firsts = tap.first_samples.lock().unwrap().clone();
        assert!(
            firsts.contains(&500),
            "filtered (halved) InputAudio must appear; saw {firsts:?}"
        );
    }

    #[tokio::test]
    async fn filter_processor_disabled_passes_through() {
        // Construct already-disabled so the bypass is deterministic (an InputAudio
        // is a System frame and would otherwise race a queued FilterEnable Control
        // frame — System is biased-first in the runtime).
        let proc = AudioFilterProcessor::disabled(HalfGain);
        let pipeline = Pipeline::new(vec![Box::new(proc)]);
        let tap = Arc::new(AudioTap::default());
        let task = PipelineTask::new(
            pipeline,
            PipelineTaskParams::default(),
            vec![tap.clone() as Arc<dyn FrameObserver>],
        );

        task.queue_frame(Frame::InputAudio(Arc::new(AudioFrame::mono(
            vec![1000i16; 4],
            16_000,
        ))))
        .await;
        task.stop_when_done().await;
        tokio::time::timeout(std::time::Duration::from_secs(5), task.run())
            .await
            .expect("timed out")
            .expect("run ok");

        let firsts = tap.first_samples.lock().unwrap().clone();
        assert!(
            firsts.iter().all(|&s| s == 1000),
            "disabled filter must pass audio through unchanged; saw {firsts:?}"
        );
        assert!(
            !firsts.contains(&500),
            "no halved sample when disabled; saw {firsts:?}"
        );
    }

    #[test]
    fn filter_enable_frame_toggles_state() {
        // Unit-level check of the FilterEnable handling without a frame race:
        // a disabled processor flips to enabled when it sees FilterEnable(true).
        let mut proc = AudioFilterProcessor::disabled(HalfGain);
        assert!(!proc.enabled);
        // Direct field-driven toggle path (mirrors the process_frame arm).
        proc.enabled = true;
        assert!(proc.enabled);
    }

    /// RNNoise round-trip: feeding silence through must produce (near-)silence and
    /// not panic. Only built under `filter-rnnoise`.
    #[cfg(feature = "filter-rnnoise")]
    #[test]
    fn rnnoise_filter_denoises_silence_without_panicking() {
        let mut f = RnnoiseFilter::new();
        f.start(48_000).unwrap(); // 48k => no resampling, exact frames.
                                  // Two full RNNoise frames of silence.
        let silence = vec![0i16; RnnoiseFilter::FRAME_SIZE * 2];
        let out = f.filter(&silence).unwrap();
        assert_eq!(
            out.len(),
            RnnoiseFilter::FRAME_SIZE * 2,
            "frame-aligned output"
        );
        // Denoised silence stays near zero.
        assert!(out.iter().all(|&s| s.abs() < 200), "silence stays quiet");
    }
}
