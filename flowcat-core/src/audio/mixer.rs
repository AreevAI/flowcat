// SPDX-License-Identifier: Apache-2.0
//
//! Output-leg audio mixers.
//!
//! Mirrors pipecat's `pipecat/audio/mixers/`: a mixer blends the bot's
//! TTS/realtime output with a secondary source (hold music, soundboard) before it
//! reaches the transport-out leg. This module defines:
//! - [`AudioMixer`] — the pure-DSP mixer trait (`mix(audio) -> audio`), the Rust
//!   analogue of pipecat's `BaseAudioMixer`;
//! - [`SilenceMixer`] — a pass-through (pipecat `SilenceAudioMixer`), always
//!   available; used to keep continuous output streaming;
//! - [`SoundfileMixer`] — mixes a pre-loaded mono PCM source at a configurable
//!   volume, looping (pipecat `SoundfileMixer`, but taking decoded PCM in-memory
//!   rather than pulling a `soundfile`/codec dep into core);
//! - [`MixerProcessor`] — a [`FrameProcessor`] that mixes `OutputAudio`/`TtsAudio`
//!   and is toggled live by [`Frame::MixerEnable`].

use async_trait::async_trait;

use crate::error::Result;
use crate::processor::frame::{AudioFrame, Frame, StartParams};
use crate::processor::{Envelope, FrameProcessor, Link, ProcessorSetup};

/// A pure-DSP audio mixer over 16-bit mono PCM. Mirrors pipecat `BaseAudioMixer`:
/// `start(sample_rate)` initializes the mixer for the output rate, `mix(audio)`
/// blends one chunk of transport audio with the mixer's own source.
///
/// Returns a chunk of the same length as the input (per-sample blend). Must be
/// deterministic (fixture-testable) and non-blocking.
pub trait AudioMixer: Send {
    /// A stable name for tracing/observers.
    fn name(&self) -> &str;
    /// Initialize for the output `sample_rate` (Hz). Default: no-op.
    fn start(&mut self, _sample_rate: u32) -> Result<()> {
        Ok(())
    }
    /// Blend the mixer's source into one chunk of transport `pcm`, returning the
    /// mixed chunk (same length).
    fn mix(&mut self, pcm: &[i16]) -> Vec<i16>;
}

/// A pass-through mixer: returns transport audio unchanged. Mirrors pipecat
/// `SilenceAudioMixer` — used so the output transport can stream continuously
/// (mixing silence) even when no TTS audio is flowing. Always available (no dep).
#[derive(Debug, Default, Clone)]
pub struct SilenceMixer;

impl AudioMixer for SilenceMixer {
    fn name(&self) -> &str {
        "SilenceMixer"
    }
    fn mix(&mut self, pcm: &[i16]) -> Vec<i16> {
        pcm.to_vec()
    }
}

/// Mixes a pre-loaded mono PCM source (hold music / soundboard) into the bot
/// audio at a configurable `volume`, looping the source. Mirrors pipecat
/// `SoundfileMixer` but takes **decoded** PCM in-memory (the caller decodes the
/// file with whatever codec it likes) so core pulls no `soundfile`/audio-decode
/// dependency.
///
/// The source must already be at the output sample rate (set via [`Self::start`]);
/// a rate mismatch is logged and mixing is disabled, mirroring pipecat.
#[derive(Debug, Clone)]
pub struct SoundfileMixer {
    source: Vec<i16>,
    source_rate: u32,
    pos: usize,
    volume: f32,
    looping: bool,
    sample_rate: u32,
    ready: bool,
}

impl SoundfileMixer {
    /// Build a mixer from decoded mono `source` PCM at `source_rate`, blended at
    /// `volume` (0.0–1.0), looping when `looping`.
    pub fn new(source: Vec<i16>, source_rate: u32, volume: f32, looping: bool) -> Self {
        Self {
            source,
            source_rate,
            pos: 0,
            volume: volume.clamp(0.0, 1.0),
            looping,
            sample_rate: source_rate,
            ready: false,
        }
    }

    /// Update the mixing volume (0.0–1.0) at runtime.
    pub fn set_volume(&mut self, volume: f32) {
        self.volume = volume.clamp(0.0, 1.0);
    }
}

impl AudioMixer for SoundfileMixer {
    fn name(&self) -> &str {
        "SoundfileMixer"
    }

    fn start(&mut self, sample_rate: u32) -> Result<()> {
        self.sample_rate = sample_rate;
        // The source must match the output rate (pipecat logs + disables on
        // mismatch rather than resampling the loop on the hot path).
        self.ready = self.source_rate == sample_rate && !self.source.is_empty();
        if !self.ready && !self.source.is_empty() {
            tracing::warn!(
                "SoundfileMixer: source rate {} != output rate {} — mixing disabled",
                self.source_rate,
                sample_rate
            );
        }
        Ok(())
    }

    fn mix(&mut self, pcm: &[i16]) -> Vec<i16> {
        if !self.ready || self.volume == 0.0 {
            return pcm.to_vec();
        }
        let mut out = Vec::with_capacity(pcm.len());
        for &s in pcm {
            if self.pos >= self.source.len() {
                if self.looping {
                    self.pos = 0;
                } else {
                    out.push(s);
                    continue;
                }
            }
            let mixed = s as f32 + self.source[self.pos] as f32 * self.volume;
            out.push(mixed.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16);
            self.pos += 1;
        }
        out
    }
}

/// A [`FrameProcessor`] that mixes a secondary [`AudioMixer`] source into the
/// bot's output audio.
///
/// On [`Frame::OutputAudio`]/[`Frame::TtsAudio`] it blends the mixer source and
/// forwards the mixed audio; on [`Frame::MixerEnable`] it toggles mixing live
/// (pipecat's `MixerEnableFrame`). Sits **after** TTS / before the transport-out
/// sink. All other frames pass through.
pub struct MixerProcessor<M: AudioMixer> {
    mixer: M,
    enabled: bool,
    sample_rate: u32,
}

impl<M: AudioMixer> MixerProcessor<M> {
    /// Wrap `mixer`, enabled by default.
    pub fn new(mixer: M) -> Self {
        Self {
            mixer,
            enabled: true,
            sample_rate: 24_000,
        }
    }

    /// Wrap `mixer`, initially disabled (pass-through until a
    /// [`Frame::MixerEnable`]`(true)` arrives).
    pub fn disabled(mixer: M) -> Self {
        Self {
            mixer,
            enabled: false,
            sample_rate: 24_000,
        }
    }
}

#[async_trait]
impl<M: AudioMixer + 'static> FrameProcessor for MixerProcessor<M> {
    fn name(&self) -> &str {
        self.mixer.name()
    }

    async fn start(&mut self, _setup: &ProcessorSetup, params: &StartParams) -> Result<()> {
        self.sample_rate = params.audio_out_sample_rate;
        self.mixer.start(self.sample_rate)
    }

    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        match &env.frame {
            Frame::MixerEnable(on) => {
                self.enabled = *on;
                link.push(env.meta, env.frame, env.direction).await;
                return Ok(());
            }
            Frame::OutputAudio(audio) if self.enabled => {
                let mixed = self.mixer.mix(&audio.pcm);
                let out = AudioFrame::mono(mixed, audio.sample_rate);
                link.push_down(Frame::OutputAudio(std::sync::Arc::new(out)))
                    .await;
                return Ok(());
            }
            Frame::TtsAudio { audio, context_id } if self.enabled => {
                let mixed = self.mixer.mix(&audio.pcm);
                let out = AudioFrame::mono(mixed, audio.sample_rate);
                link.push_down(Frame::TtsAudio {
                    audio: std::sync::Arc::new(out),
                    context_id: context_id.clone(),
                })
                .await;
                return Ok(());
            }
            _ => {}
        }
        link.push(env.meta, env.frame, env.direction).await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observer::{FrameEvent, FrameObserver};
    use crate::pipeline::{Pipeline, PipelineTask, PipelineTaskParams};
    use std::sync::{Arc, Mutex};

    #[test]
    fn silence_mixer_passes_through() {
        let mut m = SilenceMixer;
        let pcm = vec![1i16, 2, 3];
        assert_eq!(m.mix(&pcm), pcm);
    }

    #[test]
    fn soundfile_mixer_blends_at_volume() {
        // Source is constant 1000; volume 0.5 => +500 per sample.
        let mut m = SoundfileMixer::new(vec![1000i16; 4], 24_000, 0.5, true);
        m.start(24_000).unwrap();
        let out = m.mix(&[2000i16; 4]);
        assert_eq!(out, vec![2500i16; 4], "2000 + 1000*0.5 = 2500");
    }

    #[test]
    fn soundfile_mixer_loops_source() {
        // Source shorter than the chunk: it must loop.
        let mut m = SoundfileMixer::new(vec![100i16, 200], 8_000, 1.0, true);
        m.start(8_000).unwrap();
        let out = m.mix(&[0i16; 4]);
        // 0+100, 0+200, loop -> 0+100, 0+200
        assert_eq!(out, vec![100, 200, 100, 200]);
    }

    #[test]
    fn soundfile_mixer_clamps_on_overflow() {
        let mut m = SoundfileMixer::new(vec![30000i16; 2], 8_000, 1.0, true);
        m.start(8_000).unwrap();
        let out = m.mix(&[30000i16; 2]);
        assert!(
            out.iter().all(|&s| s == i16::MAX),
            "60000 saturates to i16::MAX"
        );
    }

    #[test]
    fn soundfile_mixer_rate_mismatch_disables_mixing() {
        let mut m = SoundfileMixer::new(vec![1000i16; 4], 8_000, 0.5, true);
        m.start(24_000).unwrap(); // mismatch
                                  // Mixing disabled -> pass-through.
        assert_eq!(m.mix(&[2000i16; 4]), vec![2000i16; 4]);
    }

    // Records the first sample of each OutputAudio frame.
    #[derive(Default)]
    struct OutTap {
        first_samples: Mutex<Vec<i16>>,
    }
    #[async_trait]
    impl FrameObserver for OutTap {
        async fn on_process(&self, e: &FrameEvent<'_>) {
            if let Frame::OutputAudio(a) = e.frame {
                if let Some(&s) = a.pcm.first() {
                    self.first_samples.lock().unwrap().push(s);
                }
            }
        }
    }

    #[tokio::test]
    async fn mixer_processor_mixes_output_audio() {
        let mixer = SoundfileMixer::new(vec![1000i16; 8], 24_000, 1.0, true);
        let proc = MixerProcessor::new(mixer);
        let pipeline = Pipeline::new(vec![Box::new(proc)]);
        let tap = Arc::new(OutTap::default());
        let task = PipelineTask::new(
            pipeline,
            PipelineTaskParams::default(),
            vec![tap.clone() as Arc<dyn FrameObserver>],
        );

        task.queue_frame(Frame::OutputAudio(Arc::new(AudioFrame::mono(
            vec![2000i16; 4],
            24_000,
        ))))
        .await;
        task.stop_when_done().await;
        tokio::time::timeout(std::time::Duration::from_secs(5), task.run())
            .await
            .expect("timed out")
            .expect("run ok");

        let firsts = tap.first_samples.lock().unwrap().clone();
        assert!(
            firsts.contains(&3000),
            "mixed OutputAudio (2000+1000) must appear; saw {firsts:?}"
        );
    }

    #[tokio::test]
    async fn mixer_processor_toggle_bypasses_when_disabled() {
        let mixer = SoundfileMixer::new(vec![1000i16; 8], 24_000, 1.0, true);
        let proc = MixerProcessor::new(mixer);
        let pipeline = Pipeline::new(vec![Box::new(proc)]);
        let tap = Arc::new(OutTap::default());
        let task = PipelineTask::new(
            pipeline,
            PipelineTaskParams::default(),
            vec![tap.clone() as Arc<dyn FrameObserver>],
        );

        task.queue_frame(Frame::MixerEnable(false)).await;
        task.queue_frame(Frame::OutputAudio(Arc::new(AudioFrame::mono(
            vec![2000i16; 4],
            24_000,
        ))))
        .await;
        task.stop_when_done().await;
        tokio::time::timeout(std::time::Duration::from_secs(5), task.run())
            .await
            .expect("timed out")
            .expect("run ok");

        let firsts = tap.first_samples.lock().unwrap().clone();
        assert!(
            firsts.iter().all(|&s| s == 2000),
            "disabled mixer must pass audio through unchanged; saw {firsts:?}"
        );
    }
}
