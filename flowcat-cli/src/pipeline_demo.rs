// SPDX-License-Identifier: Apache-2.0
//
//! Demo 1 — the in-process `FrameProcessor` pipeline.
//!
//! Network-free and credential-free. A synthetic source pumps a 1 s, 16 kHz sine
//! wave (20 ms / 320-sample frames) into the pipeline head via a
//! [`SourcePump`]. The chain is two trivial processors that show composition:
//!
//!   - [`EchoProcessor`] — an identity "echo": it counts each inbound
//!     [`Frame::InputAudio`] and re-emits it as a [`Frame::OutputAudio`] (the
//!     transport-output shape), passing everything else through unchanged.
//!   - [`Tap`] — a no-op pass-through that proves a second hop links cleanly.
//!
//! A [`FrameCounter`] [`FrameObserver`] sits beside the chain (not in it) and
//! tallies the `InputAudio`/`OutputAudio` frames it sees pushed, so we can report
//! frames-in vs frames-out without sitting on the hot path.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use flowcat_core::observer::{FrameObserver, FramePushEvent};
use flowcat_core::processor::frame::{AudioFrame, FrameKind};
use flowcat_core::{
    Envelope, Frame, FrameProcessor, Link, Pipeline, PipelineTask, PipelineTaskParams, Result,
    SourcePump,
};

/// 16 kHz mono, 20 ms frames → 320 samples/frame; ~1 s of audio.
const SAMPLE_RATE: u32 = 16_000;
const SAMPLES_PER_FRAME: usize = 320; // 20 ms @ 16 kHz
const NUM_FRAMES: usize = 50; // 50 × 20 ms = 1.0 s
const TONE_HZ: f32 = 440.0;

/// Build `NUM_FRAMES` of a 440 Hz sine wave, one [`AudioFrame`] per 20 ms.
fn sine_frames() -> Vec<AudioFrame> {
    let mut sample_idx: usize = 0;
    (0..NUM_FRAMES)
        .map(|_| {
            let pcm: Vec<i16> = (0..SAMPLES_PER_FRAME)
                .map(|_| {
                    let t = sample_idx as f32 / SAMPLE_RATE as f32;
                    sample_idx += 1;
                    let v = (2.0 * std::f32::consts::PI * TONE_HZ * t).sin();
                    (v * (i16::MAX as f32 * 0.6)) as i16
                })
                .collect();
            AudioFrame::mono(pcm, SAMPLE_RATE)
        })
        .collect()
}

/// Identity "echo" processor: counts inbound [`Frame::InputAudio`] frames and
/// re-emits each as a [`Frame::OutputAudio`] carrying the same PCM (no copy — the
/// payload is `Arc`-shared). All other frames forward unchanged.
struct EchoProcessor {
    /// Inbound audio frames seen by this processor.
    seen: Arc<AtomicU64>,
}

#[async_trait]
impl FrameProcessor for EchoProcessor {
    fn name(&self) -> &str {
        "Echo"
    }

    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        if let Frame::InputAudio(audio) = &env.frame {
            self.seen.fetch_add(1, Ordering::Relaxed);
            // Echo: the inbound capture becomes outbound playout, unchanged.
            link.push_down(Frame::OutputAudio(audio.clone())).await;
            return Ok(());
        }
        link.push(env.meta, env.frame, env.direction).await;
        Ok(())
    }
}

/// A trivial second hop — a pure pass-through, to show the chain composes. (It
/// relies entirely on the default `process_frame`, which forwards.)
struct Tap;

#[async_trait]
impl FrameProcessor for Tap {
    fn name(&self) -> &str {
        "Tap"
    }
}

/// A [`FrameObserver`] that tallies the audio frames flowing through the pipeline,
/// without sitting in the chain. The push hook fires once per *hop*, so to count
/// whole-pipeline traversals it gates on the hop endpoints: `InputAudio` entering
/// the user chain (pushed out of the internal `Source`) is "in", and `OutputAudio`
/// arriving at the terminal `Sink` is "out".
#[derive(Default)]
pub(crate) struct FrameCounter {
    frames_in: AtomicU64,
    frames_out: AtomicU64,
}

impl FrameCounter {
    pub(crate) fn frames_in(&self) -> u64 {
        self.frames_in.load(Ordering::Relaxed)
    }
    pub(crate) fn frames_out(&self) -> u64 {
        self.frames_out.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl FrameObserver for FrameCounter {
    async fn on_push(&self, e: &FramePushEvent<'_>) {
        match e.frame.kind() {
            // Count the input once, as it enters the user chain from the Source.
            FrameKind::InputAudio if e.source == "Source" => {
                self.frames_in.fetch_add(1, Ordering::Relaxed);
            }
            // Count the output once, as it lands at the terminal Sink.
            FrameKind::OutputAudio if e.destination == "Sink" => {
                self.frames_out.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
    }
}

/// What the demo reports.
pub(crate) struct PipelineSummary {
    pub(crate) frames_sourced: u64,
    pub(crate) echoed_in: u64,
    pub(crate) frames_in: u64,
    pub(crate) frames_out: u64,
}

impl PipelineSummary {
    pub(crate) fn print(&self, wall: Duration) {
        let audio_secs = self.frames_sourced as f64 * SAMPLES_PER_FRAME as f64 / SAMPLE_RATE as f64;
        println!("flowcat pipeline demo");
        println!("  source        : 440 Hz sine, {SAMPLE_RATE} Hz mono, {SAMPLES_PER_FRAME}-sample frames");
        println!(
            "  audio         : {} frames (~{:.2} s)",
            self.frames_sourced, audio_secs
        );
        println!("  chain         : Source -> Echo -> Tap -> Sink");
        println!("  frames in     : {} (InputAudio observed)", self.frames_in);
        println!(
            "  frames out    : {} (OutputAudio observed)",
            self.frames_out
        );
        println!("  echoed        : {} (counted in Echo)", self.echoed_in);
        println!("  wall time     : {:.3} ms", wall.as_secs_f64() * 1e3);
        let ok = self.frames_sourced == self.echoed_in
            && self.frames_in == self.frames_sourced
            && self.frames_out == self.frames_sourced;
        println!(
            "  result        : {}",
            if ok {
                "OK (in == out == sourced)"
            } else {
                "MISMATCH"
            }
        );
    }
}

/// Run the pipeline demo to completion and return the tallies.
pub(crate) async fn run() -> PipelineSummary {
    let frames = sine_frames();
    let frames_sourced = frames.len() as u64;

    let echoed = Arc::new(AtomicU64::new(0));
    let counter = Arc::new(FrameCounter::default());

    let pipeline = Pipeline::new(vec![
        Box::new(EchoProcessor {
            seen: echoed.clone(),
        }),
        Box::new(Tap),
    ]);

    let task = PipelineTask::new(
        pipeline,
        PipelineTaskParams::default(),
        vec![counter.clone()],
    );

    // Drive the head from a source reader task: emit each sine frame as InputAudio,
    // then request a graceful drain. `SourcePump` aborts the reader on drop after
    // `run()` returns. Feeding the head preserves the Start->ready handshake.
    let pump = SourcePump::spawn(task.queue_sender(), move |head| async move {
        for frame in frames {
            if head.emit(Frame::InputAudio(Arc::new(frame))).is_err() {
                return; // pipeline gone
            }
        }
        head.end();
    });

    // Bounded so a wiring bug can't hang CI.
    let _ = tokio::time::timeout(Duration::from_secs(10), task.run())
        .await
        .expect("pipeline task timed out");
    drop(pump);

    PipelineSummary {
        frames_sourced,
        echoed_in: echoed.load(Ordering::Relaxed),
        frames_in: counter.frames_in(),
        frames_out: counter.frames_out(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pipeline runs end-to-end: every sourced sine frame is echoed once and
    /// observed both as InputAudio (in) and OutputAudio (out).
    #[tokio::test]
    async fn pipeline_runs_and_counts() {
        let s = run().await;
        assert_eq!(s.frames_sourced, NUM_FRAMES as u64);
        assert_eq!(s.echoed_in, s.frames_sourced, "every input was echoed");
        assert_eq!(s.frames_in, s.frames_sourced, "every input observed");
        assert_eq!(s.frames_out, s.frames_sourced, "every echo observed");
    }

    #[test]
    fn sine_source_has_expected_shape() {
        let f = sine_frames();
        assert_eq!(f.len(), NUM_FRAMES);
        assert!(f.iter().all(|fr| fr.pcm.len() == SAMPLES_PER_FRAME));
        assert!(f.iter().all(|fr| fr.sample_rate == SAMPLE_RATE));
        // A non-trivial waveform: not all-zero.
        assert!(f.iter().any(|fr| fr.pcm.iter().any(|&s| s != 0)));
    }
}
