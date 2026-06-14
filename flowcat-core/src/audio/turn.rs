// SPDX-License-Identifier: Apache-2.0
//
//! End-of-turn / semantic-completion analyzer impls + the [`TurnProcessor`].
//!
//! The [`TurnAnalyzer`](crate::service::TurnAnalyzer) trait is frozen in
//! [`crate::service`]; this module fills the impls:
//! - [`SmartTurn`] — Smart-Turn v2/v3 ONNX, compiled only under the `vad-ort`
//!   feature (uses `ort`).
//! - [`MockTurn`] — a deterministic turn predictor, always available for tests.
//!
//! Both share [`TurnSilenceTracker`], a port of pipecat `BaseSmartTurn`'s
//! silence-accumulation fallback (`pipecat/audio/turn/smart_turn/base_smart_turn.py`):
//! while VAD reports speech the silence counter resets; once VAD reports silence,
//! accumulated silence past `stop_secs` forces an end-of-turn (`COMPLETE`) even if
//! the ML model is absent/uncertain. [`MockTurn`] uses only that tracker (plus an
//! optional scripted prediction); [`SmartTurn`] runs the ONNX endpoint model on
//! the buffered speech segment and falls back to the tracker.
//!
//! [`TurnProcessor`] wraps a [`TurnAnalyzer`], tracks VAD edges, and on a
//! `VadUserStoppedSpeaking` edge emits a [`Frame::Metrics`] carrying a
//! [`MetricsData::TurnPrediction`] — the prediction the cascaded turn controller's
//! [`TurnAnalyzerStop`](crate::audio::strategy::TurnAnalyzerStop) strategy consumes.

use async_trait::async_trait;

use crate::error::Result;
use crate::processor::frame::{AudioFrame, Frame, StartParams, TurnParams};
use crate::processor::metrics::MetricsData;
use crate::processor::{Envelope, FrameProcessor, Link, ProcessorSetup};
use crate::service::{TurnAnalyzer, TurnPrediction, VadState};

/// Default max silence (seconds) before forcing end-of-turn (pipecat `STOP_SECS`).
pub const TURN_STOP_SECS: f32 = 3.0;
/// Default pre-speech audio to include (ms) (pipecat `PRE_SPEECH_MS`).
pub const TURN_PRE_SPEECH_MS: f32 = 500.0;
/// Default max analyzed segment duration (seconds) (pipecat `MAX_DURATION_SECONDS`).
pub const TURN_MAX_DURATION_SECS: f32 = 8.0;

impl Default for TurnParams {
    fn default() -> Self {
        Self {
            stop_secs: TURN_STOP_SECS,
            pre_speech_ms: TURN_PRE_SPEECH_MS,
            max_duration_secs: Some(TURN_MAX_DURATION_SECS),
        }
    }
}

/// Accumulates per-window silence (the pipecat `BaseSmartTurn` fallback): while
/// VAD reports speech, silence resets; once silent, silence past `stop_secs`
/// forces an end-of-turn. Pure, deterministic, no model.
#[derive(Debug, Clone)]
pub struct TurnSilenceTracker {
    stop_ms: f32,
    sample_rate: u32,
    speech_triggered: bool,
    silence_ms: f32,
}

impl TurnSilenceTracker {
    /// Build a tracker for `sample_rate` with `stop_secs` of allowed silence.
    pub fn new(sample_rate: u32, stop_secs: f32) -> Self {
        Self {
            stop_ms: stop_secs * 1000.0,
            sample_rate,
            speech_triggered: false,
            silence_ms: 0.0,
        }
    }

    /// Update `stop_secs`.
    pub fn set_stop_secs(&mut self, stop_secs: f32) {
        self.stop_ms = stop_secs * 1000.0;
    }

    /// Whether any speech has been observed in the current turn.
    pub fn speech_triggered(&self) -> bool {
        self.speech_triggered
    }

    /// Feed one window of `num_samples` samples flagged `is_speech` (from VAD).
    /// Returns `true` if accumulated silence now forces an end-of-turn.
    pub fn append(&mut self, num_samples: usize, is_speech: bool) -> bool {
        if is_speech {
            self.silence_ms = 0.0;
            self.speech_triggered = true;
            false
        } else if self.speech_triggered {
            let chunk_ms = num_samples as f32 / (self.sample_rate.max(1) as f32 / 1000.0);
            self.silence_ms += chunk_ms;
            self.silence_ms >= self.stop_ms
        } else {
            false
        }
    }

    /// Reset for the next turn.
    pub fn clear(&mut self) {
        self.speech_triggered = false;
        self.silence_ms = 0.0;
    }
}

/// A deterministic turn analyzer for tests (no ONNX).
///
/// Two modes:
/// - **silence mode** (default): mirrors the pipecat fallback — the turn is
///   `complete` once accumulated post-speech silence exceeds `stop_secs`;
/// - **scripted mode** ([`MockTurn::with_script`]): each `analyze_turn` returns the
///   next scripted `(is_complete, probability)` for precise tests.
pub struct MockTurn {
    tracker: TurnSilenceTracker,
    window_samples: usize,
    script: Option<Vec<(bool, f32)>>,
    script_pos: usize,
}

impl MockTurn {
    /// A silence-driven mock for `sample_rate`, consuming `window_samples`-sample
    /// windows, with default [`TurnParams`].
    pub fn new(sample_rate: u32, window_samples: usize) -> Self {
        Self {
            tracker: TurnSilenceTracker::new(sample_rate, TURN_STOP_SECS),
            window_samples,
            script: None,
            script_pos: 0,
        }
    }

    /// A silence-driven mock with an explicit `stop_secs`.
    pub fn with_stop_secs(sample_rate: u32, window_samples: usize, stop_secs: f32) -> Self {
        Self {
            tracker: TurnSilenceTracker::new(sample_rate, stop_secs),
            window_samples,
            script: None,
            script_pos: 0,
        }
    }

    /// A scripted mock returning successive `(is_complete, probability)` results
    /// (the last entry repeats once exhausted).
    pub fn with_script(sample_rate: u32, window_samples: usize, script: Vec<(bool, f32)>) -> Self {
        Self {
            tracker: TurnSilenceTracker::new(sample_rate, TURN_STOP_SECS),
            window_samples,
            script: Some(script),
            script_pos: 0,
        }
    }

    /// The window size (samples) this mock consumes per analysis.
    pub fn window_samples(&self) -> usize {
        self.window_samples
    }
}

impl TurnAnalyzer for MockTurn {
    fn analyze_turn(&mut self, audio: &AudioFrame, vad: VadState) -> TurnPrediction {
        if let Some(script) = &self.script {
            let (is_complete, probability) = script
                .get(self.script_pos)
                .copied()
                .or_else(|| script.last().copied())
                .unwrap_or((false, 0.0));
            self.script_pos += 1;
            return TurnPrediction {
                is_complete,
                probability,
            };
        }
        let is_speech = matches!(vad, VadState::Speaking | VadState::Starting);
        let complete = self.tracker.append(audio.len(), is_speech);
        if complete {
            self.tracker.clear();
        }
        TurnPrediction {
            is_complete: complete,
            // Deterministic confidence: 1.0 when the silence threshold forced it,
            // else a low value (the model would refine this).
            probability: if complete { 1.0 } else { 0.0 },
        }
    }

    fn set_params(&mut self, params: TurnParams) {
        self.tracker.set_stop_secs(params.stop_secs);
    }
}

/// A [`FrameProcessor`] that runs a [`TurnAnalyzer`] over the inbound audio and
/// emits an end-of-turn [`MetricsData::TurnPrediction`] on each
/// [`Frame::VadUserStoppedSpeaking`] edge.
///
/// It tracks the latest VAD state from
/// [`Frame::VadUserStartedSpeaking`]/[`Frame::VadUserStoppedSpeaking`] and feeds
/// each [`Frame::InputAudio`] window to the analyzer (so the silence/ML model sees
/// the whole stream). The original frames are always forwarded; the turn-stop
/// decision itself is owned by the
/// [`TurnAnalyzerStop`](crate::audio::strategy::TurnAnalyzerStop) strategy in the
/// cascaded turn controller, which consumes this metric.
pub struct TurnProcessor<T: TurnAnalyzer> {
    analyzer: T,
    window_samples: usize,
    buffer: Vec<i16>,
    sample_rate: u32,
    vad: VadState,
    /// The most recent prediction (so the VAD-stop edge can report it).
    last_prediction: TurnPrediction,
}

impl<T: TurnAnalyzer> TurnProcessor<T> {
    /// Wrap `analyzer`, feeding `window_samples`-sample windows.
    pub fn new(analyzer: T, sample_rate: u32, window_samples: usize) -> Self {
        Self {
            analyzer,
            window_samples: window_samples.max(1),
            buffer: Vec::new(),
            sample_rate,
            vad: VadState::Quiet,
            last_prediction: TurnPrediction {
                is_complete: false,
                probability: 0.0,
            },
        }
    }

    fn drain_windows(&mut self) {
        while self.buffer.len() >= self.window_samples {
            let window: Vec<i16> = self.buffer.drain(..self.window_samples).collect();
            let frame = AudioFrame::mono(window, self.sample_rate);
            self.last_prediction = self.analyzer.analyze_turn(&frame, self.vad);
        }
    }
}

#[async_trait]
impl<T: TurnAnalyzer + 'static> FrameProcessor for TurnProcessor<T> {
    fn name(&self) -> &str {
        "TurnProcessor"
    }

    async fn start(&mut self, _setup: &ProcessorSetup, params: &StartParams) -> Result<()> {
        self.sample_rate = params.audio_in_sample_rate;
        Ok(())
    }

    fn can_generate_metrics(&self) -> bool {
        true
    }

    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        match &env.frame {
            Frame::VadUserStartedSpeaking { .. } => self.vad = VadState::Speaking,
            Frame::VadUserStoppedSpeaking { .. } => {
                self.vad = VadState::Quiet;
                // Drain any remaining audio, then report the end-of-turn prediction.
                self.drain_windows();
                link.push_down(Frame::Metrics(vec![MetricsData::TurnPrediction {
                    processor: "TurnProcessor".to_string(),
                    is_complete: self.last_prediction.is_complete,
                    probability: self.last_prediction.probability,
                    e2e_processing_ms: 0.0,
                }]))
                .await;
            }
            Frame::InputAudio(audio) => {
                self.buffer.extend_from_slice(&audio.pcm);
                self.drain_windows();
            }
            _ => {}
        }
        link.push(env.meta, env.frame, env.direction).await;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Smart-Turn ONNX impl (behind `vad-ort`).
// ---------------------------------------------------------------------------

/// Smart-Turn end-of-turn analyzer over ONNX Runtime (`ort`), behind the
/// `vad-ort` feature.
///
/// Mirrors pipecat's `BaseSmartTurn` + `LocalSmartTurnV2/V3`
/// (`pipecat/audio/turn/smart_turn/`): it buffers the user's speech segment, runs
/// the Smart-Turn endpoint-classification ONNX graph on it when VAD reports the
/// user stopped, and predicts `is_complete` + `probability`. A silence-timeout
/// fallback ([`TurnSilenceTracker`]) forces completion if the model is uncertain
/// or absent — identical to the mock, so behaviour degrades gracefully.
///
/// Requires the Smart-Turn ONNX model (not vendored); construct with
/// [`SmartTurn::from_model_path`]. The exact input tensor name/shape depends on the
/// model revision; this impl normalizes a flat float32 segment as `input` and
/// reads the completion probability from output 0 (override via
/// [`SmartTurn::with_io_names`] if the model differs).
#[cfg(feature = "vad-ort")]
pub struct SmartTurn {
    tracker: TurnSilenceTracker,
    sample_rate: u32,
    window_samples: usize,
    max_samples: usize,
    session: ort::session::Session,
    segment: Vec<f32>,
    input_name: String,
    output_index: usize,
}

#[cfg(feature = "vad-ort")]
impl SmartTurn {
    /// Load the Smart-Turn ONNX model at `model_path` for `sample_rate` with
    /// default [`TurnParams`].
    pub fn from_model_path(
        model_path: impl AsRef<std::path::Path>,
        sample_rate: u32,
    ) -> Result<Self> {
        Self::with_params(model_path, sample_rate, TurnParams::default())
    }

    /// Load with explicit `params`.
    pub fn with_params(
        model_path: impl AsRef<std::path::Path>,
        sample_rate: u32,
        params: TurnParams,
    ) -> Result<Self> {
        use crate::error::FlowcatError;
        let session = ort::session::Session::builder()
            .and_then(|mut b| b.commit_from_file(model_path.as_ref()))
            .map_err(|e| FlowcatError::Realtime(format!("smart-turn load: {e}")))?;
        // ~16kHz model input; window_samples chosen to match the analysis grid.
        let window_samples = 512usize;
        let max_samples = (params.max_duration_secs.unwrap_or(TURN_MAX_DURATION_SECS)
            * sample_rate as f32) as usize;
        Ok(Self {
            tracker: TurnSilenceTracker::new(sample_rate, params.stop_secs),
            sample_rate,
            window_samples,
            max_samples,
            session,
            segment: Vec::new(),
            input_name: "input".to_string(),
            output_index: 0,
        })
    }

    /// Override the ONNX input tensor name + output index if the model revision
    /// differs from the default (`input` / output 0).
    pub fn with_io_names(mut self, input_name: impl Into<String>, output_index: usize) -> Self {
        self.input_name = input_name.into();
        self.output_index = output_index;
        self
    }

    /// The window size (samples) this analyzer consumes per analysis.
    pub fn window_samples(&self) -> usize {
        self.window_samples
    }

    /// Run the endpoint model on the accumulated segment, returning the
    /// completion probability `[0,1]`.
    fn predict_endpoint(&mut self) -> f32 {
        use ndarray::Array2;
        // Limit to the most recent `max_samples` (pipecat trims the segment tail).
        let seg: &[f32] = if self.segment.len() > self.max_samples {
            &self.segment[self.segment.len() - self.max_samples..]
        } else {
            &self.segment
        };
        if seg.is_empty() {
            return 0.0;
        }
        let input = match Array2::from_shape_vec((1, seg.len()), seg.to_vec()) {
            Ok(a) => a,
            Err(_) => return 0.0,
        };
        let input_name = self.input_name.clone();
        let output_index = self.output_index;
        let run = (|| -> std::result::Result<f32, Box<dyn std::error::Error>> {
            let val = ort::value::Tensor::from_array(input)?;
            let outputs = self.session.run(ort::inputs![input_name => val])?;
            let (_, out) = outputs[output_index].try_extract_tensor::<f32>()?;
            Ok(out.first().copied().unwrap_or(0.0))
        })();
        match run {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!("Smart-Turn inference failed: {e}");
                0.0
            }
        }
    }
}

#[cfg(feature = "vad-ort")]
impl TurnAnalyzer for SmartTurn {
    fn analyze_turn(&mut self, audio: &AudioFrame, vad: VadState) -> TurnPrediction {
        let is_speech = matches!(vad, VadState::Speaking | VadState::Starting);
        // Accumulate the speech segment (normalized float32).
        self.segment
            .extend(audio.pcm.iter().map(|&s| s as f32 / 32768.0));

        // Silence-timeout fallback (forces completion regardless of the model).
        if self.tracker.append(audio.len(), is_speech) {
            self.tracker.clear();
            self.segment.clear();
            return TurnPrediction {
                is_complete: true,
                probability: 1.0,
            };
        }

        // On the falling edge (user stopped), run the ONNX endpoint model.
        if !is_speech && self.tracker.speech_triggered() {
            let probability = self.predict_endpoint();
            let is_complete = probability >= 0.5;
            if is_complete {
                self.tracker.clear();
                self.segment.clear();
            }
            return TurnPrediction {
                is_complete,
                probability,
            };
        }

        TurnPrediction {
            is_complete: false,
            probability: 0.0,
        }
    }

    fn set_params(&mut self, params: TurnParams) {
        self.tracker.set_stop_secs(params.stop_secs);
        self.max_samples = (params.max_duration_secs.unwrap_or(TURN_MAX_DURATION_SECS)
            * self.sample_rate as f32) as usize;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observer::{FrameEvent, FrameObserver};
    use crate::pipeline::{Pipeline, PipelineTask, PipelineTaskParams};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    fn window(n: usize, rate: u32) -> AudioFrame {
        AudioFrame::mono(vec![0i16; n], rate)
    }

    #[test]
    fn silence_tracker_forces_complete_after_stop_secs() {
        // 16kHz, 0.1s stop. Each 1600-sample window = 100ms.
        let mut t = TurnSilenceTracker::new(16000, 0.1);
        // No speech yet => never completes on silence.
        assert!(!t.append(1600, false));
        // Speech, then silence accumulates.
        assert!(!t.append(1600, true));
        // 1 window of silence = 100ms >= 100ms => complete.
        assert!(t.append(1600, false));
    }

    #[test]
    fn mock_turn_silence_mode_completes_on_silence_timeout() {
        // stop_secs=0.1 => 1 window of 1600 samples @16kHz.
        let mut turn = MockTurn::with_stop_secs(16000, 1600, 0.1);
        // Speaking window: incomplete.
        let p = turn.analyze_turn(&window(1600, 16000), VadState::Speaking);
        assert!(!p.is_complete);
        // Silent window after speech: completes.
        let p = turn.analyze_turn(&window(1600, 16000), VadState::Quiet);
        assert!(p.is_complete);
        assert_eq!(p.probability, 1.0);
    }

    #[test]
    fn mock_turn_scripted_predictions() {
        let mut turn =
            MockTurn::with_script(16000, 512, vec![(false, 0.1), (false, 0.3), (true, 0.95)]);
        assert_eq!(
            turn.analyze_turn(&window(512, 16000), VadState::Speaking),
            TurnPrediction {
                is_complete: false,
                probability: 0.1
            }
        );
        turn.analyze_turn(&window(512, 16000), VadState::Speaking);
        let p = turn.analyze_turn(&window(512, 16000), VadState::Quiet);
        assert!(p.is_complete);
        assert!((p.probability - 0.95).abs() < 1e-6);
    }

    // Observer that records whether a TurnPrediction metric was emitted.
    #[derive(Default)]
    struct TurnTap {
        saw_prediction: AtomicBool,
        complete: AtomicBool,
        names: Mutex<Vec<&'static str>>,
    }
    #[async_trait]
    impl FrameObserver for TurnTap {
        async fn on_process(&self, e: &FrameEvent<'_>) {
            self.names.lock().unwrap().push(e.frame.name());
            if let Frame::Metrics(data) = e.frame {
                for d in data {
                    if let MetricsData::TurnPrediction { is_complete, .. } = d {
                        self.saw_prediction.store(true, Ordering::Relaxed);
                        if *is_complete {
                            self.complete.store(true, Ordering::Relaxed);
                        }
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn turn_processor_emits_prediction_on_vad_stop() {
        // 16kHz, 1600-sample window, stop after 0.1s of silence.
        let turn = MockTurn::with_stop_secs(16000, 1600, 0.1);
        let proc = TurnProcessor::new(turn, 16000, 1600);
        let pipeline = Pipeline::new(vec![Box::new(proc)]);
        let tap = Arc::new(TurnTap::default());
        let task = PipelineTask::new(
            pipeline,
            PipelineTaskParams::default(),
            vec![tap.clone() as Arc<dyn FrameObserver>],
        );

        // User starts speaking, feeds a speech window, then VAD says stopped.
        task.queue_frame(Frame::VadUserStartedSpeaking { start_secs: 0.2 })
            .await;
        task.queue_frame(Frame::InputAudio(Arc::new(window(1600, 16000))))
            .await;
        task.queue_frame(Frame::VadUserStoppedSpeaking { stop_secs: 0.2 })
            .await;
        task.stop_when_done().await;

        tokio::time::timeout(std::time::Duration::from_secs(5), task.run())
            .await
            .expect("turn pipeline timed out")
            .expect("run ok");

        assert!(
            tap.saw_prediction.load(Ordering::Relaxed),
            "TurnProcessor must emit a TurnPrediction on VAD stop; saw {:?}",
            tap.names.lock().unwrap()
        );
    }

    /// Real-model smoke test — needs the Smart-Turn ONNX model. Ignored in CI.
    #[cfg(feature = "vad-ort")]
    #[tokio::test]
    #[ignore = "needs Smart-Turn ONNX model at $SMART_TURN_ONNX"]
    async fn smart_turn_smoke() {
        let path = std::env::var("SMART_TURN_ONNX")
            .expect("set SMART_TURN_ONNX to the smart-turn .onnx path");
        let mut turn = SmartTurn::from_model_path(&path, 16000).expect("load model");
        // Feed a speech window then silence — must not panic.
        let _ = turn.analyze_turn(&window(512, 16000), VadState::Speaking);
        let _ = turn.analyze_turn(&window(512, 16000), VadState::Quiet);
    }
}
