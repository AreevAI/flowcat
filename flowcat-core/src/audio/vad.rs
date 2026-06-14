// SPDX-License-Identifier: Apache-2.0
//
//! Voice-activity-detector impls + the [`VadProcessor`].
//!
//! The [`VadAnalyzer`](crate::service::VadAnalyzer) trait is frozen in
//! [`crate::service`]; this module fills the impls:
//! - [`SileroVad`] — ONNX Silero VAD, compiled only under the `vad-ort` feature
//!   (uses `ort`); the heavy ONNX runtime stays opt-in.
//! - [`MockVad`] — a deterministic, scriptable VAD, always available for tests and
//!   the cascaded-pipeline fixture (no `ort` dep).
//!
//! Both analyzers reuse [`VadStateMachine`], a faithful port of pipecat's
//! `VADAnalyzer._run_analyzer` (`pipecat/audio/vad/vad_analyzer.py`): a per-window
//! confidence + volume gate feeding a QUIET→STARTING→SPEAKING→STOPPING hysteresis
//! with `start_secs`/`stop_secs` debounce. The analyzer only computes the raw
//! voice **confidence** per window; the state machine owns the timing/hysteresis,
//! exactly as in pipecat (the `voice_confidence` hook vs. the shared base loop).
//!
//! [`VadProcessor`] wraps a [`VadAnalyzer`] and emits the edge frames already in
//! the [`Frame`] enum: [`Frame::VadUserStartedSpeaking`] /
//! [`Frame::VadUserStoppedSpeaking`] (definitive VAD edges) plus the lifecycle
//! [`Frame::UserStartedSpeaking`] / [`Frame::UserStoppedSpeaking`] /
//! [`Frame::Interruption`] (barge-in when the bot is speaking).

use async_trait::async_trait;

use crate::error::Result;
use crate::processor::frame::{AudioFrame, Frame, StartParams, VadParams};
use crate::processor::{Envelope, FrameProcessor, Link, ProcessorSetup};
use crate::service::{VadAnalyzer, VadState};

/// Default VAD confidence threshold (pipecat `VAD_CONFIDENCE`).
pub const VAD_CONFIDENCE: f32 = 0.7;
/// Default voice-start debounce in seconds (pipecat `VAD_START_SECS`).
pub const VAD_START_SECS: f32 = 0.2;
/// Default voice-stop debounce in seconds (pipecat `VAD_STOP_SECS`).
pub const VAD_STOP_SECS: f32 = 0.2;
/// Default minimum normalized volume to count a window as speech
/// (pipecat `VAD_MIN_VOLUME`).
pub const VAD_MIN_VOLUME: f32 = 0.6;

impl Default for VadParams {
    fn default() -> Self {
        Self {
            confidence: VAD_CONFIDENCE,
            start_secs: VAD_START_SECS,
            stop_secs: VAD_STOP_SECS,
            min_volume: VAD_MIN_VOLUME,
        }
    }
}

/// The QUIET→STARTING→SPEAKING→STOPPING hysteresis shared by every VAD impl.
///
/// A faithful port of pipecat `VADAnalyzer._run_analyzer`: the analyzer supplies
/// the raw per-window voice **confidence** and a smoothed **volume**; this machine
/// gates `speaking = confidence >= threshold && volume >= min_volume` and applies
/// the debounce: STARTING must persist `start_frames` windows to become SPEAKING,
/// STOPPING must persist `stop_frames` windows to become QUIET. `start_frames`/
/// `stop_frames` are derived from `start_secs`/`stop_secs` and the window duration
/// (`window_samples / sample_rate`), exactly as pipecat rounds them.
#[derive(Debug, Clone)]
pub struct VadStateMachine {
    params: VadParams,
    sample_rate: u32,
    window_samples: usize,
    start_frames: u32,
    stop_frames: u32,
    starting_count: u32,
    stopping_count: u32,
    state: VadState,
}

impl VadStateMachine {
    /// Build a state machine for an analyzer producing `window_samples`-sample
    /// windows at `sample_rate`, with the given debounce/threshold `params`.
    pub fn new(sample_rate: u32, window_samples: usize, params: VadParams) -> Self {
        let mut m = Self {
            params,
            sample_rate,
            window_samples,
            start_frames: 1,
            stop_frames: 1,
            starting_count: 0,
            stopping_count: 0,
            state: VadState::Quiet,
        };
        m.recompute_debounce();
        m
    }

    fn recompute_debounce(&mut self) {
        // pipecat: vad_frames_per_sec = window_samples / sample_rate;
        //          start_frames = round(start_secs / vad_frames_per_sec)
        let window_secs = self.window_samples as f32 / self.sample_rate.max(1) as f32;
        let per = |secs: f32| -> u32 {
            if window_secs <= 0.0 {
                1
            } else {
                (secs / window_secs).round().max(1.0) as u32
            }
        };
        self.start_frames = per(self.params.start_secs);
        self.stop_frames = per(self.params.stop_secs);
    }

    /// Replace the params and recompute debounce, resetting the running state to
    /// QUIET (mirrors pipecat `set_params`).
    pub fn set_params(&mut self, params: VadParams) {
        self.params = params;
        self.starting_count = 0;
        self.stopping_count = 0;
        self.state = VadState::Quiet;
        self.recompute_debounce();
    }

    /// The current state without advancing.
    pub fn state(&self) -> VadState {
        self.state
    }

    /// Advance one analysis window. `confidence` is the model's voice probability
    /// for the window; `volume` is the window's normalized loudness in `[0, 1]`
    /// already computed/smoothed by the analyzer (pipecat does the exponential
    /// smoothing in the base analyzer before this gate). Returns the new state.
    pub fn advance(&mut self, confidence: f32, volume: f32) -> VadState {
        let speaking = confidence >= self.params.confidence && volume >= self.params.min_volume;

        if speaking {
            match self.state {
                VadState::Quiet => {
                    self.state = VadState::Starting;
                    self.starting_count = 1;
                }
                VadState::Starting => self.starting_count += 1,
                VadState::Stopping => {
                    self.state = VadState::Speaking;
                    self.stopping_count = 0;
                }
                VadState::Speaking => {}
            }
        } else {
            match self.state {
                VadState::Starting => {
                    self.state = VadState::Quiet;
                    self.starting_count = 0;
                }
                VadState::Speaking => {
                    self.state = VadState::Stopping;
                    self.stopping_count = 1;
                }
                VadState::Stopping => self.stopping_count += 1,
                VadState::Quiet => {}
            }
        }

        if self.state == VadState::Starting && self.starting_count >= self.start_frames {
            self.state = VadState::Speaking;
            self.starting_count = 0;
        }
        if self.state == VadState::Stopping && self.stopping_count >= self.stop_frames {
            self.state = VadState::Quiet;
            self.stopping_count = 0;
        }

        self.state
    }
}

/// Normalized loudness of a 16-bit PCM window in `[0, 1]`. A pure-Rust analogue of
/// pipecat's `calculate_audio_volume`/`normalize_value` (RMS in dBFS, mapped from
/// roughly `[-60, 0]` dB to `[0, 1]`) — no `pyloudnorm`/EBU-R128 dependency. Used
/// by both the mock and real VAD so the volume gate behaves consistently.
pub fn window_volume(pcm: &[i16]) -> f32 {
    if pcm.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = pcm
        .iter()
        .map(|&s| {
            let f = s as f64 / 32768.0;
            f * f
        })
        .sum();
    let rms = (sum_sq / pcm.len() as f64).sqrt();
    if rms <= 0.0 {
        return 0.0;
    }
    let db = 20.0 * rms.log10(); // dBFS, <= 0
                                 // Map [-60, 0] dBFS -> [0, 1], clamped.
    (((db + 60.0) / 60.0).clamp(0.0, 1.0)) as f32
}

/// Exponential volume smoother (pipecat `exp_smoothing`, factor 0.2). Used by the
/// real analyzer to debounce the volume gate; the mock skips it for determinism.
#[derive(Debug, Clone, Default)]
pub struct VolumeSmoother {
    prev: f32,
}

impl VolumeSmoother {
    /// Smooth `volume` against the running average (factor 0.2) and return it.
    pub fn smooth(&mut self, volume: f32) -> f32 {
        self.prev += 0.2 * (volume - self.prev);
        self.prev
    }
}

/// A deterministic, scriptable VAD for tests + the cascaded fixture (no ONNX).
///
/// It owns a [`VadStateMachine`] so it exercises the **real** debounce/hysteresis,
/// but instead of an ML model it reads the per-window voice confidence from a
/// caller-controlled source:
/// - by default, `confidence = 1.0` when the window's volume exceeds the
///   `min_volume` gate and `0.0` otherwise (i.e. loud window ⇒ speech), so feeding
///   loud PCM drives QUIET→SPEAKING and silence drives the reverse;
/// - or a fixed scripted confidence sequence via [`MockVad::with_script`] for
///   precise edge-timing tests.
pub struct MockVad {
    sm: VadStateMachine,
    window_samples: usize,
    script: Option<Vec<f32>>,
    script_pos: usize,
}

impl MockVad {
    /// A volume-driven mock at `sample_rate` with `window_samples`-sample windows
    /// and default params. Loud windows are treated as speech.
    pub fn new(sample_rate: u32, window_samples: usize) -> Self {
        Self {
            sm: VadStateMachine::new(sample_rate, window_samples, VadParams::default()),
            window_samples,
            script: None,
            script_pos: 0,
        }
    }

    /// Like [`MockVad::new`] with explicit params.
    pub fn with_params(sample_rate: u32, window_samples: usize, params: VadParams) -> Self {
        Self {
            sm: VadStateMachine::new(sample_rate, window_samples, params),
            window_samples,
            script: None,
            script_pos: 0,
        }
    }

    /// A mock whose per-window confidences come from `script` (the last value is
    /// repeated once exhausted), for precise debounce/edge tests. The volume gate
    /// is bypassed (the scripted confidence is taken as the speaking signal). Uses
    /// default [`VadParams`]; for a custom debounce use
    /// [`MockVad::with_script_params`].
    pub fn with_script(sample_rate: u32, window_samples: usize, script: Vec<f32>) -> Self {
        Self::with_script_params(sample_rate, window_samples, script, VadParams::default())
    }

    /// Like [`MockVad::with_script`] with explicit `params` (debounce/threshold).
    pub fn with_script_params(
        sample_rate: u32,
        window_samples: usize,
        script: Vec<f32>,
        params: VadParams,
    ) -> Self {
        Self {
            sm: VadStateMachine::new(sample_rate, window_samples, params),
            window_samples,
            script: Some(script),
            script_pos: 0,
        }
    }

    /// The window size (samples) this mock consumes per [`VadAnalyzer::analyze`].
    pub fn window_samples(&self) -> usize {
        self.window_samples
    }
}

impl VadAnalyzer for MockVad {
    fn sample_rate(&self) -> u32 {
        self.sm.sample_rate
    }

    fn analyze(&mut self, audio: &AudioFrame) -> VadState {
        // The mock is deterministic: it does NOT apply the volume exp-smoothing the
        // real analyzer does (that lag would make single-window fixture tests
        // brittle). It exercises the real state-machine hysteresis, which is the
        // fidelity point.
        let (confidence, gate_volume) = match &self.script {
            Some(script) => {
                let c = script
                    .get(self.script_pos)
                    .copied()
                    .or_else(|| script.last().copied())
                    .unwrap_or(0.0);
                self.script_pos += 1;
                // Bypass the volume gate for scripted runs: drive purely by script
                // (feed the threshold so the volume gate always passes).
                (c, self.sm.params.min_volume)
            }
            None => {
                // Loud window ⇒ confident speech; quiet window ⇒ no speech.
                let volume = window_volume(&audio.pcm);
                let c = if volume >= self.sm.params.min_volume {
                    1.0
                } else {
                    0.0
                };
                (c, volume)
            }
        };
        self.sm.advance(confidence, gate_volume)
    }

    fn set_params(&mut self, params: VadParams) {
        self.sm.set_params(params);
    }
}

/// A [`FrameProcessor`] that runs a [`VadAnalyzer`] over inbound audio and emits
/// the VAD edge + lifecycle frames.
///
/// On each [`Frame::InputAudio`] it buffers PCM into analyzer-sized windows, runs
/// the analyzer, and translates QUIET↔SPEAKING transitions into:
/// - rising edge → [`Frame::VadUserStartedSpeaking`] + [`Frame::UserStartedSpeaking`],
///   and (if the bot is speaking) a broadcast [`Frame::Interruption`] (barge-in);
/// - falling edge → [`Frame::VadUserStoppedSpeaking`] + [`Frame::UserStoppedSpeaking`].
///
/// The original `InputAudio` frame is **always forwarded** downstream (STT/turn
/// stages still need the raw audio). Tracks bot-speaking state from
/// [`Frame::BotStartedSpeaking`]/[`Frame::BotStoppedSpeaking`] so interruption is
/// only broadcast while the bot holds the floor (pipecat barge-in semantics).
pub struct VadProcessor<V: VadAnalyzer> {
    analyzer: V,
    window_samples: usize,
    buffer: Vec<i16>,
    sample_rate: u32,
    last_state: VadState,
    bot_speaking: bool,
    /// If true, emit `Interruption` on a rising edge while the bot speaks.
    interrupt_on_barge_in: bool,
}

impl<V: VadAnalyzer> VadProcessor<V> {
    /// Wrap `analyzer`, consuming `window_samples`-sample windows. Barge-in
    /// interruption is enabled by default.
    pub fn new(analyzer: V, window_samples: usize) -> Self {
        let sample_rate = analyzer.sample_rate();
        Self {
            analyzer,
            window_samples: window_samples.max(1),
            buffer: Vec::new(),
            sample_rate,
            last_state: VadState::Quiet,
            bot_speaking: false,
            interrupt_on_barge_in: true,
        }
    }

    /// Disable broadcasting `Interruption` on barge-in (the turn controller may own
    /// interruption via an [`InterruptionStrategy`](crate::audio::strategy::InterruptionStrategy)).
    pub fn without_barge_in_interruption(mut self) -> Self {
        self.interrupt_on_barge_in = false;
        self
    }

    /// Run the analyzer over every complete buffered window and push edge frames.
    async fn drain_windows(&mut self, link: &Link) {
        while self.buffer.len() >= self.window_samples {
            let window: Vec<i16> = self.buffer.drain(..self.window_samples).collect();
            let frame = AudioFrame::mono(window, self.sample_rate);
            let new_state = self.analyzer.analyze(&frame);

            let was_speaking = matches!(self.last_state, VadState::Speaking | VadState::Stopping);
            let is_speaking = matches!(new_state, VadState::Speaking);

            // Rising edge: Quiet/Starting/Stopping -> Speaking.
            if is_speaking && !was_speaking {
                let secs = self.window_samples as f32 / self.sample_rate.max(1) as f32;
                link.push_down(Frame::VadUserStartedSpeaking { start_secs: secs })
                    .await;
                link.push_down(Frame::UserStartedSpeaking).await;
                if self.bot_speaking && self.interrupt_on_barge_in {
                    link.broadcast(Frame::Interruption).await;
                }
            }
            // Falling edge: Speaking/Stopping -> Quiet.
            else if was_speaking && matches!(new_state, VadState::Quiet) {
                let secs = self.window_samples as f32 / self.sample_rate.max(1) as f32;
                link.push_down(Frame::VadUserStoppedSpeaking { stop_secs: secs })
                    .await;
                link.push_down(Frame::UserStoppedSpeaking).await;
            }

            self.last_state = new_state;
        }
    }
}

#[async_trait]
impl<V: VadAnalyzer + 'static> FrameProcessor for VadProcessor<V> {
    fn name(&self) -> &str {
        "VadProcessor"
    }

    async fn start(&mut self, _setup: &ProcessorSetup, params: &StartParams) -> Result<()> {
        self.sample_rate = params.audio_in_sample_rate;
        Ok(())
    }

    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        match &env.frame {
            Frame::BotStartedSpeaking => self.bot_speaking = true,
            Frame::BotStoppedSpeaking => self.bot_speaking = false,
            Frame::InputAudio(audio) => {
                self.buffer.extend_from_slice(&audio.pcm);
                self.drain_windows(link).await;
            }
            _ => {}
        }
        // Always forward the frame unchanged (STT/turn stages still need it).
        link.push(env.meta, env.frame, env.direction).await;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Silero ONNX impl (behind `vad-ort`).
// ---------------------------------------------------------------------------

/// Silero VAD over ONNX Runtime (`ort`), behind the `vad-ort` feature.
///
/// Mirrors pipecat's `SileroVADAnalyzer`/`SileroOnnxModel`
/// (`pipecat/audio/vad/silero.py`): it runs the Silero VAD ONNX graph per window
/// (512 samples at 16 kHz, 256 at 8 kHz), carrying the recurrent `state` +
/// `context` tensors between calls and periodically resetting them. The raw model
/// confidence then feeds the shared [`VadStateMachine`] for the
/// QUIET/STARTING/SPEAKING/STOPPING hysteresis — identical to the mock, so the
/// behaviour is consistent.
///
/// Requires the `silero_vad.onnx` model file (not vendored); construct with
/// [`SileroVad::from_model_path`]. Only 8 kHz and 16 kHz are supported.
#[cfg(feature = "vad-ort")]
pub struct SileroVad {
    sm: VadStateMachine,
    sample_rate: u32,
    window_samples: usize,
    session: ort::session::Session,
    // Recurrent state tensor (Silero v5: shape [2,1,128]).
    state: ndarray::Array3<f32>,
    context: Vec<f32>,
    context_size: usize,
    windows_since_reset: u32,
    reset_every_windows: u32,
    smoother: VolumeSmoother,
}

#[cfg(feature = "vad-ort")]
impl SileroVad {
    /// Load the Silero VAD ONNX model at `model_path` for `sample_rate`
    /// (must be 8000 or 16000) with default [`VadParams`].
    pub fn from_model_path(
        model_path: impl AsRef<std::path::Path>,
        sample_rate: u32,
    ) -> Result<Self> {
        Self::with_params(model_path, sample_rate, VadParams::default())
    }

    /// Load the model with explicit `params`.
    pub fn with_params(
        model_path: impl AsRef<std::path::Path>,
        sample_rate: u32,
        params: VadParams,
    ) -> Result<Self> {
        use crate::error::FlowcatError;
        if sample_rate != 8000 && sample_rate != 16000 {
            return Err(FlowcatError::Codec(format!(
                "Silero VAD sample rate must be 8000 or 16000 (got {sample_rate})"
            )));
        }
        let (window_samples, context_size) = if sample_rate == 16000 {
            (512usize, 64usize)
        } else {
            (256usize, 32usize)
        };

        let session = ort::session::Session::builder()
            .and_then(|mut b| b.commit_from_file(model_path.as_ref()))
            .map_err(|e| FlowcatError::Realtime(format!("silero load: {e}")))?;

        // Reset roughly every ~5s of audio (pipecat `_MODEL_RESET_STATES_TIME`).
        let window_secs = window_samples as f32 / sample_rate as f32;
        let reset_every_windows = (5.0 / window_secs).round().max(1.0) as u32;

        Ok(Self {
            sm: VadStateMachine::new(sample_rate, window_samples, params),
            sample_rate,
            window_samples,
            session,
            state: ndarray::Array3::<f32>::zeros((2, 1, 128)),
            context: vec![0.0; context_size],
            context_size,
            windows_since_reset: 0,
            reset_every_windows,
            smoother: VolumeSmoother::default(),
        })
    }

    /// The window size (samples) this analyzer consumes per [`VadAnalyzer::analyze`].
    pub fn window_samples(&self) -> usize {
        self.window_samples
    }

    fn reset_states(&mut self) {
        self.state = ndarray::Array3::<f32>::zeros((2, 1, 128));
        self.context = vec![0.0; self.context_size];
    }

    /// Run the ONNX graph for one window, returning the voice probability `[0,1]`.
    fn voice_confidence(&mut self, pcm: &[i16]) -> f32 {
        use ndarray::{Array1, Array2};

        // int16 -> float32 normalized, prepended with the carried context.
        let mut x: Vec<f32> = Vec::with_capacity(self.context_size + pcm.len());
        x.extend_from_slice(&self.context);
        x.extend(pcm.iter().map(|&s| s as f32 / 32768.0));

        let input_len = x.len();
        let input = match Array2::from_shape_vec((1, input_len), x.clone()) {
            Ok(a) => a,
            Err(_) => return 0.0,
        };
        let sr = Array1::from_vec(vec![self.sample_rate as i64]);

        let run = (|| -> std::result::Result<f32, Box<dyn std::error::Error>> {
            let input_val = ort::value::Tensor::from_array(input)?;
            let state_val = ort::value::Tensor::from_array(self.state.clone())?;
            let sr_val = ort::value::Tensor::from_array(sr)?;
            let outputs = self.session.run(ort::inputs![
                "input" => input_val,
                "state" => state_val,
                "sr" => sr_val,
            ])?;
            // Output order: [output, state]. Update recurrent state.
            let (_, prob) = outputs[0].try_extract_tensor::<f32>()?;
            let confidence = prob.first().copied().unwrap_or(0.0);
            let (_, new_state) = outputs[1].try_extract_tensor::<f32>()?;
            if new_state.len() == self.state.len() {
                self.state = ndarray::Array3::from_shape_vec((2, 1, 128), new_state.to_vec())?;
            }
            Ok(confidence)
        })();

        // Carry the last `context_size` samples for the next window.
        if input_len >= self.context_size {
            self.context = x[input_len - self.context_size..].to_vec();
        }

        match run {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("Silero VAD inference failed: {e}");
                0.0
            }
        }
    }
}

#[cfg(feature = "vad-ort")]
impl VadAnalyzer for SileroVad {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn analyze(&mut self, audio: &AudioFrame) -> VadState {
        // Periodically reset the recurrent state so memory/state doesn't drift
        // (pipecat resets every ~5s).
        self.windows_since_reset += 1;
        if self.windows_since_reset >= self.reset_every_windows {
            self.reset_states();
            self.windows_since_reset = 0;
        }
        let confidence = self.voice_confidence(&audio.pcm);
        let volume = self.smoother.smooth(window_volume(&audio.pcm));
        self.sm.advance(confidence, volume)
    }

    fn set_params(&mut self, params: VadParams) {
        self.sm.set_params(params);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observer::{FrameEvent, FrameObserver};
    use crate::pipeline::{Pipeline, PipelineTask, PipelineTaskParams};
    use crate::processor::frame::Direction;
    use crate::processor::Envelope;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    // A loud 16-bit window (near full-scale) -> high volume -> speech.
    fn loud(n: usize, rate: u32) -> AudioFrame {
        AudioFrame::mono(vec![20000i16; n], rate)
    }
    fn quiet(n: usize, rate: u32) -> AudioFrame {
        AudioFrame::mono(vec![0i16; n], rate)
    }

    #[test]
    fn state_machine_debounce_start_and_stop() {
        // 16kHz, 512-sample windows => 32ms/window. start_secs=0.064 => 2 windows;
        // stop_secs=0.064 => 2 windows.
        let params = VadParams {
            confidence: 0.5,
            start_secs: 0.064,
            stop_secs: 0.064,
            min_volume: 0.0,
        };
        let mut sm = VadStateMachine::new(16000, 512, params);
        assert_eq!(
            sm.advance(1.0, 1.0),
            VadState::Starting,
            "1st speech window"
        );
        assert_eq!(sm.advance(1.0, 1.0), VadState::Speaking, "2nd => debounced");
        assert_eq!(sm.advance(1.0, 1.0), VadState::Speaking);
        // Now silence: 1 window => Stopping, 2 => Quiet.
        assert_eq!(sm.advance(0.0, 0.0), VadState::Stopping);
        assert_eq!(sm.advance(0.0, 0.0), VadState::Quiet);
    }

    #[test]
    fn state_machine_brief_blip_does_not_reach_speaking() {
        let params = VadParams {
            confidence: 0.5,
            start_secs: 0.1, // ~3 windows @32ms
            stop_secs: 0.064,
            min_volume: 0.0,
        };
        let mut sm = VadStateMachine::new(16000, 512, params);
        assert_eq!(sm.advance(1.0, 1.0), VadState::Starting);
        // A single quiet window before the start debounce completes => back to Quiet.
        assert_eq!(sm.advance(0.0, 0.0), VadState::Quiet);
    }

    #[test]
    fn mock_vad_volume_driven_reaches_speaking_then_quiet() {
        let mut vad = MockVad::with_params(
            16000,
            512,
            VadParams {
                confidence: 0.5,
                start_secs: 0.032, // 1 window
                stop_secs: 0.032,  // 1 window
                min_volume: 0.3,
            },
        );
        assert_eq!(vad.analyze(&loud(512, 16000)), VadState::Speaking);
        assert_eq!(vad.analyze(&quiet(512, 16000)), VadState::Quiet);
    }

    #[test]
    fn mock_vad_scripted_confidence() {
        // start_secs/stop_secs = 1 window each (0.032s @ 8kHz/256 ≈ 1 window).
        let mut vad = MockVad::with_script_params(
            8000,
            256,
            vec![0.0, 1.0, 1.0, 0.0, 0.0],
            VadParams {
                confidence: 0.5,
                start_secs: 0.032,
                stop_secs: 0.032,
                min_volume: 0.0,
            },
        );
        let w = || AudioFrame::mono(vec![0i16; 256], 8000);
        // window0: conf 0 -> Quiet
        assert_eq!(vad.analyze(&w()), VadState::Quiet);
        // window1: conf 1 -> Starting -> Speaking (start debounce = 1)
        assert_eq!(vad.analyze(&w()), VadState::Speaking);
        // window2 stays Speaking
        vad.analyze(&w());
        // window3: conf 0 -> Stopping -> Quiet (stop debounce = 1)
        assert_eq!(vad.analyze(&w()), VadState::Quiet);
    }

    // An observer that counts the VAD edge frames it sees.
    #[derive(Default)]
    struct EdgeTap {
        started: AtomicUsize,
        stopped: AtomicUsize,
        interruptions: AtomicUsize,
        names: Mutex<Vec<&'static str>>,
    }
    #[async_trait]
    impl FrameObserver for EdgeTap {
        async fn on_process(&self, e: &FrameEvent<'_>) {
            self.names.lock().unwrap().push(e.frame.name());
            match e.frame {
                Frame::VadUserStartedSpeaking { .. } => {
                    self.started.fetch_add(1, Ordering::Relaxed);
                }
                Frame::VadUserStoppedSpeaking { .. } => {
                    self.stopped.fetch_add(1, Ordering::Relaxed);
                }
                Frame::Interruption => {
                    self.interruptions.fetch_add(1, Ordering::Relaxed);
                }
                _ => {}
            }
        }
    }

    #[tokio::test]
    async fn vad_processor_emits_start_and_stop_edges() {
        let vad = MockVad::with_params(
            16000,
            512,
            VadParams {
                confidence: 0.5,
                start_secs: 0.032,
                stop_secs: 0.032,
                min_volume: 0.3,
            },
        );
        let proc = VadProcessor::new(vad, 512);
        let pipeline = Pipeline::new(vec![Box::new(proc)]);
        let tap = Arc::new(EdgeTap::default());
        let task = PipelineTask::new(
            pipeline,
            PipelineTaskParams::default(),
            vec![tap.clone() as Arc<dyn FrameObserver>],
        );

        // Feed loud audio (speech) then quiet audio (silence).
        task.queue_frame(Frame::InputAudio(Arc::new(loud(512, 16000))))
            .await;
        task.queue_frame(Frame::InputAudio(Arc::new(loud(512, 16000))))
            .await;
        task.queue_frame(Frame::InputAudio(Arc::new(quiet(512, 16000))))
            .await;
        task.queue_frame(Frame::InputAudio(Arc::new(quiet(512, 16000))))
            .await;
        task.stop_when_done().await;

        tokio::time::timeout(std::time::Duration::from_secs(5), task.run())
            .await
            .expect("vad pipeline timed out")
            .expect("run ok");

        assert_eq!(
            tap.started.load(Ordering::Relaxed),
            1,
            "exactly one start edge; saw {:?}",
            tap.names.lock().unwrap()
        );
        assert_eq!(
            tap.stopped.load(Ordering::Relaxed),
            1,
            "exactly one stop edge"
        );
    }

    // Build a `Link` with capturing downstream + upstream channels so a processor's
    // `process_frame` can be unit-tested directly (no PipelineTask lifecycle). This
    // isolates the broadcast assertion from the framework's interruption-vs-End drain
    // ordering (which is exercised by the pipeline-level edge tests above).
    fn capture_link() -> (
        Link,
        crate::processor::runtime::ProcessorRx,
        crate::processor::runtime::ProcessorRx,
    ) {
        use crate::processor::runtime::channel;
        let (down_tx, down_rx) = channel(Arc::from("down"), 64);
        let (up_tx, up_rx) = channel(Arc::from("up"), 64);
        let link = Link {
            next: Some(down_tx),
            prev: Some(up_tx),
            name: Arc::from("VadProcessor"),
            clock: crate::processor::Clock::new(),
            observer: None,
            enable_metrics: false,
            enable_usage_metrics: false,
            ttfb_start: Arc::new(std::sync::atomic::AtomicI64::new(0)),
            processing_start: Arc::new(std::sync::atomic::AtomicI64::new(0)),
        };
        (link, down_rx, up_rx)
    }

    // Drain everything currently queued on a receiver's system+normal channels.
    fn drain(rx: &mut crate::processor::runtime::ProcessorRx) -> Vec<&'static str> {
        let mut out = Vec::new();
        while let Ok(e) = rx.system.try_recv() {
            out.push(e.frame.name());
        }
        while let Ok(e) = rx.normal.try_recv() {
            out.push(e.frame.name());
        }
        out
    }

    #[tokio::test]
    async fn vad_processor_broadcasts_interruption_during_bot_speech() {
        let vad = MockVad::with_params(
            16000,
            512,
            VadParams {
                confidence: 0.5,
                start_secs: 0.032,
                stop_secs: 0.032,
                min_volume: 0.3,
            },
        );
        let mut proc = VadProcessor::new(vad, 512);
        let (link, mut down_rx, mut up_rx) = capture_link();

        // Bot is speaking; user barges in (loud audio) -> rising edge -> broadcast.
        proc.process_frame(
            Envelope::new(Frame::BotStartedSpeaking, Direction::Downstream),
            &link,
        )
        .await
        .unwrap();
        let _ = drain(&mut down_rx);
        proc.process_frame(
            Envelope::new(
                Frame::InputAudio(Arc::new(loud(512, 16000))),
                Direction::Downstream,
            ),
            &link,
        )
        .await
        .unwrap();

        let down = drain(&mut down_rx);
        let up = drain(&mut up_rx);
        assert!(
            down.contains(&"VadUserStartedSpeaking") && down.contains(&"UserStartedSpeaking"),
            "rising edge frames must be emitted downstream; saw {down:?}"
        );
        // broadcast fans the Interruption both directions (one copy each).
        assert!(
            down.iter().filter(|n| **n == "Interruption").count() == 1,
            "exactly one downstream Interruption; saw {down:?}"
        );
        assert!(
            up.iter().filter(|n| **n == "Interruption").count() == 1,
            "exactly one upstream Interruption; saw {up:?}"
        );
    }

    #[tokio::test]
    async fn vad_processor_no_interruption_when_bot_silent() {
        let vad = MockVad::with_params(
            16000,
            512,
            VadParams {
                confidence: 0.5,
                start_secs: 0.032,
                stop_secs: 0.032,
                min_volume: 0.3,
            },
        );
        let mut proc = VadProcessor::new(vad, 512);
        let (link, mut down_rx, mut up_rx) = capture_link();

        // No BotStartedSpeaking — a rising edge must NOT broadcast an interruption.
        proc.process_frame(
            Envelope::new(
                Frame::InputAudio(Arc::new(loud(512, 16000))),
                Direction::Downstream,
            ),
            &link,
        )
        .await
        .unwrap();

        let down = drain(&mut down_rx);
        let up = drain(&mut up_rx);
        assert!(
            down.contains(&"UserStartedSpeaking"),
            "rising edge still fires; saw {down:?}"
        );
        assert!(
            !down.contains(&"Interruption") && !up.contains(&"Interruption"),
            "no interruption when the bot is not speaking; down={down:?} up={up:?}"
        );
    }

    /// Real-model smoke test — needs the Silero ONNX model. Ignored in CI
    /// (no model/keys); run with `--features vad-ort -- --ignored` after placing
    /// `silero_vad.onnx` at $SILERO_VAD_ONNX.
    #[cfg(feature = "vad-ort")]
    #[tokio::test]
    #[ignore = "needs Silero ONNX model at $SILERO_VAD_ONNX"]
    async fn silero_vad_smoke() {
        let path = std::env::var("SILERO_VAD_ONNX")
            .expect("set SILERO_VAD_ONNX to the silero_vad.onnx path");
        let mut vad = SileroVad::from_model_path(&path, 16000).expect("load model");
        // Feed a few windows of silence — must not panic and stays Quiet.
        for _ in 0..5 {
            let st = vad.analyze(&AudioFrame::mono(vec![0i16; 512], 16000));
            assert_eq!(st, VadState::Quiet);
        }
    }
}
