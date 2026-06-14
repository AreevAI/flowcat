// SPDX-License-Identifier: Apache-2.0
//
//! Service-processor trait *signatures* the cascaded path builds against
//! (PROCESSOR-DESIGN §6.1) + no-op mock impls + the audio-intelligence traits.
//!
//! These traits are **frozen here** so providers and the audio-intelligence layer
//! build against a stable surface. The core ships only the trait + a mock impl of
//! each, and the mock cascaded pipeline fixture test (mock-STT → mock-LLM →
//! mock-TTS through a real [`Pipeline`](crate::pipeline::Pipeline)) that provider
//! crates extend.

pub mod adapters;

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{self, BoxStream};
use futures::StreamExt;

use crate::error::Result;
use crate::processor::frame::{AudioFrame, Frame, LlmContext, StartParams, TurnParams, VadParams};
use crate::types::{RealtimeEvent, RealtimeSetup, ToolDecl};

/// Connection-time setup for a [`RealtimeLlmService`] — the realtime S2S contract
/// (today's [`crate::realtime::RealtimeLlm`], restated as the canonical service
/// trait the processor wraps). Aliased to the existing [`RealtimeSetup`].
pub type RealtimeServiceSetup = RealtimeSetup;

/// A tool declaration for the cascaded LLM service (alias to the existing
/// [`ToolDecl`] data shape so the realtime + cascaded paths agree).
pub type Tool = ToolDecl;

// ---------------------------------------------------------------------------
// Streaming service-processor traits (PROCESSOR-DESIGN §6.1).
// ---------------------------------------------------------------------------

/// Streaming speech→text. Consumes `InputAudio`, emits `InterimTranscription`
/// then final `Transcription`. Mirrors pipecat `STTService`.
#[async_trait]
pub trait SttService: Send {
    fn name(&self) -> &str;
    async fn start(&mut self, params: &StartParams) -> Result<()>;
    /// Feed one audio chunk; transcript frames are returned for the processor to
    /// forward downstream.
    async fn run_stt(&mut self, audio: Arc<AudioFrame>) -> Result<Vec<Frame>>;
    async fn set_muted(&mut self, muted: bool);
}

/// Streaming text→speech. Consumes `TtsSpeak`/`Text`, emits `TtsStarted`,
/// `TtsAudio`*, `TtsStopped`. Mirrors pipecat `TTSService`.
#[async_trait]
pub trait TtsService: Send {
    fn name(&self) -> &str;
    fn sample_rate(&self) -> u32;
    async fn start(&mut self, params: &StartParams) -> Result<()>;
    async fn run_tts(&mut self, text: &str) -> Result<Vec<Frame>>;
}

/// Context-driven LLM. Consumes `LlmContext`/`LlmRun`, emits `LlmResponseStart`,
/// `LlmText`*, optional `FunctionCallsStarted`, `LlmResponseEnd`.
#[async_trait]
pub trait LlmService: Send {
    fn name(&self) -> &str;
    async fn start(&mut self, params: &StartParams) -> Result<()>;
    async fn run_llm<'a>(&'a mut self, ctx: &'a LlmContext) -> Result<BoxStream<'a, Frame>>;
    fn set_tools(&mut self, tools: Vec<Tool>);
}

// ---------------------------------------------------------------------------
// Object-forwarding blanket impls.
//
// `build_cascaded_task`/`build_cascaded_pipeline` are generic over the three
// service traits (`St: SttService`, …). A host that picks a provider at runtime
// (a provider-string factory, e.g. the embedder's `cascaded_factory`) must unify
// the per-provider concrete types behind a trait object, so it builds
// `Box<dyn SttService>` and passes that as the generic argument. These blanket
// impls make `Box<dyn Trait>: Trait` hold — the standard "boxed trait object is
// itself the trait" idiom — so the generic builders accept the boxed services
// without any change to their signatures. Additive + behaviour-preserving.
// ---------------------------------------------------------------------------

#[async_trait]
impl SttService for Box<dyn SttService> {
    fn name(&self) -> &str {
        (**self).name()
    }
    async fn start(&mut self, params: &StartParams) -> Result<()> {
        (**self).start(params).await
    }
    async fn run_stt(&mut self, audio: Arc<AudioFrame>) -> Result<Vec<Frame>> {
        (**self).run_stt(audio).await
    }
    async fn set_muted(&mut self, muted: bool) {
        (**self).set_muted(muted).await
    }
}

#[async_trait]
impl TtsService for Box<dyn TtsService> {
    fn name(&self) -> &str {
        (**self).name()
    }
    fn sample_rate(&self) -> u32 {
        (**self).sample_rate()
    }
    async fn start(&mut self, params: &StartParams) -> Result<()> {
        (**self).start(params).await
    }
    async fn run_tts(&mut self, text: &str) -> Result<Vec<Frame>> {
        (**self).run_tts(text).await
    }
}

#[async_trait]
impl LlmService for Box<dyn LlmService> {
    fn name(&self) -> &str {
        (**self).name()
    }
    async fn start(&mut self, params: &StartParams) -> Result<()> {
        (**self).start(params).await
    }
    async fn run_llm<'a>(&'a mut self, ctx: &'a LlmContext) -> Result<BoxStream<'a, Frame>> {
        (**self).run_llm(ctx).await
    }
    fn set_tools(&mut self, tools: Vec<Tool>) {
        (**self).set_tools(tools)
    }
}

/// The realtime S2S contract (today's [`crate::realtime::RealtimeLlm`] — UNCHANGED
/// shape, restated as the canonical service trait the processor wraps).
#[async_trait]
pub trait RealtimeLlmService: Send {
    async fn connect(&mut self, setup: RealtimeServiceSetup) -> Result<()>;
    async fn send_audio(&mut self, chunk: Arc<AudioFrame>) -> Result<()>;
    async fn update_system(&mut self, prompt: String, tools: Vec<Tool>) -> Result<()>;
    async fn send_tool_result(&mut self, id: String, result: serde_json::Value) -> Result<()>;
    async fn next_event(&mut self) -> Option<RealtimeEvent>;

    /// The PCM sample rate (Hz) this model expects on its audio **input**. The s2s
    /// pipeline resamples caller audio to this before `send_audio` and advertises it
    /// in [`RealtimeSetup::input_sample_rate`]. Default 16 kHz; a provider that needs
    /// another rate (OpenAI Realtime requires ≥ 24 kHz) overrides this. Lives on the
    /// connector so the consuming app never has to know provider rates.
    fn input_sample_rate(&self) -> u32 {
        16_000
    }

    /// Trigger an initial **bot-first** turn so the agent greets before the caller
    /// speaks. Default no-op (caller speaks first); a provider that supports it
    /// overrides (e.g. OpenAI Realtime sends a `response.create`). The pipeline calls
    /// this through [`ServiceRealtimeAdapter`](crate::realtime::ServiceRealtimeAdapter)
    /// → [`RealtimeKickoff`](crate::realtime::RealtimeKickoff).
    async fn kickoff(&mut self) -> Result<()> {
        Ok(())
    }

    /// Readiness notify for the **lock-free** event path (mirrors
    /// [`RealtimeLlm::event_notify`](crate::realtime::RealtimeLlm::event_notify)).
    /// `None` (default) → the pipeline falls back to the blocking `next_event`, which
    /// holds the session lock across the idle wait between bot turns and **starves
    /// `send_audio`** (caller audio stalls after the greeting). A connector with an
    /// internal event channel SHOULD return a `Notify` it fires on each event so the
    /// pipeline can await readiness without the lock.
    fn event_notify(&self) -> Option<Arc<tokio::sync::Notify>> {
        None
    }

    /// Non-blocking sibling of [`next_event`](Self::next_event) (mirrors
    /// [`RealtimeLlm::poll_event`](crate::realtime::RealtimeLlm::poll_event)). The
    /// default blocks; a connector that returns a real [`event_notify`](Self::event_notify)
    /// MUST override this to be genuinely non-blocking (e.g. a channel `try_recv`).
    async fn poll_event(&mut self) -> crate::realtime::PollEvent {
        crate::realtime::PollEvent::Ready(self.next_event().await)
    }
}

// ---------------------------------------------------------------------------
// Audio-intelligence traits — signatures only (PROCESSOR-DESIGN §6.1).
// ---------------------------------------------------------------------------

/// VAD classification of a frame of audio. Mirrors pipecat `VADState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadState {
    Quiet,
    Starting,
    Speaking,
    Stopping,
}

/// An end-of-turn prediction. Mirrors pipecat's turn-analyzer output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TurnPrediction {
    pub is_complete: bool,
    pub probability: f32,
}

/// Voice-activity detector. Mirrors pipecat `VADAnalyzer`. Silero (ONNX/`ort`) is
/// the reference impl (behind the `vad-ort` feature — the *trait* is always here).
pub trait VadAnalyzer: Send {
    fn sample_rate(&self) -> u32;
    /// Classify a frame of audio: Quiet / Starting / Speaking / Stopping.
    fn analyze(&mut self, audio: &AudioFrame) -> VadState;
    fn set_params(&mut self, params: VadParams);
}

/// End-of-turn / semantic-completion analyzer. Mirrors pipecat `BaseTurnAnalyzer`
/// (Smart-Turn v2/v3 is the reference impl).
pub trait TurnAnalyzer: Send {
    /// Given accumulated speech + the VAD edge, predict whether the turn is
    /// complete.
    fn analyze_turn(&mut self, audio: &AudioFrame, vad: VadState) -> TurnPrediction;
    fn set_params(&mut self, params: TurnParams);
}

// ---------------------------------------------------------------------------
// No-op / mock impls (PROCESSOR-DESIGN §6.1 — cascaded pipeline fixtures).
// ---------------------------------------------------------------------------

/// A mock STT: turns any audio chunk into a fixed final transcription. Provider
/// implementations replace this with real backends.
pub struct MockStt {
    pub transcript: String,
    pub user_id: Arc<str>,
}

impl MockStt {
    pub fn new(transcript: impl Into<String>) -> Self {
        Self {
            transcript: transcript.into(),
            user_id: Arc::from("user"),
        }
    }
}

#[async_trait]
impl SttService for MockStt {
    fn name(&self) -> &str {
        "MockStt"
    }
    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        Ok(())
    }
    async fn run_stt(&mut self, _audio: Arc<AudioFrame>) -> Result<Vec<Frame>> {
        Ok(vec![
            Frame::InterimTranscription {
                text: self.transcript.clone(),
                user_id: self.user_id.clone(),
                language: None,
            },
            Frame::Transcription {
                text: self.transcript.clone(),
                user_id: self.user_id.clone(),
                language: None,
                final_: true,
            },
        ])
    }
    async fn set_muted(&mut self, _muted: bool) {}
}

/// A mock LLM: echoes the last user transcription back as a single LLM text +
/// response framing. Provider implementations replace this with real backends.
pub struct MockLlm {
    pub prefix: String,
    pub tools: Vec<Tool>,
}

impl MockLlm {
    pub fn new(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
            tools: Vec::new(),
        }
    }
}

#[async_trait]
impl LlmService for MockLlm {
    fn name(&self) -> &str {
        "MockLlm"
    }
    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        Ok(())
    }
    async fn run_llm<'a>(&'a mut self, ctx: &'a LlmContext) -> Result<BoxStream<'a, Frame>> {
        // Echo the last message's "content" text if present, else a canned reply.
        let user_text = ctx
            .messages
            .last()
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("hello")
            .to_string();
        let reply = format!("{}{}", self.prefix, user_text);
        let frames = vec![
            Frame::LlmResponseStart,
            Frame::LlmText(reply),
            Frame::LlmResponseEnd,
        ];
        Ok(stream::iter(frames).boxed())
    }
    fn set_tools(&mut self, tools: Vec<Tool>) {
        self.tools = tools;
    }
}

/// A mock TTS: turns text into a `TtsStarted` + one `TtsAudio` chunk (silence
/// sized to the text length) + `TtsStopped`. Provider implementations replace this
/// with real backends.
pub struct MockTts {
    pub rate: u32,
}

impl MockTts {
    pub fn new(rate: u32) -> Self {
        Self { rate }
    }
}

#[async_trait]
impl TtsService for MockTts {
    fn name(&self) -> &str {
        "MockTts"
    }
    fn sample_rate(&self) -> u32 {
        self.rate
    }
    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        Ok(())
    }
    async fn run_tts(&mut self, text: &str) -> Result<Vec<Frame>> {
        // One sample per character — deterministic, no real audio model.
        let pcm = vec![0i16; text.len().max(1)];
        let audio = Arc::new(AudioFrame::mono(pcm, self.rate));
        Ok(vec![
            Frame::TtsStarted { context_id: None },
            Frame::TtsAudio {
                audio,
                context_id: None,
            },
            Frame::TtsStopped { context_id: None },
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // The service-processor adapters now live in the public `adapters` module
    // (promoted out of this test block — PROCESSOR-DESIGN §6.1);
    // the fixture below drives those promoted processors.
    use crate::observer::{FrameEvent, FrameObserver};
    use crate::pipeline::{Pipeline, PipelineTask, PipelineTaskParams};
    use crate::processor::frame::Frame;
    use crate::service::adapters::{LlmProcessor, SttProcessor, TtsProcessor};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    // An observer that records every frame name it saw processed.
    #[derive(Default)]
    struct Tap {
        names: Mutex<Vec<&'static str>>,
        saw_output_audio: AtomicBool,
    }
    #[async_trait]
    impl FrameObserver for Tap {
        async fn on_process(&self, e: &FrameEvent<'_>) {
            self.names.lock().unwrap().push(e.frame.name());
            if matches!(e.frame, Frame::OutputAudio(_)) {
                self.saw_output_audio.store(true, Ordering::Relaxed);
            }
        }
    }

    /// The §9 step-10 gate: mock-STT → mock-LLM → mock-TTS runs a turn end-to-end
    /// through a **real** `Pipeline`. Provider crates build against this fixture.
    #[tokio::test]
    async fn mock_cascaded_pipeline_runs_a_turn_end_to_end() {
        let stt = SttProcessor::new(MockStt::new("book a dentist appointment"));
        let llm = LlmProcessor::new(MockLlm::new("you said: "));
        let tts = TtsProcessor::new(MockTts::new(24_000));
        let pipeline = Pipeline::new(vec![Box::new(stt), Box::new(llm), Box::new(tts)]);

        let tap = Arc::new(Tap::default());
        let task = PipelineTask::new(
            pipeline,
            PipelineTaskParams::default(),
            vec![tap.clone() as Arc<dyn FrameObserver>],
        );

        // Drive a turn: feed one InputAudio chunk, then end.
        let audio = Arc::new(AudioFrame::mono(vec![1, 2, 3, 4], 16_000));
        task.queue_frame(Frame::InputAudio(audio)).await;
        task.stop_when_done().await;
        let _ = task.cancel_token();

        // Run with a generous timeout.
        tokio::time::timeout(std::time::Duration::from_secs(5), task.run())
            .await
            .expect("cascaded pipeline timed out")
            .expect("run ok");

        // The turn produced output audio (STT→LLM→TTS all fired).
        assert!(
            tap.saw_output_audio.load(Ordering::Relaxed),
            "mock cascaded pipeline must emit OutputAudio; saw {:?}",
            tap.names.lock().unwrap()
        );
        let names = tap.names.lock().unwrap().clone();
        assert!(
            names.contains(&"Transcription"),
            "STT must emit Transcription"
        );
        assert!(names.contains(&"LlmText"), "LLM must emit LlmText");
    }
}
