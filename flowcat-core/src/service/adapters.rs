// SPDX-License-Identifier: Apache-2.0
//
//! Service-processor adapters — the public [`FrameProcessor`] wrappers that lift a
//! frozen [`SttService`]/[`LlmService`]/[`TtsService`] (in [`crate::service`]) into
//! a pipeline node.
//!
//! These were the tiny adapters previously living in `service/mod.rs`'s
//! `#[cfg(test)]` block; promoted here as public processors so the cascaded
//! builder ([`crate::pipeline::cascaded`]) and provider implementations compose
//! them without re-declaring them. The service **traits** stay frozen in
//! [`crate::service`]; this file only wraps them.
//!
//! ## Where the round-trip is driven (and why ordering is preserved)
//!
//! The framework rule ([`FrameProcessor::process_frame`]) is **"must not block"**:
//! a *long-lived, unsolicited* reader (a streaming STT WebSocket that emits
//! transcriptions with no triggering frame) belongs in an **internally-spawned task**
//! — exactly how `RealtimeServiceProcessor` (`pipeline::s2s`) spawns its
//! `next_event` reader. **That reader lives inside the provider impl** (in
//! `flowcat-services`): the frozen trait methods (`run_stt` → `Vec<Frame>`,
//! `run_llm` → a stream, `run_tts` → `Vec<Frame>`) are the *bounded, per-trigger*
//! request side that the streaming reader hands results to. So a real Deepgram
//! `run_stt` returns quickly (it forwards whatever its background WS reader has
//! decoded), and these adapters simply **`await` that bounded call in
//! `process_frame`**, pushing the produced frames downstream in order.
//!
//! Awaiting the bounded call here (rather than detaching it into a fresh task per
//! frame) is what keeps the pipeline correct: a detached task would race the
//! lifecycle frames — a `Frame::End` queued right after one `InputAudio` would drain
//! and tear down the downstream STT/LLM/TTS processors **before** the detached
//! task's `push_down` arrived, dropping the turn. The `.await` here yields to the
//! runtime (so the per-processor task loop stays responsive between frames) yet
//! guarantees the produced frames are enqueued downstream *before* this processor
//! returns and the next (possibly terminal) frame is handled — mirroring the s2s
//! realtime processor, which likewise `await`s its bounded `send_audio` inline and
//! only spawns the *unsolicited* event reader.
//!
//! The wrapped service is held behind a `tokio::sync::Mutex` (an async lock, safe
//! across `.await`; never a `std::sync::Mutex`) so the provider impl can share it
//! with its own internal reader task without a "std guard held across await" hazard.

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::json;
use tokio::sync::Mutex;

use crate::error::Result;
use crate::processor::frame::{Frame, LlmContext, ServiceKind, StartParams};
use crate::processor::metrics::MetricsData;
use crate::processor::{Envelope, FrameProcessor, Link, ProcessorSetup};
use crate::service::{LlmService, SttService, TtsService};

// ===========================================================================
// SttProcessor — wraps an SttService.
// ===========================================================================

/// Wraps an [`SttService`]: on `InputAudio`, run STT and forward its frames
/// downstream; on [`Frame::SttMute`] toggles the service's mute; everything else
/// passes through unchanged.
pub struct SttProcessor<S: SttService> {
    /// The wrapped STT service (shared so a provider impl can also drive its own
    /// internal streaming reader from the same handle).
    svc: Arc<Mutex<S>>,
    /// Audio seconds forwarded to STT since the last `SttUsage` flush. Flushed
    /// each ~1s as a `Frame::Metrics(SttUsage)` so the recorder can fold it into
    /// `usage_metrics` for per-minute billing.
    pending_audio_seconds: f64,
}

impl<S: SttService> SttProcessor<S> {
    /// Wrap `svc` as a pipeline processor.
    pub fn new(svc: S) -> Self {
        Self {
            svc: Arc::new(Mutex::new(svc)),
            pending_audio_seconds: 0.0,
        }
    }
}

#[async_trait]
impl<S: SttService + 'static> FrameProcessor for SttProcessor<S> {
    fn name(&self) -> &str {
        // `name()` is `&str`; the provider name lives behind the async mutex, so we
        // return a stable static name for tracing/observers.
        "Stt"
    }

    async fn start(&mut self, _s: &ProcessorSetup, p: &StartParams) -> Result<()> {
        self.svc.lock().await.start(p).await
    }

    fn can_generate_metrics(&self) -> bool {
        true
    }

    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        match &env.frame {
            Frame::InputAudio(audio) => {
                // Accrue the audio duration fed to STT (the billable basis for
                // per-minute streaming STT); flush as an `SttUsage` metric each ~1s.
                let channels = audio.num_channels.max(1) as f64;
                self.pending_audio_seconds +=
                    audio.pcm.len() as f64 / channels / audio.sample_rate.max(1) as f64;
                if self.pending_audio_seconds >= 1.0 {
                    link.push_down(Frame::Metrics(vec![MetricsData::SttUsage {
                        processor: "stt".to_string(),
                        audio_seconds: self.pending_audio_seconds,
                    }]))
                    .await;
                    self.pending_audio_seconds = 0.0;
                }
                // Bounded round-trip: the (streaming) provider impl returns whatever
                // its internal reader has decoded. Await it so the produced frames
                // are enqueued downstream in order before the next frame is handled.
                let frames = self.svc.lock().await.run_stt(audio.clone()).await;
                match frames {
                    Ok(frames) => {
                        for f in frames {
                            link.push_down(f).await;
                        }
                    }
                    Err(e) => link.push_error(format!("stt: {e}"), false).await,
                }
            }
            // End of a VAD-delimited utterance: give a batch service the chance to
            // transcribe its buffered tail now instead of waiting for its next
            // fixed window (which would split the turn). No-op for a streaming
            // service — `SttService::flush` defaults to returning nothing.
            Frame::UserStoppedSpeaking => {
                match self.svc.lock().await.flush().await {
                    Ok(frames) => {
                        for f in frames {
                            link.push_down(f).await;
                        }
                    }
                    Err(e) => link.push_error(format!("stt flush: {e}"), false).await,
                }
                link.push(env.meta, env.frame, env.direction).await;
            }
            // STT mute control (pipecat `STTMuteFrame`). Forward so a downstream
            // observer/UI still sees it.
            Frame::SttMute(muted) => {
                self.svc.lock().await.set_muted(*muted).await;
                link.push(env.meta, env.frame, env.direction).await;
            }
            _ => link.push(env.meta, env.frame, env.direction).await,
        }
        Ok(())
    }
}

// ===========================================================================
// LlmProcessor — wraps an LlmService, driven by the context aggregator.
// ===========================================================================

/// Wraps an [`LlmService`]: on a [`Frame::LlmContext`] (built by the user context
/// aggregator) it runs the LLM and forwards the streamed response frames
/// downstream. As a fallback it also runs on a bare final `Transcription` (wrapping
/// it in a single-message context) so the adapter works **without** an aggregator
/// in front (the direct `STT → LLM → TTS` fixture path). In the cascaded builder
/// the user aggregator *consumes* the final transcription and emits `LlmContext`
/// instead, so the LLM is triggered exactly once per turn (no double-run).
///
/// The streamed frames are pushed as they arrive (true token streaming); the whole
/// stream is consumed before `process_frame` returns so a following terminal frame
/// can't drain the downstream TTS before the response lands. Tools pushed via
/// [`Frame::UpdateSettings`] (target `Llm`) update the service's tool set.
pub struct LlmProcessor<L: LlmService> {
    /// The wrapped LLM service.
    svc: Arc<Mutex<L>>,
    /// Optional barge-in generation counter (see
    /// [`VadProcessor::with_interrupt_flag`](crate::audio::VadProcessor)):
    /// checked between streamed chunks so an in-flight completion stops pushing
    /// promptly when the user barges in. The frame-path `Interruption` cannot do
    /// this — `process_frame` is already running when it arrives.
    interrupt_flag: Option<Arc<std::sync::atomic::AtomicU64>>,
}

impl<L: LlmService> LlmProcessor<L> {
    /// Wrap `svc` as a pipeline processor.
    pub fn new(svc: L) -> Self {
        Self {
            svc: Arc::new(Mutex::new(svc)),
            interrupt_flag: None,
        }
    }

    /// Cancel an in-flight stream when `flag` is bumped (duplex barge-in).
    pub fn with_interrupt_flag(mut self, flag: Arc<std::sync::atomic::AtomicU64>) -> Self {
        self.interrupt_flag = Some(flag);
        self
    }

    /// Run the LLM over `ctx`, pushing each streamed frame downstream in order.
    async fn run(&self, ctx: &LlmContext, link: &Link) {
        tracing::debug!(messages = ctx.messages.len(), "cascaded LLM run");
        let mut guard = self.svc.lock().await;
        // `run_llm` returns a `BoxStream<'a, Frame>` borrowing the guard, so the
        // guard outlives the stream. Push each frame as it arrives (true streaming);
        // `link.push_down` doesn't touch the guard, so the borrow is fine.
        let mut pushed = 0usize;
        let start_gen = self
            .interrupt_flag
            .as_ref()
            .map(|f| f.load(std::sync::atomic::Ordering::SeqCst));
        let mut cancelled = false;
        let err = match guard.run_llm(ctx).await {
            Ok(mut stream) => {
                while let Some(f) = stream.next().await {
                    // Cooperative barge-in cancel: stop consuming (and pushing)
                    // as soon as the barge-in generation moves. Dropping the
                    // stream aborts the underlying request.
                    if let (Some(flag), Some(g0)) = (&self.interrupt_flag, start_gen) {
                        if flag.load(std::sync::atomic::Ordering::SeqCst) != g0 {
                            cancelled = true;
                            break;
                        }
                    }
                    pushed += 1;
                    link.push_down(f).await;
                }
                None
            }
            Err(e) => Some(format!("llm: {e}")),
        };
        if cancelled {
            tracing::debug!(pushed, "cascaded LLM stream cancelled by barge-in");
            // Close the response framing so downstream aggregators can't wedge
            // in an open in_response span.
            link.push_down(Frame::LlmResponseEnd).await;
        }
        drop(guard);
        match &err {
            Some(msg) => tracing::warn!(error = %msg, "cascaded LLM error"),
            None => tracing::debug!(frames = pushed, "cascaded LLM produced frames"),
        }
        if let Some(msg) = err {
            link.push_error(msg, false).await;
        }
    }
}

#[async_trait]
impl<L: LlmService + 'static> FrameProcessor for LlmProcessor<L> {
    fn name(&self) -> &str {
        "Llm"
    }

    async fn start(&mut self, _s: &ProcessorSetup, p: &StartParams) -> Result<()> {
        self.svc.lock().await.start(p).await
    }

    fn can_generate_metrics(&self) -> bool {
        true
    }

    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        match &env.frame {
            // The aggregated context to run (the cascaded path's single trigger).
            Frame::LlmContext(ctx) => {
                self.run(ctx, link).await;
            }
            // Fallback (no-aggregator fixture path): a bare final transcription —
            // wrap it in a single-message context so the adapter still works. The
            // cascaded user aggregator consumes the transcription before it reaches
            // here, so this never double-fires alongside `LlmContext`.
            Frame::Transcription {
                text, final_: true, ..
            } => {
                let ctx = LlmContext {
                    messages: vec![json!({"role": "user", "content": text})],
                    tools: vec![],
                };
                self.run(&ctx, link).await;
            }
            // Live tool-set update (pipecat `ServiceUpdateSettingsFrame` → Llm).
            Frame::UpdateSettings {
                target: ServiceKind::Llm,
                settings,
            } => {
                if let Some(tools) = settings.get("tools").and_then(|t| t.as_array()) {
                    let tools = tools
                        .iter()
                        .filter_map(|t| serde_json::from_value(t.clone()).ok())
                        .collect();
                    self.svc.lock().await.set_tools(tools);
                }
                link.push(env.meta, env.frame, env.direction).await;
            }
            _ => link.push(env.meta, env.frame, env.direction).await,
        }
        Ok(())
    }
}

// ===========================================================================
// TtsProcessor — wraps a TtsService.
// ===========================================================================

/// Wraps a [`TtsService`]: on `TtsSpeak`/`LlmText`, synthesize and forward audio
/// frames (mapping `TtsAudio` → `OutputAudio` so a transport-out sink can play it).
///
/// `TtsSpeak` is the aggregator-driven trigger (one full sentence/utterance at a
/// time); a bare `LlmText` is also accepted so the adapter works token-by-token
/// without an aggregator in front.
pub struct TtsProcessor<T: TtsService> {
    /// The wrapped TTS service.
    svc: Arc<Mutex<T>>,
}

impl<T: TtsService> TtsProcessor<T> {
    /// Wrap `svc` as a pipeline processor.
    pub fn new(svc: T) -> Self {
        Self {
            svc: Arc::new(Mutex::new(svc)),
        }
    }

    /// Synthesize `text`, mapping each `TtsAudio` to `OutputAudio` for the sink.
    async fn run(&self, text: &str, link: &Link) {
        tracing::debug!(chars = text.len(), "cascaded TTS run");
        let frames = self.svc.lock().await.run_tts(text).await;
        match frames {
            Ok(frames) => {
                tracing::debug!(frames = frames.len(), "cascaded TTS produced frames");
                // Report characters synthesized for cost accounting. Count Unicode
                // chars (not byte len) so multilingual text (Hindi/Telugu) bills
                // correctly. Emitted only on success (a failed synth bills nothing);
                // folded by the recorder into usage_metrics, priced per-1k-chars.
                link.push_down(Frame::Metrics(vec![MetricsData::TtsUsage {
                    processor: "tts".to_string(),
                    characters: text.chars().count() as u64,
                }]))
                .await;
                for f in frames {
                    let f = match f {
                        // Map TtsAudio → OutputAudio so a transport-out sink plays it.
                        Frame::TtsAudio { audio, .. } => Frame::OutputAudio(audio),
                        other => other,
                    };
                    link.push_down(f).await;
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "cascaded TTS error");
                link.push_error(format!("tts: {e}"), false).await;
            }
        }
    }
}

#[async_trait]
impl<T: TtsService + 'static> FrameProcessor for TtsProcessor<T> {
    fn name(&self) -> &str {
        "Tts"
    }

    async fn start(&mut self, _s: &ProcessorSetup, p: &StartParams) -> Result<()> {
        self.svc.lock().await.start(p).await
    }

    fn can_generate_metrics(&self) -> bool {
        true
    }

    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        match &env.frame {
            // The assistant aggregator emits one TtsSpeak per assembled utterance.
            Frame::TtsSpeak { text, .. } => {
                self.run(text, link).await;
            }
            // Fallback: a raw LLM text token with no aggregator in front.
            Frame::LlmText(text) => {
                self.run(text, link).await;
            }
            _ => link.push(env.meta, env.frame, env.direction).await,
        }
        Ok(())
    }
}

// ===========================================================================
// Tests — the barge-in seams the adapters own (offline, no provider).
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::observer::{FrameEvent, FrameObserver};
    use crate::pipeline::{Pipeline, PipelineTask, PipelineTaskParams};
    use crate::processor::frame::AudioFrame;
    use crate::service::Tool;
    use futures::stream::BoxStream;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex as StdMutex;
    use std::time::Duration;

    /// Records every frame name the pipeline processed, plus the `LlmText` payloads.
    #[derive(Default)]
    struct Tap {
        names: StdMutex<Vec<&'static str>>,
        llm_text: StdMutex<Vec<String>>,
        transcripts: StdMutex<Vec<String>>,
    }
    #[async_trait]
    impl FrameObserver for Tap {
        async fn on_process(&self, e: &FrameEvent<'_>) {
            self.names.lock().unwrap().push(e.frame.name());
            match &e.frame {
                // One entry per *distinct* emitted frame — `on_process` fires once
                // per processor that sees it, so dedupe on the internal Sink hop.
                Frame::LlmText(t) if e.processor == "Sink" => {
                    self.llm_text.lock().unwrap().push(t.clone())
                }
                Frame::Transcription { text, .. } if e.processor == "Sink" => {
                    self.transcripts.lock().unwrap().push(text.clone())
                }
                _ => {}
            }
        }
    }

    // ---- STT flush seam ---------------------------------------------------

    /// A batch-style STT: buffers audio and only transcribes on `flush()` — the
    /// shape `whisper_local` has, and the reason the seam exists.
    struct BufferingStt {
        buffered: usize,
    }
    #[async_trait]
    impl SttService for BufferingStt {
        fn name(&self) -> &str {
            "BufferingStt"
        }
        async fn start(&mut self, _p: &StartParams) -> Result<()> {
            Ok(())
        }
        async fn run_stt(&mut self, audio: Arc<AudioFrame>) -> Result<Vec<Frame>> {
            self.buffered += audio.pcm.len();
            Ok(vec![])
        }
        async fn flush(&mut self) -> Result<Vec<Frame>> {
            if self.buffered == 0 {
                return Ok(vec![]);
            }
            let n = std::mem::take(&mut self.buffered);
            Ok(vec![Frame::Transcription {
                text: format!("{n} samples"),
                user_id: Arc::from("user"),
                language: None,
                final_: true,
            }])
        }
        async fn set_muted(&mut self, _muted: bool) {}
    }

    /// `UserStoppedSpeaking` (a VAD falling edge) must reach the service as a
    /// `flush()` — otherwise a batch STT's last window sits in its buffer until the
    /// *next* turn's audio arrives, and two utterances merge into one late
    /// transcript. (The edge is a System frame and the transcription a Control one,
    /// so their *arrival* order downstream is set by the channel split, not here;
    /// what this pins is that the flush happens and its text is emitted.)
    #[tokio::test]
    async fn user_stopped_speaking_flushes_a_batch_stt_before_forwarding_the_edge() {
        let tap = Arc::new(Tap::default());
        let pipeline = Pipeline::new(vec![Box::new(SttProcessor::new(BufferingStt {
            buffered: 0,
        }))]);
        let task = PipelineTask::new(
            pipeline,
            PipelineTaskParams::default(),
            vec![tap.clone() as Arc<dyn FrameObserver>],
        );

        task.queue_frame(Frame::InputAudio(Arc::new(AudioFrame::mono(
            vec![1i16; 40],
            16_000,
        ))))
        .await;
        task.queue_frame(Frame::UserStoppedSpeaking).await;
        task.stop_when_done().await;
        tokio::time::timeout(Duration::from_secs(5), task.run())
            .await
            .expect("stt flush pipeline timed out")
            .expect("run ok");

        assert_eq!(
            tap.transcripts.lock().unwrap().clone(),
            vec!["40 samples".to_string()],
            "the buffered tail must be transcribed at the falling edge"
        );
        let names = tap.names.lock().unwrap().clone();
        assert!(
            names.contains(&"UserStoppedSpeaking"),
            "the edge must still be forwarded after the flush; saw {names:?}"
        );
    }

    /// The default `flush()` is a no-op, so a streaming service (which does its own
    /// endpointing) is unaffected by the new seam.
    #[tokio::test]
    async fn default_flush_is_a_no_op_for_a_streaming_service() {
        struct Streaming;
        #[async_trait]
        impl SttService for Streaming {
            fn name(&self) -> &str {
                "Streaming"
            }
            async fn start(&mut self, _p: &StartParams) -> Result<()> {
                Ok(())
            }
            async fn run_stt(&mut self, _a: Arc<AudioFrame>) -> Result<Vec<Frame>> {
                Ok(vec![])
            }
            async fn set_muted(&mut self, _muted: bool) {}
        }
        assert!(Streaming.flush().await.expect("flush ok").is_empty());
    }

    // ---- LLM cooperative barge-in cancel -----------------------------------

    /// An LLM whose stream raises the shared barge-in flag partway through, i.e.
    /// the user barges in while the completion is still streaming.
    struct BargeInMidStream {
        flag: Arc<AtomicU64>,
        chunks: usize,
        /// Zero-based index of the chunk at which the barge-in fires.
        at: usize,
    }
    #[async_trait]
    impl LlmService for BargeInMidStream {
        fn name(&self) -> &str {
            "BargeInMidStream"
        }
        async fn start(&mut self, _p: &StartParams) -> Result<()> {
            Ok(())
        }
        async fn run_llm<'a>(&'a mut self, _ctx: &'a LlmContext) -> Result<BoxStream<'a, Frame>> {
            let flag = self.flag.clone();
            let (chunks, at) = (self.chunks, self.at);
            Ok(Box::pin(futures::stream::unfold(0usize, move |i| {
                let flag = flag.clone();
                async move {
                    if i >= chunks {
                        return None;
                    }
                    if i == at {
                        flag.fetch_add(1, Ordering::SeqCst);
                    }
                    Some((Frame::LlmText(format!("t{i}")), i + 1))
                }
            })))
        }
        fn set_tools(&mut self, _tools: Vec<Tool>) {}
    }

    async fn run_llm_turn(llm: LlmProcessor<BargeInMidStream>) -> Arc<Tap> {
        let tap = Arc::new(Tap::default());
        let task = PipelineTask::new(
            Pipeline::new(vec![Box::new(llm)]),
            PipelineTaskParams::default(),
            vec![tap.clone() as Arc<dyn FrameObserver>],
        );
        task.queue_frame(Frame::LlmContext(Arc::new(LlmContext {
            messages: vec![json!({"role": "user", "content": "hi"})],
            tools: vec![],
        })))
        .await;
        task.stop_when_done().await;
        tokio::time::timeout(Duration::from_secs(5), task.run())
            .await
            .expect("llm pipeline timed out")
            .expect("run ok");
        tap
    }

    /// With the flag wired, a barge-in mid-stream stops the adapter pushing further
    /// tokens and closes the response framing. Without the cancel the whole reply
    /// would still be assembled and spoken *after* the user interrupted — the frame
    /// path can't preempt a `process_frame` that is already inside the stream.
    #[tokio::test]
    async fn barge_in_flag_cancels_an_in_flight_llm_stream() {
        let flag = Arc::new(AtomicU64::new(0));
        let tap = run_llm_turn(
            LlmProcessor::new(BargeInMidStream {
                flag: flag.clone(),
                chunks: 6,
                at: 2,
            })
            .with_interrupt_flag(flag),
        )
        .await;

        assert_eq!(
            tap.llm_text.lock().unwrap().clone(),
            vec!["t0".to_string(), "t1".to_string()],
            "tokens from the barge-in chunk onward must not be pushed"
        );
        assert!(
            tap.names.lock().unwrap().contains(&"LlmResponseEnd"),
            "a cancelled stream must still close its response framing"
        );
    }

    /// Same LLM, no flag wired (the default / half-duplex path): every token is
    /// pushed. Guards against the cancel changing stock behaviour.
    #[tokio::test]
    async fn without_the_flag_the_whole_stream_is_pushed() {
        // Same barge-in mid-stream, but no `with_interrupt_flag` — the adapter has
        // nothing to poll, so it streams to completion exactly as it does today.
        let tap = run_llm_turn(LlmProcessor::new(BargeInMidStream {
            flag: Arc::new(AtomicU64::new(0)),
            chunks: 4,
            at: 1,
        }))
        .await;
        assert_eq!(tap.llm_text.lock().unwrap().len(), 4);
    }
}
