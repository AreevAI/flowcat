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
}

impl<S: SttService> SttProcessor<S> {
    /// Wrap `svc` as a pipeline processor.
    pub fn new(svc: S) -> Self {
        Self {
            svc: Arc::new(Mutex::new(svc)),
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
}

impl<L: LlmService> LlmProcessor<L> {
    /// Wrap `svc` as a pipeline processor.
    pub fn new(svc: L) -> Self {
        Self {
            svc: Arc::new(Mutex::new(svc)),
        }
    }

    /// Run the LLM over `ctx`, pushing each streamed frame downstream in order.
    async fn run(&self, ctx: &LlmContext, link: &Link) {
        tracing::debug!(messages = ctx.messages.len(), "cascaded LLM run");
        let mut guard = self.svc.lock().await;
        // `run_llm` returns a `BoxStream<'a, Frame>` borrowing the guard, so the
        // guard outlives the stream. Push each frame as it arrives (true streaming);
        // `link.push_down` doesn't touch the guard, so the borrow is fine.
        let mut pushed = 0usize;
        let err = match guard.run_llm(ctx).await {
            Ok(mut stream) => {
                while let Some(f) = stream.next().await {
                    pushed += 1;
                    link.push_down(f).await;
                }
                None
            }
            Err(e) => Some(format!("llm: {e}")),
        };
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
