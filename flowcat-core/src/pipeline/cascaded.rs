// SPDX-License-Identifier: Apache-2.0
//
//! Cascaded STT→LLM→TTS topology builder + context aggregators.
//!
//! Builds the cascaded [`Pipeline`](crate::pipeline::Pipeline) from concrete
//! service processors (the [`adapters`](crate::service::adapters) wrapping the
//! frozen `SttService`/`LlmService`/`TtsService` traits), wiring in the
//! user/assistant **context aggregators** and an optional summarizer hook. This is
//! the cascaded counterpart to the live realtime/S2S path ([`crate::pipeline::s2s`]);
//! the service traits it builds against are frozen in [`crate::service`].
//!
//! ## Topology (pipecat `pipeline_builder.build_pipeline` shape)
//!
//! ```text
//!   transport.input → STT → UserContextAggregator → LLM → AssistantContextAggregator → TTS → transport.output
//! ```
//!
//! - **STT** ([`SttProcessor`]) turns `InputAudio` → `Transcription`.
//! - The **user aggregator** ([`UserContextAggregator`]) accrues each final
//!   `Transcription` into the shared rolling [`LlmContext`] as a `user` message and,
//!   on turn completion, emits a [`Frame::LlmContext`] that runs the LLM (mirrors
//!   pipecat `LLMUserContextAggregator`).
//! - **LLM** ([`LlmProcessor`]) streams `LlmResponseStart` / `LlmText`* /
//!   `LlmResponseEnd`.
//! - The **assistant aggregator** ([`AssistantContextAggregator`]) accrues the
//!   streamed `LlmText` between the start/end framing, appends the completed reply to
//!   the shared context as an `assistant` message, and emits a [`Frame::TtsSpeak`]
//!   to synthesize it (mirrors pipecat `LLMAssistantContextAggregator`).
//! - **TTS** ([`TtsProcessor`]) turns `TtsSpeak` → `OutputAudio` (mapped from
//!   `TtsAudio`) for the transport-out sink.
//!
//! A `transport.input` source and `transport.output` sink are **not** owned by this
//! builder (the live media path / transport layer provides them, fed via
//! [`SourcePump`](crate::pipeline::SourcePump)); the builder produces the inner
//! STT→…→TTS chain as a [`PipelineTask`], and the caller queues `InputAudio` at the
//! head and consumes `OutputAudio` at the tail. The mock fixture below drives it
//! exactly that way.
//!
//! ## The summarizer hook (fire-and-forget, pipecat parity)
//!
//! Per pipecat's `LLMContextSummarizer`, when the rolling context grows past a
//! threshold (or a turn boundary fires) the aggregator emits an
//! **on-transition summarize** signal: a fire-and-forget call into an injected
//! [`ContextSummarizer`] that compresses old history off the hot path. The builder
//! wires an optional summarizer; if none is given, the hook is a no-op (the mock
//! path). This is the cascaded analogue of the realtime path's transition re-prompt.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::time::Instant;

use crate::brain::AgentBrain;
use crate::codec::Resampler;
use crate::error::{FlowcatError, Result};
use crate::processor::frame::{Frame, LlmContext, StartParams};
use crate::processor::{Envelope, FrameProcessor, Link, ProcessorSetup};
use crate::service::adapters::{LlmProcessor, SttProcessor, TtsProcessor};
use crate::service::{LlmService, SttService, TtsService};
use crate::session::SessionSource;
use crate::transport::MediaTransport;
use crate::types::AudioChunk;

// Reuse the **realtime** path's outer processors verbatim (no parallel impl): the
// brain/transition wiring, the recorder/transcript/finalize taps, the shared
// live-call state, the transport head pump, and the tool-relay seam. These are the
// `pub(crate)` items in [`super::s2s`] — the cascaded task differs only in its
// *inner* chain (STT→agg→LLM→agg→TTS) and the LLM→brain tool-call bridge.
use super::s2s::{
    advance_playout, spawn_transport_pump, BrainProcessor, FinalizeProcessor, LiveState,
    ModelToolCall, RecorderProcessor, Reprompt, SessionToolRelay, SharedState, SharedTransport,
    ToolRelay, ToolResult, TranscriptProcessor, TransportInput, MAX_PLAYOUT_DRAIN,
};
use super::{Pipeline, PipelineTask, PipelineTaskParams, SourcePump};

// ===========================================================================
// Shared rolling LLM context — the cascaded analogue of S2S's LiveState.
// ===========================================================================

/// The rolling conversation context shared by the user + assistant aggregators.
/// Holds the message history (the same `messages` shape [`LlmContext`] carries) and
/// the current tool set. Guarded by a **`std::sync::Mutex`** that is only ever held
/// for the brief synchronous mutate-and-clone — **never across an `.await`** (every
/// aggregator snapshots under the lock, drops the guard, then awaits the push).
#[derive(Default)]
struct RollingContext {
    /// Message history (role/content JSON objects), oldest-first.
    messages: Vec<Value>,
    /// Optional system prompt prepended when building the run context.
    system_prompt: Option<String>,
    /// Tools advertised to the LLM (carried into each `LlmContext`).
    tools: Vec<Value>,
}

impl RollingContext {
    fn new(system_prompt: Option<String>, tools: Vec<Value>) -> Self {
        Self {
            messages: Vec::new(),
            system_prompt,
            tools,
        }
    }

    /// Append a `{role, content}` message.
    fn push(&mut self, role: &str, content: &str) {
        self.messages
            .push(json!({ "role": role, "content": content }));
    }

    /// Build an immutable [`LlmContext`] snapshot to run the LLM over (system
    /// prompt first if set, then the rolling history).
    fn snapshot(&self) -> LlmContext {
        let mut messages = Vec::with_capacity(self.messages.len() + 1);
        if let Some(sys) = &self.system_prompt {
            messages.push(json!({ "role": "system", "content": sys }));
        }
        messages.extend(self.messages.iter().cloned());
        LlmContext {
            messages,
            tools: self.tools.clone(),
        }
    }

    /// Number of non-system messages in the rolling history.
    fn len(&self) -> usize {
        self.messages.len()
    }

    /// Swap the active system prompt + tool set on a graph **transition** (the
    /// cascaded analogue of the realtime path's `update_system`). Subsequent
    /// [`snapshot`](RollingContext::snapshot)s carry the new prompt + tools, so the
    /// re-run LLM speaks for the destination node. `tools` arrive as the brain's
    /// [`ToolDecl`] shape and are stored as the opaque JSON the [`LlmContext`]
    /// carries.
    fn reprompt(&mut self, system_prompt: String, tools: Vec<Value>) {
        self.system_prompt = Some(system_prompt);
        self.tools = tools;
    }

    /// Append a workflow (MCP/HTTP) **tool result** to the rolling history as a
    /// `tool` message so the re-run LLM sees the result and continues the turn
    /// (the cascaded analogue of the realtime `send_tool_result`). The content is
    /// stored verbatim (a bare string, matching the brain's relay contract).
    fn push_tool_result(&mut self, tool_call_id: &str, content: &Value) {
        self.messages.push(json!({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }));
    }
}

type SharedContext = Arc<Mutex<RollingContext>>;

// ===========================================================================
// TurnMute — half-duplex turn lock (serializes turns, no self-echo).
// ===========================================================================

/// Half-duplex turn lock: STT stays muted from turn start (a final `Transcription`
/// → `LlmContext`, or the kickoff greeting) until the bot's reply has played out, so
/// a second question can't queue a concurrent LLM run and the bot can't transcribe
/// its own echo. Shared (Clone/`Arc`) by the user + kickoff processors (which
/// `begin` a turn) and the sink (which extends the playout estimate). No barge-in.
#[derive(Clone)]
struct TurnMute(Arc<TurnMuteInner>);

struct TurnMuteInner {
    muted: std::sync::atomic::AtomicBool,
    /// Bumped each `begin()` so a stale unmute watchdog supersedes itself.
    generation: std::sync::atomic::AtomicU64,
    /// When the bot's audio sent so far finishes playing at the carrier.
    bot_until: Mutex<Option<Instant>>,
    /// Injects `SttMute` at the pipeline head (→ down to the `SttProcessor`).
    end_tx: tokio::sync::mpsc::UnboundedSender<Frame>,
    /// Unmute after this long if a turn produces NO bot audio (a tool-only / no-reply
    /// turn), so the lock can never deadlock.
    no_audio_timeout: std::time::Duration,
}

impl TurnMute {
    fn new(
        end_tx: tokio::sync::mpsc::UnboundedSender<Frame>,
        no_audio_timeout: std::time::Duration,
    ) -> Self {
        TurnMute(Arc::new(TurnMuteInner {
            muted: std::sync::atomic::AtomicBool::new(false),
            generation: std::sync::atomic::AtomicU64::new(0),
            bot_until: Mutex::new(None),
            end_tx,
            no_audio_timeout,
        }))
    }

    /// Mute the STT (once) and spawn the unmute watchdog. No-op if a turn is already
    /// in flight — the lock holds the whole turn (covering the LLM-thinking gap too).
    fn begin(&self) {
        use std::sync::atomic::Ordering;
        if self.0.muted.swap(true, Ordering::SeqCst) {
            return; // already muted — a turn is in flight
        }
        tracing::debug!("half-duplex: muting STT (turn started)");
        let _ = self.0.end_tx.send(Frame::SttMute(true));
        *self.0.bot_until.lock().unwrap() = None;
        let my_gen = self.0.generation.fetch_add(1, Ordering::SeqCst) + 1;
        let inner = self.0.clone();
        tokio::spawn(async move {
            let deadline = Instant::now() + inner.no_audio_timeout;
            loop {
                if inner.generation.load(Ordering::SeqCst) != my_gen {
                    return; // superseded by a newer turn
                }
                let until = *inner.bot_until.lock().unwrap();
                let now = Instant::now();
                match until {
                    Some(t) if t > now => tokio::time::sleep(t - now).await, // bot still playing
                    Some(_) => break,                                        // reply done → unmute
                    None if now >= deadline => break, // no audio (tool-only turn) → safety unmute
                    None => tokio::time::sleep(std::time::Duration::from_millis(200)).await,
                }
            }
            if inner.generation.load(Ordering::SeqCst) == my_gen
                && inner.muted.swap(false, Ordering::SeqCst)
            {
                tracing::debug!("half-duplex: unmuting STT (turn complete)");
                let _ = inner.end_tx.send(Frame::SttMute(false));
            }
        });
    }

    /// Extend the playout estimate as the reply plays (the watchdog waits on it).
    fn note_bot_audio(&self, samples: usize, carrier_rate: u32) {
        let mut until = self.0.bot_until.lock().unwrap();
        *until = Some(advance_playout(
            *until,
            Instant::now(),
            samples,
            carrier_rate,
        ));
    }

    /// Take (and clear) the current playout estimate — for the sink's End-of-call drain.
    fn take_bot_until(&self) -> Option<Instant> {
        self.0.bot_until.lock().unwrap().take()
    }
}

// ===========================================================================
// Context summarizer hook (pipecat `LLMContextSummarizer`).
// ===========================================================================

/// A fire-and-forget context summarizer (pipecat `LLMContextSummarizer`). When the
/// rolling context crosses [`SummarizerConfig::trigger_after_messages`], the user
/// aggregator spawns [`summarize`](ContextSummarizer::summarize) off the hot path so
/// it never delays the turn. The default impl is a no-op (the mock/no-summarizer
/// path); a real impl compresses old history into a summary message.
#[async_trait]
pub trait ContextSummarizer: Send + Sync {
    /// Summarize `messages` (the rolling history at the transition) into a single
    /// replacement summary string, or `None` to leave the context untouched. Called
    /// fire-and-forget — its result is applied opportunistically and never blocks
    /// the next turn.
    async fn summarize(&self, messages: &[Value]) -> Option<String>;
}

/// Tuning for the summarizer hook (pipecat `LLMAutoContextSummarizationConfig`).
#[derive(Debug, Clone)]
pub struct SummarizerConfig {
    /// Fire the summarize hook once the rolling history reaches this many messages.
    pub trigger_after_messages: usize,
}

impl Default for SummarizerConfig {
    fn default() -> Self {
        // Mirrors pipecat's default of summarizing on a long-ish context; small
        // enough that a multi-turn call exercises it, large enough not to thrash.
        Self {
            trigger_after_messages: 20,
        }
    }
}

/// A no-op summarizer — the default when no summarizer is wired (the mock path).
struct NoopSummarizer;

#[async_trait]
impl ContextSummarizer for NoopSummarizer {
    async fn summarize(&self, _messages: &[Value]) -> Option<String> {
        None
    }
}

// ===========================================================================
// UserContextAggregator (pipecat LLMUserContextAggregator).
// ===========================================================================

/// Accrues final user `Transcription` frames into the shared rolling context as
/// `user` messages and, on each completed user turn, emits a [`Frame::LlmContext`]
/// snapshot downstream to run the LLM. Also fires the fire-and-forget summarizer
/// hook when the context grows past the configured threshold (pipecat
/// `LLMUserContextAggregator` + `LLMContextSummarizer` trigger).
///
/// The final transcription is **consumed** (transformed into the `LlmContext`, not
/// forwarded) so the downstream [`LlmProcessor`] is triggered exactly once per turn
/// — by the `LlmContext`, never also by a raw `Transcription` (which would
/// double-run the LLM). Observers still see the transcription via `on_process` at
/// the STT processor and here. Only a `final_` transcription closes the user turn
/// (the cascaded turn boundary); interim transcriptions and everything else pass
/// through unchanged.
pub struct UserContextAggregator {
    ctx: SharedContext,
    summarizer: Arc<dyn ContextSummarizer>,
    summarizer_cfg: SummarizerConfig,
    /// Guards against re-firing the summarizer for the same threshold crossing.
    summarize_in_flight: Arc<std::sync::atomic::AtomicBool>,
    /// Optional shared call state: when set, the final user transcription is tapped
    /// into the transcript here (it is *consumed* into the LlmContext below, so it
    /// never reaches the end-of-chain `TranscriptProcessor`). `None` in the bare
    /// inner pipeline / unit tests.
    transcript_state: Option<SharedState>,
    /// Optional half-duplex turn lock: `begin()` is called as each user turn starts
    /// (when the `LlmContext` is emitted), so a second question can't queue a
    /// concurrent LLM run. `None` in the bare inner pipeline / unit tests.
    turn_mute: Option<TurnMute>,
}

impl UserContextAggregator {
    fn new(
        ctx: SharedContext,
        summarizer: Arc<dyn ContextSummarizer>,
        summarizer_cfg: SummarizerConfig,
    ) -> Self {
        Self {
            ctx,
            summarizer,
            summarizer_cfg,
            summarize_in_flight: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            transcript_state: None,
            turn_mute: None,
        }
    }

    /// Tap the final user transcription into this shared call state's transcript
    /// (the cascaded path's user side; the bot side rides `LlmText` to the
    /// `TranscriptProcessor`). Builder-style so unit tests need no change.
    fn with_transcript_state(mut self, state: SharedState) -> Self {
        self.transcript_state = Some(state);
        self
    }

    /// Wire the turn lock (live builder sets it; unit tests omit it).
    fn with_turn_mute(mut self, tm: TurnMute) -> Self {
        self.turn_mute = Some(tm);
        self
    }

    /// Fire the summarizer off the hot path if the context grew past the threshold
    /// (pipecat: summarize on transition, fire-and-forget). Applies the returned
    /// summary by collapsing the history into a single summary message.
    fn maybe_summarize(&self, len: usize) {
        use std::sync::atomic::Ordering;
        if len < self.summarizer_cfg.trigger_after_messages {
            return;
        }
        // One in-flight summarize at a time (the pipecat `_summarization_in_progress`
        // guard) — a compare-exchange so we never double-spawn.
        if self
            .summarize_in_flight
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return;
        }
        let ctx = self.ctx.clone();
        let summarizer = self.summarizer.clone();
        let in_flight = self.summarize_in_flight.clone();
        tokio::spawn(async move {
            // Snapshot the history under the std lock, drop the guard before await.
            let history = {
                let c = ctx.lock().unwrap();
                c.messages.clone()
            };
            if let Some(summary) = summarizer.summarize(&history).await {
                // Apply: replace the history with a single summary message (the
                // pipecat "compress old history into a summary" result).
                let mut c = ctx.lock().unwrap();
                c.messages = vec![
                    json!({ "role": "system", "content": format!("Summary so far: {summary}") }),
                ];
            }
            in_flight.store(false, Ordering::SeqCst);
        });
    }
}

#[async_trait]
impl FrameProcessor for UserContextAggregator {
    fn name(&self) -> &str {
        "UserContextAggregator"
    }

    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        if let Frame::Transcription {
            text, final_: true, ..
        } = &env.frame
        {
            // Tap the user line into the transcript (it's consumed below, so the
            // end-of-chain TranscriptProcessor never sees it).
            if let Some(st) = &self.transcript_state {
                st.lock().unwrap().transcript.push_user(text);
            }
            // Accrue the user message + snapshot the context under the lock; drop the
            // guard before awaiting the push (no std Mutex held across await).
            let (snapshot, len) = {
                let mut c = self.ctx.lock().unwrap();
                c.push("user", text);
                (c.snapshot(), c.len())
            };
            // Fire the summarizer hook off the hot path (pipecat transition summarize).
            self.maybe_summarize(len);
            // Lock the turn (mute STT until the reply plays out) so a second question
            // can't queue a concurrent LLM run.
            if let Some(tm) = &self.turn_mute {
                tm.begin();
            }
            // Run the LLM over the aggregated context (pipecat: aggregator → LLMRun).
            // The transcription is CONSUMED here (transformed into the LlmContext, not
            // forwarded) so the LLM fires exactly once per turn — never also on a raw
            // Transcription. Observers already saw it via on_process at STT + here.
            link.push_down(Frame::LlmContext(Arc::new(snapshot))).await;
            return Ok(());
        }
        link.push(env.meta, env.frame, env.direction).await;
        Ok(())
    }
}

// ===========================================================================
// AssistantContextAggregator (pipecat LLMAssistantContextAggregator).
// ===========================================================================

/// Accrues the streamed LLM response (`LlmText` between `LlmResponseStart` and
/// `LlmResponseEnd`) into a single assistant turn, appends the completed reply to
/// the shared rolling context as an `assistant` message, and emits a
/// [`Frame::TtsSpeak`] to synthesize it (pipecat `LLMAssistantContextAggregator`).
///
/// `LlmText` chunks are buffered (not forwarded individually); the assembled reply
/// is emitted once as a `TtsSpeak` on `LlmResponseEnd`, so the TTS service speaks
/// whole utterances. The framing + text frames are still forwarded so downstream
/// transcript taps observe the response. Everything else passes through unchanged.
pub struct AssistantContextAggregator {
    ctx: SharedContext,
    /// The in-progress assistant turn (accrued `LlmText`).
    buffer: String,
    /// Whether we are inside a `LlmResponseStart`/`End` span.
    in_response: bool,
    /// Optional shared call state: when set, the completed bot reply is tapped into
    /// the transcript here (the cascaded bot side; the user side is tapped in
    /// `UserContextAggregator`). `None` in the bare inner pipeline / unit tests.
    transcript_state: Option<SharedState>,
}

impl AssistantContextAggregator {
    fn new(ctx: SharedContext) -> Self {
        Self {
            ctx,
            buffer: String::new(),
            in_response: false,
            transcript_state: None,
        }
    }

    /// Tap the completed bot reply into this shared call state's transcript.
    /// Builder-style so unit tests need no change.
    fn with_transcript_state(mut self, state: SharedState) -> Self {
        self.transcript_state = Some(state);
        self
    }
}

#[async_trait]
impl FrameProcessor for AssistantContextAggregator {
    fn name(&self) -> &str {
        "AssistantContextAggregator"
    }

    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        match &env.frame {
            Frame::LlmResponseStart => {
                self.in_response = true;
                self.buffer.clear();
                link.push(env.meta, env.frame, env.direction).await;
            }
            Frame::LlmText(chunk) => {
                // Accrue the streamed token and CONSUME it (do NOT forward). The
                // assembled reply is emitted once as a `TtsSpeak` on `LlmResponseEnd`,
                // so the TTS service speaks the whole utterance exactly once.
                // Forwarding each token would make `TtsProcessor` synthesize every
                // word individually (it also speaks raw `LlmText` — its no-aggregator
                // fallback), producing the choppy word-by-word "hi. how. can. i…"
                // double-speak. The bot transcript is tapped directly below
                // (`push_bot`), so nothing downstream needs the raw tokens. Symmetric
                // with `UserContextAggregator`, which consumes the `Transcription`.
                self.buffer.push_str(chunk);
            }
            Frame::LlmResponseEnd => {
                self.in_response = false;
                let reply = std::mem::take(&mut self.buffer);
                if !reply.is_empty() {
                    // Tap the bot reply into the transcript (the cascaded bot side).
                    if let Some(st) = &self.transcript_state {
                        st.lock().unwrap().transcript.push_bot(&reply);
                    }
                    // Append the completed reply to the rolling context (under the
                    // lock, then drop before awaiting).
                    {
                        let mut c = self.ctx.lock().unwrap();
                        c.push("assistant", &reply);
                    }
                    // Drive TTS with the whole assembled utterance.
                    link.push_down(Frame::TtsSpeak {
                        text: reply,
                        append_to_context: Some(false),
                    })
                    .await;
                }
                link.push(env.meta, env.frame, env.direction).await;
            }
            _ => link.push(env.meta, env.frame, env.direction).await,
        }
        Ok(())
    }
}

// ===========================================================================
// The builder.
// ===========================================================================

/// Optional knobs for [`build_cascaded_pipeline`]. Defaults give the mock-friendly
/// shape (no system prompt, no tools, no summarizer, default task params).
pub struct CascadedConfig {
    /// Initial system prompt prepended to every LLM run context.
    pub system_prompt: Option<String>,
    /// Initial tool set advertised to the LLM (carried into each `LlmContext`).
    pub tools: Vec<Value>,
    /// Optional context summarizer (fire-and-forget on transition). `None` ⇒ no-op.
    pub summarizer: Option<Arc<dyn ContextSummarizer>>,
    /// Summarizer trigger tuning.
    pub summarizer_cfg: SummarizerConfig,
    /// Pipeline task params (audio rates, idle timeout, …).
    pub task_params: PipelineTaskParams,
}

impl Default for CascadedConfig {
    fn default() -> Self {
        Self {
            system_prompt: None,
            tools: Vec::new(),
            summarizer: None,
            summarizer_cfg: SummarizerConfig::default(),
            // The cascaded path has its own turn boundaries; keep the idle gate off
            // by default so a fixture run never trips the 300 s timer.
            task_params: PipelineTaskParams {
                idle_timeout: None,
                ..Default::default()
            },
        }
    }
}

/// Assemble the cascaded STT→user-agg→LLM→assistant-agg→TTS [`Pipeline`] +
/// `PipelineTaskParams` from the service impls + config (the shared inner builder
/// behind [`build_cascaded_pipeline`] / [`build_cascaded_task_with_observers`]).
fn assemble<S, L, T>(
    stt: S,
    llm: L,
    tts: T,
    config: CascadedConfig,
) -> (Pipeline, PipelineTaskParams)
where
    S: SttService + 'static,
    L: LlmService + 'static,
    T: TtsService + 'static,
{
    let ctx: SharedContext = Arc::new(Mutex::new(RollingContext::new(
        config.system_prompt,
        config.tools,
    )));
    let summarizer = config
        .summarizer
        .unwrap_or_else(|| Arc::new(NoopSummarizer) as Arc<dyn ContextSummarizer>);

    let processors: Vec<Box<dyn FrameProcessor>> = vec![
        Box::new(SttProcessor::new(stt)),
        Box::new(UserContextAggregator::new(
            ctx.clone(),
            summarizer,
            config.summarizer_cfg,
        )),
        Box::new(LlmProcessor::new(llm)),
        Box::new(AssistantContextAggregator::new(ctx.clone())),
        Box::new(TtsProcessor::new(tts)),
    ];
    (Pipeline::new(processors), config.task_params)
}

/// Build the cascaded STT→LLM→TTS [`PipelineTask`]
/// (`transport.input → STT → user-agg → LLM → assistant-agg → TTS → transport.output`).
///
/// `stt`/`llm`/`tts` are any impls of the frozen
/// [`SttService`]/[`LlmService`]/[`TtsService`] traits — the mock impls for tests,
/// real providers (Deepgram/OpenAI/Cartesia in `flowcat-services`) in production.
/// The returned task is driven exactly like the S2S task: queue `InputAudio` at the
/// head (or feed a transport via [`SourcePump`](crate::pipeline::SourcePump)) and
/// consume `OutputAudio` at the tail.
///
/// This is generic over the three service traits and pulls **no network dependency**
/// — flowcat-core stays dep-light; the providers live in `flowcat-services`.
pub fn build_cascaded_pipeline<S, L, T>(
    stt: S,
    llm: L,
    tts: T,
    config: CascadedConfig,
) -> PipelineTask
where
    S: SttService + 'static,
    L: LlmService + 'static,
    T: TtsService + 'static,
{
    build_cascaded_task_with_observers(stt, llm, tts, config, vec![])
}

/// As [`build_cascaded_pipeline`], but attaches `observers` to the task (the
/// fixture path provider tests use to assert frame flow, and the live path uses to wire
/// metrics/RTVI/transcript observers).
pub fn build_cascaded_task_with_observers<S, L, T>(
    stt: S,
    llm: L,
    tts: T,
    config: CascadedConfig,
    observers: Vec<Arc<dyn crate::observer::FrameObserver>>,
) -> PipelineTask
where
    S: SttService + 'static,
    L: LlmService + 'static,
    T: TtsService + 'static,
{
    let (pipeline, params) = assemble(stt, llm, tts, config);
    PipelineTask::new(pipeline, params, observers)
}

// ===========================================================================
// `build_cascaded_task`: the cascaded analogue of `build_s2s_task`.
//
// The realtime `build_s2s_task` composes the realtime model with the brain /
// recorder / transcript / finalize outer processors. The cascaded path needs the
// SAME outer processors around the cascaded INNER chain (STT→user-agg→LLM→
// assistant-agg→TTS). The one new piece is the LLM→brain **tool-call bridge**:
//
// ```text
//  [pump]→TransportInput→STT→UserAgg→LLM→CascadedToolBridge→Brain→AssistantAgg→TTS→CascadedOutput→Recorder→Transcript→Finalize
//                                        │  (FunctionCallsStarted → ModelToolCall, downstream)         ▲
//                                        ▲  (Reprompt/ToolResult upstream → context update + LLM re-run)│
// ```
//
// In realtime mode the model's tool-calls drive both workflow tools (via
// `ToolRelay`) and graph transitions (via the `AgentBrain` seam on a tool call); in
// cascaded mode the **cascaded LLM's** function-calls drive the exact same brain —
// the only difference is the *source* of the tool-call (the LLM, not the realtime
// model) and how the brain's result is consumed (fed back into the rolling LLM
// context + a fresh LLM run, rather than `realtime.send_tool_result`).
//
// Generic over the four seams + the three services, so it works with both the mocks
// and the real flowcat-services providers, and flowcat-core stays embedder-agnostic.
// ===========================================================================

// ---------------------------------------------------------------------------
// CascadedToolBridge — the LLM↔brain glue.
// ---------------------------------------------------------------------------

/// The bridge that lets the **reused realtime [`BrainProcessor`]** drive the
/// cascaded LLM. It sits between the [`LlmProcessor`] (upstream) and the
/// [`BrainProcessor`] (downstream) and translates in both directions:
///
/// - **Downstream (LLM → brain):** the cascaded LLM emits
///   [`Frame::FunctionCallsStarted`] when it calls a function. The bridge converts
///   each [`FunctionCall`](crate::processor::frame::FunctionCall) into a
///   [`ModelToolCall`] custom frame — the **exact**
///   frame the realtime `RealtimeServiceProcessor` emits — so the downstream
///   `BrainProcessor` treats a cascaded LLM tool-call identically to a realtime
///   model tool-call (workflow-relay vs. transition vs. end). The
///   `FunctionCallsStarted` frame itself is **consumed** (not forwarded) so the
///   assistant aggregator / TTS never speak a tool-call as if it were a reply.
///
/// - **Upstream (brain → LLM):** the `BrainProcessor` pushes its decision **upstream**
///   as [`Reprompt`] / [`ToolResult`] custom frames (toward where the realtime
///   service would consume them). The bridge intercepts them and applies the
///   cascaded analogue:
///     * [`Reprompt`] (a graph **transition**) → swap the rolling context's system
///       prompt + tools, then re-run the LLM (emit a fresh [`Frame::LlmContext`]
///       upstream to the [`LlmProcessor`]) so the bot speaks for the new node.
///     * [`ToolResult`] whose payload is a bare value (a **workflow/MCP** result,
///       per the brain's relay contract — `json!(content)`, not a `{status}`
///       envelope) → append it to the rolling context as a `tool` message and re-run
///       the LLM so the model continues the turn with the result in hand.
///     * [`ToolResult`] carrying a `{status: moved|stay|ended}` ack → a no-op (the
///       transition is already driven by the paired `Reprompt`; `ended` is driven by
///       the brain's head-`End`; `stay` needs no re-run).
///
/// This is the cascaded counterpart of the realtime `RealtimeServiceProcessor`'s
/// upstream-frame consumer (`send_tool_result`/`update_system`) — same brain, same
/// custom frames, a different back-end.
///
/// True iff a [`ToolResult`] payload is a graph transition/stay/end **ACK** (the
/// brain's `{status: moved|stay|ended}` envelope), as opposed to a workflow/MCP tool
/// result that must be fed back to the LLM. We match the brain's exact ACK vocabulary
/// — NOT merely the presence of a `status` key — because MCP tool results are JSON
/// objects (see [`s2s::mcp_result_struct`]) and many carry their own
/// `{"status": "ok"/"confirmed"}`; those must reach the LLM, not be swallowed as acks.
fn is_transition_ack(result: &Value) -> bool {
    result
        .get("status")
        .and_then(Value::as_str)
        .is_some_and(|s| matches!(s, "moved" | "stay" | "ended"))
}

struct CascadedToolBridge {
    /// The shared rolling LLM context (also held by the user/assistant aggregators),
    /// updated on transitions + tool results.
    ctx: SharedContext,
}

impl CascadedToolBridge {
    fn new(ctx: SharedContext) -> Self {
        Self { ctx }
    }
}

#[async_trait]
impl FrameProcessor for CascadedToolBridge {
    fn name(&self) -> &str {
        "CascadedToolBridge"
    }

    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        match &env.frame {
            // ---- Downstream: LLM tool-calls → the brain (as ModelToolCall) ----
            // Convert each FunctionCall to the realtime path's ModelToolCall frame so
            // the reused BrainProcessor consumes it unchanged. Consume the original
            // (don't forward it past the brain into the assistant aggregator / TTS).
            Frame::FunctionCallsStarted(calls) => {
                for c in calls {
                    link.push_down(Frame::Custom(Arc::new(ModelToolCall {
                        id: c.tool_call_id.clone(),
                        name: c.function_name.clone(),
                        args: c.arguments.clone(),
                    })))
                    .await;
                }
            }

            // ---- Upstream: a transition re-prompt from the brain ----
            // Swap the rolling context's prompt + tools, then re-run the LLM so the
            // bot speaks for the destination node (cascaded analogue of update_system
            // + the model continuing). Tools carry as the opaque JSON the LlmContext
            // holds.
            Frame::Custom(c) if c.as_any().is::<Reprompt>() => {
                let rp = c.as_any().downcast_ref::<Reprompt>().unwrap();
                let tools_json: Vec<Value> = rp
                    .tools
                    .iter()
                    .map(|t| serde_json::to_value(t).unwrap_or(Value::Null))
                    .collect();
                let snapshot = {
                    let mut ctx = self.ctx.lock().unwrap();
                    ctx.reprompt(rp.prompt.clone(), tools_json);
                    ctx.snapshot()
                };
                // Re-run the LLM (upstream) for the new node. A transition → a second
                // utterance; this log helps spot an auto-transition right after the greeting.
                tracing::debug!("cascaded transition: reprompt → re-running LLM for new node");
                link.push_up(Frame::LlmContext(Arc::new(snapshot))).await;
            }

            // ---- Upstream: a tool result from the brain ----
            Frame::Custom(c) if c.as_any().is::<ToolResult>() => {
                let tr = c.as_any().downcast_ref::<ToolResult>().unwrap();
                // A `{status: moved|stay|ended}` envelope is a transition/stay/end
                // ACK (no re-run here — the transition re-runs via its paired
                // Reprompt; `ended` via the brain's head-End). Any other payload is a
                // workflow/MCP result → feed it back + re-run.
                if !is_transition_ack(&tr.result) {
                    let snapshot = {
                        let mut ctx = self.ctx.lock().unwrap();
                        ctx.push_tool_result(&tr.id, &tr.result);
                        ctx.snapshot()
                    };
                    link.push_up(Frame::LlmContext(Arc::new(snapshot))).await;
                }
            }

            // Everything else (LlmResponseStart/Text/End, audio, lifecycle) flows on
            // to the brain → assistant aggregator → TTS unchanged.
            _ => link.push(env.meta, env.frame, env.direction).await,
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CascadedTransportOutput — the sink, parameterized by the TTS source rate.
// ---------------------------------------------------------------------------

/// The cascaded sink-side stage. Like the realtime [`super::s2s::TransportOutput`]
/// but the source rate is the **TTS service's** sample rate (not Gemini's fixed
/// 24 kHz): it records the bot leg at the TTS rate and resamples TTS-rate→carrier
/// before `send_audio`. Consumes `Interruption` → `send_clear`. Records into the
/// **same** shared [`LiveState`] the reused [`RecorderProcessor`]/[`FinalizeProcessor`]
/// read.
struct CascadedTransportOutput<T: MediaTransport> {
    transport: T,
    out_resampler: Option<Resampler>,
    tts_rate: u32,
    carrier_rate: u32,
    state: SharedState,
    /// Sends `Frame::End` once on transport death.
    end_tx: tokio::sync::mpsc::UnboundedSender<Frame>,
    /// Turn lock: extend its playout estimate per chunk (drives unmute + the End-drain).
    turn_mute: TurnMute,
}

impl<T: MediaTransport> CascadedTransportOutput<T> {
    fn new(
        transport: T,
        tts_rate: u32,
        carrier_rate: u32,
        state: SharedState,
        end_tx: tokio::sync::mpsc::UnboundedSender<Frame>,
        turn_mute: TurnMute,
    ) -> Self {
        Self {
            transport,
            out_resampler: None,
            tts_rate,
            carrier_rate,
            state,
            end_tx,
            turn_mute,
        }
    }

    /// Terminal transport error (peer gone): record it (de-duped to one log) and end
    /// the call **once**. The `OutputAudio` guard then drops remaining buffered frames
    /// instead of re-failing on the dead transport.
    fn fail_transport(&self, e: FlowcatError) {
        let newly_dead = {
            let mut st = self.state.lock().unwrap();
            let was = st.transport_dead;
            st.record_error(e); // logs once + sets transport_dead for a transport error
            st.transport_dead && !was
        };
        if newly_dead {
            let _ = self.end_tx.send(Frame::End { reason: None });
        }
    }
}

#[async_trait]
impl<T: MediaTransport + 'static> FrameProcessor for CascadedTransportOutput<T> {
    fn name(&self) -> &str {
        "CascadedTransportOutput"
    }

    async fn start(&mut self, _setup: &ProcessorSetup, _params: &StartParams) -> Result<()> {
        self.out_resampler = Some(Resampler::new(self.tts_rate, self.carrier_rate)?);
        Ok(())
    }

    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        match &env.frame {
            // Bot audio out (the TtsProcessor maps TtsAudio → OutputAudio): record at
            // the TTS rate, resample TTS-rate→carrier, send to the carrier.
            Frame::OutputAudio(audio) | Frame::TtsAudio { audio, .. } => {
                // Transport already dead → drop buffered bot audio (no re-send, no
                // repeat WARN); the call is tearing down.
                if self.state.lock().unwrap().transport_dead {
                    return Ok(());
                }
                let chunk = AudioChunk::from(audio.as_ref());
                self.state.lock().unwrap().recorder.push_outbound(&chunk);
                let resampler = self
                    .out_resampler
                    .as_mut()
                    .expect("out_resampler set in start()");
                match resampler.process(&chunk) {
                    Ok(down) => {
                        if !down.is_empty() {
                            let samples = down.pcm.len();
                            if let Err(e) = self.transport.send_audio(down).await {
                                self.fail_transport(e);
                            } else {
                                // Keep STT muted until this reply finishes playing.
                                self.turn_mute.note_bot_audio(samples, self.carrier_rate);
                            }
                        }
                    }
                    Err(e) => self.state.lock().unwrap().record_error(e),
                }
            }
            // Barge-in: flush the carrier playback.
            Frame::Interruption => {
                if let Err(e) = self.transport.send_clear().await {
                    self.fail_transport(e);
                }
                let _ = self.turn_mute.take_bot_until();
                link.push(env.meta, env.frame, env.direction).await;
            }
            // End-of-call: wait out the final bot utterance still playing at the
            // carrier before teardown (mirrors the realtime sink — see s2s.rs).
            Frame::End { .. } => {
                let until = self.turn_mute.take_bot_until();
                if let Some(until) = until {
                    let now = Instant::now();
                    if until > now {
                        tokio::time::sleep((until - now).min(MAX_PLAYOUT_DRAIN)).await;
                    }
                }
                link.push(env.meta, env.frame, env.direction).await;
            }
            _ => link.push(env.meta, env.frame, env.direction).await,
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// The assembler.
// ---------------------------------------------------------------------------

/// A built cascaded task plus its transport pump handle — the cascaded analogue of
/// [`super::s2s::S2sTask`].
///
/// Drive it with `task.run().await`; the pump feeds the transport's `recv()` into
/// the pipeline head and aborts when `run()` returns.
pub struct CascadedTask {
    /// The wired pipeline task — run it to completion.
    pub task: PipelineTask,
    /// The transport-pump source reader (aborted on drop / after `run`).
    pump: SourcePump,
}

impl CascadedTask {
    /// Run the cascaded pipeline to completion, then stop the transport pump.
    pub async fn run(self) -> Result<()> {
        let res = self.task.run().await;
        self.pump.abort();
        res
    }
}

// ===========================================================================
// CascadedKickoffProcessor — the cascaded analogue of the realtime `kickoff`.
// ===========================================================================

/// Makes the bot **greet on connect** (pipecat parity). The transport pump emits
/// [`Frame::ClientConnected`] when the media stream starts; the realtime path
/// consumes it and calls `realtime.kickoff()` so the model speaks first
/// ([`super::s2s`] `ClientConnected → kickoff`). The cascaded chain had no such
/// consumer, so the bot stayed silent until the user spoke. This processor closes
/// that gap: on the **first** `ClientConnected` it snapshots the already-seeded
/// rolling context (system prompt + start-node tools, empty history) and emits a
/// [`Frame::LlmContext`] — the exact frame the user aggregator emits per turn — so
/// the downstream `LlmProcessor → BrainProcessor → AssistantContextAggregator →
/// TtsProcessor` path generates and speaks the greeting from the node prompt. The
/// assistant aggregator appends the greeting to the rolling context, so the first
/// real user turn carries correct history. Idempotent (greets once per call).
struct CascadedKickoffProcessor {
    ctx: SharedContext,
    kicked_off: bool,
    turn_mute: Option<TurnMute>,
}

impl CascadedKickoffProcessor {
    fn new(ctx: SharedContext) -> Self {
        Self {
            ctx,
            kicked_off: false,
            turn_mute: None,
        }
    }

    /// Wire the turn lock (live builder sets it; unit tests omit it).
    fn with_turn_mute(mut self, tm: TurnMute) -> Self {
        self.turn_mute = Some(tm);
        self
    }
}

#[async_trait]
impl FrameProcessor for CascadedKickoffProcessor {
    fn name(&self) -> &str {
        "CascadedKickoff"
    }

    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        if !self.kicked_off && matches!(env.frame, Frame::ClientConnected) {
            self.kicked_off = true;
            tracing::info!("cascaded kickoff: greeting on ClientConnected");
            // The greeting is a bot turn → lock it.
            if let Some(tm) = &self.turn_mute {
                tm.begin();
            }
            // Run the LLM once over the seeded context (system prompt only). Lock is
            // dropped before the await (no std Mutex held across `.await`).
            let mut snapshot = self.ctx.lock().unwrap().snapshot();
            // Text-only greeting: clear tools so the opening turn can't `endCall`/
            // transition and hang up before the bot speaks. Tools return on user turns.
            snapshot.tools.clear();
            link.push_down(Frame::LlmContext(Arc::new(snapshot))).await;
        }
        // Always forward the original frame so `ClientConnected` keeps propagating.
        link.push(env.meta, env.frame, env.direction).await;
        Ok(())
    }
}

/// Assemble the cascaded processor task beside [`build_s2s_task`](super::s2s::build_s2s_task):
/// the cascaded **inner** chain (STT → user-agg → LLM → assistant-agg → TTS, the
/// [`build_cascaded_pipeline`] topology) wrapped with the **same outer processors**
/// `build_s2s_task` uses — `TransportInput` (fed by the transport [`SourcePump`]),
/// the brain/transition wiring ([`BrainProcessor`] + [`SessionToolRelay`]), the
/// recorder / transcript / finalize taps, and a TTS-rate-aware transport sink. The
/// LLM↔brain [`CascadedToolBridge`] integrates the cascaded LLM's function-calls into
/// the brain exactly as the realtime model's tool-calls feed it in `build_s2s_task`.
///
/// Mirrors `build_s2s_task`'s structure: resolve the start node's tools, build the
/// shared state, share the transport between the pump and the sink, route an inner
/// `End` request through the head, and spawn the pump. Generic over the four
/// flowcat-core seams (`MediaTransport`/`AgentBrain`/`SessionSource` + the three
/// services), so it drives both mocks and real providers; flowcat-core stays
/// embedder-agnostic.
///
/// Delegates to [`build_cascaded_task_with_observers`] with no observers
/// (historical signature — unchanged for every existing caller + test).
#[allow(clippy::too_many_arguments)]
pub async fn build_cascaded_task<Tr, St, L, Ts, B, Se>(
    transport: Tr,
    stt: St,
    llm: L,
    tts: Ts,
    brain: B,
    session: Se,
    run_id: i64,
    token: String,
    config: CascadedConfig,
) -> Result<CascadedTask>
where
    Tr: MediaTransport + 'static,
    St: SttService + 'static,
    L: LlmService + 'static,
    Ts: TtsService + 'static,
    B: AgentBrain + 'static,
    Se: SessionSource + 'static,
{
    build_cascaded_call_with_observers(
        transport,
        stt,
        llm,
        tts,
        brain,
        session,
        run_id,
        token,
        config,
        vec![],
    )
    .await
}

/// `build_cascaded_task` with external pipeline `observers` (e.g. an `RtviObserver`
/// streaming live transcript/RTF events). The full-call (outer) counterpart of the
/// inner [`build_cascaded_task_with_observers`]; this is the one the media host's
/// `run_call` uses.
#[allow(clippy::too_many_arguments)]
pub async fn build_cascaded_call_with_observers<Tr, St, L, Ts, B, Se>(
    transport: Tr,
    stt: St,
    llm: L,
    tts: Ts,
    brain: B,
    session: Se,
    run_id: i64,
    token: String,
    config: CascadedConfig,
    observers: Vec<Arc<dyn crate::observer::FrameObserver>>,
) -> Result<CascadedTask>
where
    Tr: MediaTransport + 'static,
    St: SttService + 'static,
    L: LlmService + 'static,
    Ts: TtsService + 'static,
    B: AgentBrain + 'static,
    Se: SessionSource + 'static,
{
    let carrier_rate = transport.carrier_rate();
    let tts_rate = tts.sample_rate();
    let session = Arc::new(session);
    let relay = Arc::new(SessionToolRelay::new(
        session.clone(),
        run_id,
        token.clone(),
    ));

    // 1. Resolve the start node's tool set (transitions + node MCP tools) and the
    //    MCP-name branch set — verbatim build_s2s_task's resolve ordering.
    let node_id = brain.current_node_id();
    let mcp = relay.node_tools(&node_id).await;
    let mcp_names: std::collections::HashSet<String> = mcp.iter().map(|t| t.name.clone()).collect();
    let mut initial_tools = brain.tools();
    initial_tools.extend(mcp);
    let initial_tools_json: Vec<Value> = initial_tools
        .iter()
        .map(|t| serde_json::to_value(t).unwrap_or(Value::Null))
        .collect();

    // 2. The shared rolling LLM context (the cascaded analogue of S2S's LiveState
    //    re-prompt): seeded with the brain's opening system prompt + the resolved
    //    start-node tools, shared by the user/assistant aggregators + the bridge.
    let ctx: SharedContext = Arc::new(Mutex::new(RollingContext::new(
        Some(brain.system_prompt()),
        initial_tools_json,
    )));

    // 3. The shared live-call state (reused from the realtime path).
    let state: SharedState = Arc::new(Mutex::new(LiveState::new(carrier_rate)));

    // 4. Share the one transport between the pump (recv) and the sink (send).
    let shared = SharedTransport::new(transport);

    // 5. The end-request channel (reused pattern): an inner processor (brain End)
    //    requests a clean drain through the pipeline head.
    let (end_tx, mut end_rx) = tokio::sync::mpsc::unbounded_channel::<Frame>();

    // 5b. The half-duplex turn lock (12 s safety timeout for a no-reply turn).
    let turn_mute = TurnMute::new(end_tx.clone(), std::time::Duration::from_secs(12));

    // 6. The summarizer (reuse the cascaded builder's no-op default unless wired).
    let summarizer = config
        .summarizer
        .unwrap_or_else(|| Arc::new(NoopSummarizer) as Arc<dyn ContextSummarizer>);

    // 7. Build the processor chain: the cascaded inner chain wrapped with the
    //    realtime path's outer processors + the LLM↔brain bridge.
    let processors: Vec<Box<dyn FrameProcessor>> = vec![
        Box::new(TransportInput::new()),
        Box::new(SttProcessor::new(stt)),
        Box::new(
            UserContextAggregator::new(ctx.clone(), summarizer, config.summarizer_cfg)
                .with_transcript_state(state.clone())
                .with_turn_mute(turn_mute.clone()),
        ),
        // Greet on connect (pipecat parity): on ClientConnected, run the LLM once
        // over the seeded system prompt so the bot speaks first. Placed right before
        // the LLM so its emitted LlmContext reaches LlmProcessor directly.
        Box::new(CascadedKickoffProcessor::new(ctx.clone()).with_turn_mute(turn_mute.clone())),
        Box::new(LlmProcessor::new(llm)),
        Box::new(CascadedToolBridge::new(ctx.clone())),
        Box::new(BrainProcessor::new(
            brain,
            relay,
            mcp_names,
            state.clone(),
            end_tx.clone(),
        )),
        Box::new(AssistantContextAggregator::new(ctx.clone()).with_transcript_state(state.clone())),
        Box::new(TtsProcessor::new(tts)),
        Box::new(CascadedTransportOutput::new(
            shared.clone(),
            tts_rate,
            carrier_rate,
            state.clone(),
            end_tx.clone(),
            turn_mute.clone(),
        )),
        Box::new(RecorderProcessor::new(state.clone())),
        Box::new(TranscriptProcessor::new(state.clone())),
        Box::new(FinalizeProcessor::new(
            session,
            run_id,
            token,
            state.clone(),
        )),
    ];
    let pipeline = Pipeline::new(processors);

    // 8. Build the task. The cascaded path has its own turn boundaries; keep the
    //    idle gate off (the call ends on brain End / transport Stop), mirroring
    //    build_s2s_task. Honor the caller's task_params but force the idle gate off.
    let params = PipelineTaskParams {
        idle_timeout: None,
        ..config.task_params
    };
    let task = PipelineTask::new(pipeline, params, observers);

    // 9. Forward end-requests into the task head (Source breaks → clean drain).
    let head = task.queue_sender();
    tokio::spawn(async move {
        while let Some(f) = end_rx.recv().await {
            if head.send(f).is_err() {
                break;
            }
        }
    });

    // 10. Spawn the transport pump feeding the task head (reused from the realtime
    //     path — emits ClientConnected / InputAudio / End at the head).
    let pump = spawn_transport_pump(shared, task.queue_sender());

    Ok(CascadedTask { task, pump })
}

// ===========================================================================
// Tests — the cascaded pipeline integration tests.
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::observer::{FrameEvent, FrameObserver};
    use crate::processor::frame::AudioFrame;
    use crate::service::{MockLlm, MockStt, MockTts};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    /// An observer that records every processed frame name + flags output audio.
    /// `on_process` fires once **per processor** that receives a frame, so a
    /// `TtsSpeak` is observed at both the TTS processor and the internal Sink; the
    /// tap dedupes the captured TtsSpeak texts by the frame's monotonic id so it
    /// records one entry per *distinct emitted* utterance.
    #[derive(Default)]
    struct Tap {
        names: Mutex<Vec<&'static str>>,
        saw_output_audio: AtomicBool,
        tts_speak: Mutex<std::collections::BTreeMap<u64, String>>,
    }
    impl Tap {
        /// Distinct emitted TtsSpeak texts, in id order.
        fn tts_speak_texts(&self) -> Vec<String> {
            self.tts_speak.lock().unwrap().values().cloned().collect()
        }
    }
    #[async_trait]
    impl FrameObserver for Tap {
        async fn on_process(&self, e: &FrameEvent<'_>) {
            self.names.lock().unwrap().push(e.frame.name());
            match &e.frame {
                Frame::OutputAudio(_) => self.saw_output_audio.store(true, Ordering::Relaxed),
                Frame::TtsSpeak { text, .. } => {
                    self.tts_speak
                        .lock()
                        .unwrap()
                        .insert(e.meta.id, text.clone());
                }
                _ => {}
            }
        }
    }

    /// The §9 step-10 gate, cascaded form: mock-STT → mock-LLM → mock-TTS runs a
    /// full turn end-to-end through a **real** [`PipelineTask`] built by
    /// [`build_cascaded_pipeline`] — caller audio → transcription → LLM text → TTS
    /// audio → output. Provider tests reuse this fixture against real providers.
    #[tokio::test]
    async fn cascaded_pipeline_runs_a_turn_end_to_end() {
        let tap = Arc::new(Tap::default());
        let task = build_cascaded_task_with_observers(
            MockStt::new("book a dentist appointment"),
            MockLlm::new("you said: "),
            MockTts::new(24_000),
            CascadedConfig::default(),
            vec![tap.clone() as Arc<dyn FrameObserver>],
        );

        // Drive a turn: feed one InputAudio chunk, then end.
        let audio = Arc::new(AudioFrame::mono(vec![1, 2, 3, 4], 16_000));
        task.queue_frame(Frame::InputAudio(audio)).await;
        task.stop_when_done().await;

        tokio::time::timeout(Duration::from_secs(5), task.run())
            .await
            .expect("cascaded pipeline timed out")
            .expect("run ok");

        // The turn produced output audio (STT→user-agg→LLM→assistant-agg→TTS fired).
        let names = tap.names.lock().unwrap().clone();
        assert!(
            tap.saw_output_audio.load(Ordering::Relaxed),
            "cascaded pipeline must emit OutputAudio; saw {names:?}"
        );
        assert!(
            names.contains(&"Transcription"),
            "STT must emit Transcription"
        );
        assert!(
            names.contains(&"LlmContext"),
            "user aggregator must emit LlmContext"
        );
        assert!(names.contains(&"LlmText"), "LLM must emit LlmText");
        assert!(
            names.contains(&"TtsSpeak"),
            "assistant aggregator must emit TtsSpeak"
        );

        // The assistant aggregator assembled the whole reply into one TtsSpeak.
        let speaks = tap.tts_speak_texts();
        assert_eq!(
            speaks,
            vec!["you said: book a dentist appointment".to_string()]
        );
    }

    /// The kickoff makes the bot **greet on connect**: a `ClientConnected` with NO
    /// user audio still runs the LLM once over the seeded system prompt and produces
    /// a spoken greeting (TtsSpeak → OutputAudio). Idempotent — a second
    /// `ClientConnected` does not greet again. (Closes the cascaded "silent until the
    /// user speaks" gap; the realtime path's `kickoff` analogue.)
    #[tokio::test]
    async fn kickoff_greets_on_connect_once() {
        let tap = Arc::new(Tap::default());
        // The context is seeded exactly as build_cascaded_task seeds it: the brain's
        // opening system prompt + start-node tools, with empty history.
        let ctx: SharedContext = Arc::new(Mutex::new(RollingContext::new(
            Some("Greet the caller.".into()),
            vec![],
        )));
        let pipeline = Pipeline::new(vec![
            Box::new(CascadedKickoffProcessor::new(ctx.clone())),
            Box::new(LlmProcessor::new(MockLlm::new("reply: "))),
            Box::new(AssistantContextAggregator::new(ctx.clone())),
            Box::new(TtsProcessor::new(MockTts::new(24_000))),
        ]);
        let task = PipelineTask::new(
            pipeline,
            PipelineTaskParams {
                idle_timeout: None,
                ..Default::default()
            },
            vec![tap.clone() as Arc<dyn FrameObserver>],
        );
        // Two ClientConnected, no user input at all.
        task.queue_frame(Frame::ClientConnected).await;
        task.queue_frame(Frame::ClientConnected).await;
        task.stop_when_done().await;
        tokio::time::timeout(Duration::from_secs(5), task.run())
            .await
            .expect("kickoff pipeline timed out")
            .expect("run ok");

        let names = tap.names.lock().unwrap().clone();
        // Greeted with NO user turn: output audio produced, but no Transcription.
        assert!(
            tap.saw_output_audio.load(Ordering::Relaxed),
            "kickoff must produce a greeting OutputAudio; saw {names:?}"
        );
        assert!(
            !names.contains(&"Transcription"),
            "the greeting must require no user input; saw {names:?}"
        );
        // Idempotent: exactly one greeting utterance (the MockLlm echoes the system
        // prompt as the last message's content).
        let speaks = tap.tts_speak_texts();
        assert_eq!(
            speaks,
            vec!["reply: Greet the caller.".to_string()],
            "greet exactly once"
        );
    }

    /// The kickoff greeting must be **tool-free**: even though the context is seeded
    /// with tools (transitions/endCall), the emitted greeting `LlmContext` carries NO
    /// tools — so the opening turn can't call `endCall`/a transition and hang up before
    /// the bot speaks (the live "AI can't speak back" regression).
    #[tokio::test]
    async fn kickoff_greeting_is_tool_free() {
        let tap = Arc::new(CtxTap::default());
        let ctx: SharedContext = Arc::new(Mutex::new(RollingContext::new(
            Some("Greet the caller.".into()),
            vec![
                json!({"name": "end_call", "description": "end the call", "params": {"type":"object"}}),
            ],
        )));
        let pipeline = Pipeline::new(vec![Box::new(CascadedKickoffProcessor::new(ctx.clone()))]);
        let task = PipelineTask::new(
            pipeline,
            PipelineTaskParams {
                idle_timeout: None,
                ..Default::default()
            },
            vec![tap.clone() as Arc<dyn FrameObserver>],
        );
        task.queue_frame(Frame::ClientConnected).await;
        task.stop_when_done().await;
        tokio::time::timeout(Duration::from_secs(5), task.run())
            .await
            .expect("timed out")
            .expect("run ok");
        let ctxs = tap.contexts.lock().unwrap().clone();
        assert_eq!(ctxs.len(), 1, "kickoff emits exactly one greeting context");
        assert!(
            ctxs[0].tools.is_empty(),
            "greeting context must carry NO tools"
        );
        assert_eq!(
            ctxs[0].messages[0]["role"], "system",
            "system prompt still present so it can greet"
        );
    }

    /// TurnMute locks at turn start (SttMute(true)) and unmutes once the bot reply has
    /// played out. A second `begin()` while muted is a no-op (the lock holds the whole
    /// turn — covering the LLM-thinking gap, not just while the bot speaks).
    #[tokio::test]
    async fn turn_mute_locks_then_unmutes_after_playout() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Frame>();
        let tm = TurnMute::new(tx, Duration::from_secs(5)); // long safety timeout (not used here)
        tm.begin();
        assert!(
            matches!(rx.recv().await, Some(Frame::SttMute(true))),
            "turn start mutes STT"
        );
        tm.begin(); // already muted → no-op (no second SttMute)
                    // A short bot reply: 100 ms of carrier audio (800 samples @ 8 kHz).
        tm.note_bot_audio(800, 8000);
        let unmute = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("unmute did not arrive");
        assert!(
            matches!(unmute, Some(Frame::SttMute(false))),
            "unmutes after playout drains"
        );
    }

    /// A turn that produces NO bot audio (tool-only / no reply) still unmutes via the
    /// safety timeout — the lock never deadlocks.
    #[tokio::test]
    async fn turn_mute_unmutes_via_safety_timeout_when_no_audio() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Frame>();
        let tm = TurnMute::new(tx, Duration::from_millis(200)); // short safety timeout
        tm.begin();
        assert!(matches!(rx.recv().await, Some(Frame::SttMute(true))));
        // No note_bot_audio at all → unmute fires via the timeout.
        let unmute = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("safety-timeout unmute did not arrive");
        assert!(matches!(unmute, Some(Frame::SttMute(false))));
    }

    // ---- Context aggregator unit tests --------------------------------------

    /// A tap that captures the LlmContext snapshots the user aggregator emits.
    #[derive(Default)]
    struct CtxTap {
        contexts: Mutex<Vec<LlmContext>>,
    }
    #[async_trait]
    impl FrameObserver for CtxTap {
        async fn on_process(&self, e: &FrameEvent<'_>) {
            if let Frame::LlmContext(c) = &e.frame {
                self.contexts.lock().unwrap().push((**c).clone());
            }
        }
    }

    /// The user aggregator accrues each final transcription into the rolling context
    /// and emits a growing `LlmContext` per turn (transcription accrual).
    #[tokio::test]
    async fn user_aggregator_accrues_transcription_into_context() {
        let tap = Arc::new(CtxTap::default());
        let ctx: SharedContext = Arc::new(Mutex::new(RollingContext::new(
            Some("be helpful".into()),
            vec![],
        )));
        let pipeline = Pipeline::new(vec![Box::new(UserContextAggregator::new(
            ctx.clone(),
            Arc::new(NoopSummarizer),
            SummarizerConfig::default(),
        ))]);
        let task = PipelineTask::new(
            pipeline,
            PipelineTaskParams {
                idle_timeout: None,
                ..Default::default()
            },
            vec![tap.clone() as Arc<dyn FrameObserver>],
        );
        let uid: Arc<str> = Arc::from("user");
        task.queue_frame(Frame::Transcription {
            text: "hello".into(),
            user_id: uid.clone(),
            language: None,
            final_: true,
        })
        .await;
        // An interim transcription must NOT trigger a run.
        task.queue_frame(Frame::InterimTranscription {
            text: "wor".into(),
            user_id: uid.clone(),
            language: None,
        })
        .await;
        task.queue_frame(Frame::Transcription {
            text: "world".into(),
            user_id: uid.clone(),
            language: None,
            final_: true,
        })
        .await;
        task.stop_when_done().await;
        tokio::time::timeout(Duration::from_secs(5), task.run())
            .await
            .expect("timed out")
            .expect("run ok");

        let contexts = tap.contexts.lock().unwrap().clone();
        assert_eq!(contexts.len(), 2, "one LlmContext per final transcription");
        // First run: system + "hello".
        assert_eq!(contexts[0].messages.len(), 2);
        assert_eq!(contexts[0].messages[0]["role"], "system");
        assert_eq!(contexts[0].messages[1]["content"], "hello");
        // Second run accrues: system + "hello" + "world".
        assert_eq!(contexts[1].messages.len(), 3);
        assert_eq!(contexts[1].messages[2]["content"], "world");
    }

    /// A tap that captures the TtsSpeak texts the assistant aggregator emits.
    #[derive(Default)]
    struct SpeakTap {
        texts: Mutex<Vec<String>>,
    }
    #[async_trait]
    impl FrameObserver for SpeakTap {
        async fn on_process(&self, e: &FrameEvent<'_>) {
            if let Frame::TtsSpeak { text, .. } = &e.frame {
                self.texts.lock().unwrap().push(text.clone());
            }
        }
    }

    /// The assistant aggregator accrues streamed `LlmText` between the start/end
    /// framing into one `assistant` message + a single `TtsSpeak` (response accrual).
    #[tokio::test]
    async fn assistant_aggregator_accrues_response_into_one_utterance() {
        let tap = Arc::new(SpeakTap::default());
        let ctx: SharedContext = Arc::new(Mutex::new(RollingContext::default()));
        let pipeline = Pipeline::new(vec![Box::new(AssistantContextAggregator::new(ctx.clone()))]);
        let task = PipelineTask::new(
            pipeline,
            PipelineTaskParams {
                idle_timeout: None,
                ..Default::default()
            },
            vec![tap.clone() as Arc<dyn FrameObserver>],
        );
        // Stream a response in three tokens.
        task.queue_frame(Frame::LlmResponseStart).await;
        task.queue_frame(Frame::LlmText("Hello ".into())).await;
        task.queue_frame(Frame::LlmText("there, ".into())).await;
        task.queue_frame(Frame::LlmText("friend".into())).await;
        task.queue_frame(Frame::LlmResponseEnd).await;
        task.stop_when_done().await;
        tokio::time::timeout(Duration::from_secs(5), task.run())
            .await
            .expect("timed out")
            .expect("run ok");

        // One assembled utterance.
        let texts = tap.texts.lock().unwrap().clone();
        assert_eq!(texts, vec!["Hello there, friend".to_string()]);
        // And it was appended to the rolling context as an assistant message.
        let c = ctx.lock().unwrap();
        assert_eq!(c.messages.len(), 1);
        assert_eq!(c.messages[0]["role"], "assistant");
        assert_eq!(c.messages[0]["content"], "Hello there, friend");
    }

    /// A summarizer that records it fired and returns a fixed summary.
    struct CountingSummarizer {
        fired: Arc<AtomicUsize>,
    }
    #[async_trait]
    impl ContextSummarizer for CountingSummarizer {
        async fn summarize(&self, _messages: &[Value]) -> Option<String> {
            self.fired.fetch_add(1, Ordering::SeqCst);
            Some("compressed".into())
        }
    }

    /// The summarizer hook fires (fire-and-forget) once the rolling context crosses
    /// the threshold, and its result collapses the history (summarizer fire).
    #[tokio::test]
    async fn summarizer_fires_once_past_threshold_and_compresses() {
        let fired = Arc::new(AtomicUsize::new(0));
        let ctx: SharedContext = Arc::new(Mutex::new(RollingContext::default()));
        let pipeline = Pipeline::new(vec![Box::new(UserContextAggregator::new(
            ctx.clone(),
            Arc::new(CountingSummarizer {
                fired: fired.clone(),
            }),
            // Fire after just 3 messages so the test is quick.
            SummarizerConfig {
                trigger_after_messages: 3,
            },
        ))]);
        let task = PipelineTask::new(
            pipeline,
            PipelineTaskParams {
                idle_timeout: None,
                ..Default::default()
            },
            vec![],
        );
        let uid: Arc<str> = Arc::from("user");
        for i in 0..4 {
            task.queue_frame(Frame::Transcription {
                text: format!("msg {i}"),
                user_id: uid.clone(),
                language: None,
                final_: true,
            })
            .await;
        }
        task.stop_when_done().await;
        tokio::time::timeout(Duration::from_secs(5), task.run())
            .await
            .expect("timed out")
            .expect("run ok");

        // Give the fire-and-forget summarize task a beat to apply.
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            fired.load(Ordering::SeqCst) >= 1,
            "summarizer must fire once past the threshold"
        );
        // The history was compressed to the single summary message.
        let c = ctx.lock().unwrap();
        assert_eq!(c.messages.len(), 1, "history collapsed to the summary");
        assert!(c.messages[0]["content"]
            .as_str()
            .unwrap()
            .contains("compressed"));
    }

    // =======================================================================
    // `build_cascaded_task` turn fixture.
    //
    // Drives `build_cascaded_task` with a MOCK transport + mock STT/LLM/TTS +
    // mock brain/session through a full turn and asserts the turn flows end to
    // end, a cascaded-LLM tool-call drives a brain **transition** (not a stay),
    // and `session.complete` fires once with the expected finalize payload. No
    // live key needed.
    // =======================================================================

    use crate::pipeline::s2s_test_mocks::{Captured, MockSession, MockSocket, CARRIER_RATE};
    use crate::processor::frame::FunctionCall;
    use crate::serializer::PlivoSerializer;
    use crate::transport::WsCarrierTransport;
    use crate::types::{BrainAction, ToolDecl, WsOut};
    use serde_json::json;

    const TRANSITION_TOOL: &str = "go_to_booking";
    const END_TOOL: &str = "end_call";

    /// A brain that **transitions** on `TRANSITION_TOOL` (the cascaded gate's key
    /// assertion: a tool-call moves the graph, not just a `Stay`) and **ends** on
    /// `END_TOOL`. Records every tool-call name + the actions it returned.
    struct TransitioningMockBrain {
        node_id: Arc<Mutex<String>>,
        seen_tools: Arc<Mutex<Vec<String>>>,
        actions: Arc<Mutex<Vec<&'static str>>>,
    }
    impl AgentBrain for TransitioningMockBrain {
        fn system_prompt(&self) -> String {
            "You are a test agent (start node).".into()
        }
        fn tools(&self) -> Vec<ToolDecl> {
            vec![ToolDecl {
                name: TRANSITION_TOOL.into(),
                description: "Move to the booking node.".into(),
                params: json!({ "type": "object", "properties": {} }),
            }]
        }
        fn current_node_id(&self) -> String {
            self.node_id.lock().unwrap().clone()
        }
        fn current_node_name(&self) -> String {
            // A display name distinct from BOTH the node id ("booking") and the
            // transition slug ("go_to_booking"), so the transcript-marker test
            // proves the marker uses the display name, not either of those.
            if *self.node_id.lock().unwrap() == "booking" {
                "Booking Node".into()
            } else {
                "Start Node".into()
            }
        }
        fn on_tool_call(&mut self, name: &str, _args: &serde_json::Value) -> BrainAction {
            self.seen_tools.lock().unwrap().push(name.to_string());
            if name == END_TOOL {
                self.actions.lock().unwrap().push("end");
                BrainAction::End {
                    disposition: Some("booked".into()),
                }
            } else if name == TRANSITION_TOOL && *self.node_id.lock().unwrap() == "start" {
                // Move the active node so the post-transition tool set + node_id
                // reflect the booking node (the relay scopes by node_id). Idempotent:
                // a repeat transition-call once already at booking is a no-op `Stay`.
                *self.node_id.lock().unwrap() = "booking".into();
                self.actions.lock().unwrap().push("transition");
                BrainAction::Transition {
                    system_prompt: "You are at the booking node.".into(),
                    tools: vec![ToolDecl {
                        name: END_TOOL.into(),
                        description: "End the call.".into(),
                        params: json!({ "type": "object", "properties": {} }),
                    }],
                    say: Some("Sure, let's get you booked.".into()),
                }
            } else {
                self.actions.lock().unwrap().push("stay");
                BrainAction::Stay
            }
        }
        fn is_finished(&self) -> bool {
            false
        }
        fn collected_vars(&self) -> serde_json::Value {
            json!({ "name": "Ada", "intent": "booking" })
        }
    }

    /// A scripted cascaded LLM driven by the brain's **current node** (a shared
    /// `node_id`, so the script is deterministic no matter how many user turns the
    /// mock socket produces):
    /// - while at the **start** node: emit *only* a
    ///   `FunctionCallsStarted([TRANSITION_TOOL])` (no spoken text) — a tool-call the
    ///   bridge feeds to the brain to drive a **transition** to the booking node.
    /// - while at the **booking** node (after the transition re-prompt re-runs the
    ///   LLM): speak the booking reply, then **after `LlmResponseEnd`** emit a
    ///   `FunctionCallsStarted([END_TOOL])` to end the call. Emitting the end-call
    ///   *after* the response framing means the assistant aggregator has already
    ///   produced the `TtsSpeak` (→TTS→`OutputAudio`, downstream toward the carrier)
    ///   before the brain's terminal `End` is injected at the head — so the reply is
    ///   spoken before teardown (the cascaded analogue of the s2s script emitting
    ///   `AudioOut` before the end tool-call).
    struct ScriptedCascadedLlm {
        node_id: Arc<Mutex<String>>,
        runs: Arc<AtomicUsize>,
        booking_reply: String,
    }
    #[async_trait]
    impl LlmService for ScriptedCascadedLlm {
        fn name(&self) -> &str {
            "ScriptedCascadedLlm"
        }
        async fn start(&mut self, _params: &StartParams) -> Result<()> {
            Ok(())
        }
        async fn run_llm<'a>(
            &'a mut self,
            _ctx: &'a LlmContext,
        ) -> Result<futures::stream::BoxStream<'a, Frame>> {
            use futures::StreamExt;
            self.runs.fetch_add(1, Ordering::SeqCst);
            let at_start = *self.node_id.lock().unwrap() == "start";
            let frames: Vec<Frame> = if at_start {
                // Start node: a transition tool-call, no spoken text.
                vec![
                    Frame::LlmResponseStart,
                    Frame::FunctionCallsStarted(vec![FunctionCall {
                        function_name: TRANSITION_TOOL.into(),
                        tool_call_id: "fc-tr-1".into(),
                        arguments: json!({}),
                    }]),
                    Frame::LlmResponseEnd,
                ]
            } else {
                // Booking node: speak the reply, THEN (after the response framing
                // flushed the TtsSpeak downstream) emit the end tool-call.
                vec![
                    Frame::LlmResponseStart,
                    Frame::LlmText(self.booking_reply.clone()),
                    Frame::LlmResponseEnd,
                    Frame::FunctionCallsStarted(vec![FunctionCall {
                        function_name: END_TOOL.into(),
                        tool_call_id: "fc-end-1".into(),
                        arguments: json!({}),
                    }]),
                ]
            };
            Ok(futures::stream::iter(frames).boxed())
        }
        fn set_tools(&mut self, _tools: Vec<crate::service::Tool>) {}
    }

    /// Extract the `playAudio` carrier frames (the bot audio that reached the
    /// carrier) from the captured outbound socket frames.
    fn play_audio_frames(sent: &[WsOut]) -> Vec<String> {
        sent.iter()
            .filter_map(|o| match o {
                WsOut::Text(t) if t.contains("playAudio") => Some(t.clone()),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn cascaded_task_runs_a_turn_with_a_transition_and_finalizes() {
        let sent = Arc::new(Mutex::new(Vec::<WsOut>::new()));
        let captured = Arc::new(Mutex::new(Captured::default()));
        let seen_tools = Arc::new(Mutex::new(Vec::<String>::new()));
        let actions = Arc::new(Mutex::new(Vec::<&'static str>::new()));
        let runs = Arc::new(AtomicUsize::new(0));

        let transport = WsCarrierTransport::new(
            MockSocket::new(sent.clone()),
            PlivoSerializer::new(CARRIER_RATE),
        );
        // The brain + the scripted LLM share the active node id — the LLM follows
        // the (re-prompted) node exactly as a real cascaded LLM follows its prompt.
        let node_id = Arc::new(Mutex::new(String::from("start")));
        let brain = TransitioningMockBrain {
            node_id: node_id.clone(),
            seen_tools: seen_tools.clone(),
            actions: actions.clone(),
        };
        let task = build_cascaded_task(
            transport,
            MockStt::new("book a dentist appointment"),
            ScriptedCascadedLlm {
                node_id: node_id.clone(),
                runs: runs.clone(),
                // Long enough that MockTts (one PCM sample per char) yields ≥ the
                // resampler's 256-sample block, so the 24k→8k carrier audio is
                // non-empty and actually reaches the socket as a `playAudio` frame.
                booking_reply: "Sure, let's get you booked in. ".repeat(12),
            },
            MockTts::new(24_000),
            brain,
            MockSession::new(captured.clone()),
            7777,
            "tok-cascaded".into(),
            CascadedConfig::default(),
        )
        .await
        .expect("build_cascaded_task");

        tokio::time::timeout(Duration::from_secs(5), task.run())
            .await
            .expect("cascaded task timed out")
            .expect("cascaded task errored");

        // --- (1) The turn flowed end-to-end: bot audio reached the carrier. ------
        let play = play_audio_frames(&sent.lock().unwrap());
        assert!(
            !play.is_empty(),
            "expected ≥1 playAudio frame to the carrier (STT→LLM→brain→TTS→output)"
        );

        // --- (2) A cascaded-LLM tool-call drove a brain TRANSITION (not a stay). --
        let seen = seen_tools.lock().unwrap().clone();
        assert!(
            seen.contains(&TRANSITION_TOOL.to_string()),
            "the transition tool-call must reach the brain; saw {seen:?}"
        );
        let acts = actions.lock().unwrap().clone();
        assert!(
            acts.contains(&"transition"),
            "a tool-call must drive a TRANSITION, not just a stay; actions={acts:?}"
        );
        assert!(
            acts.contains(&"end"),
            "the end tool-call must end the call; actions={acts:?}"
        );
        // Exactly one real transition (the brain is idempotent once at booking).
        assert_eq!(
            acts.iter().filter(|a| **a == "transition").count(),
            1,
            "exactly one graph transition; actions={acts:?}"
        );
        // ≥2 LLM runs fired: the start-node turn (transition) + the post-transition
        // re-prompt re-run that produces the spoken booking reply.
        assert!(
            runs.load(Ordering::SeqCst) >= 2,
            "the transition must re-run the LLM (≥2 runs); got {}",
            runs.load(Ordering::SeqCst)
        );

        // --- (3) session.complete fired ONCE with the expected finalize payload. -
        let c = captured.lock().unwrap();
        assert_eq!(
            c.complete_calls, 1,
            "session.complete must fire exactly once"
        );
        let fin = c.finalize.as_ref().expect("finalize payload present");
        // Recording + transcript stored keys (the run-scoped keys, never the
        // secret-bearing presigned URL).
        assert_eq!(fin.recording_url.as_deref(), Some("runs/4242/recording"));
        assert_eq!(fin.transcript_url.as_deref(), Some("runs/4242/transcript"));
        // collected_vars from the brain + the folded disposition from the End.
        assert_eq!(fin.collected_vars["name"], "Ada");
        assert_eq!(fin.collected_vars["intent"], "booking");
        assert_eq!(
            fin.collected_vars["call_disposition"], "booked",
            "the End disposition must fold into collected_vars"
        );
        // Usage is present in the finalize payload (default totals — no provider
        // usage wired into the mock LLM, but the field is always serialized).
        assert!(fin.usage.is_object(), "usage must be a JSON object");
        // Wall-clock duration rides in usage_metrics — the control plane reads
        // `usage_metrics.duration_seconds` for the Interactions duration/cost.
        assert!(
            fin.usage["duration_seconds"].is_number(),
            "finalize usage must carry duration_seconds"
        );

        // --- (4) the transcript transition marker uses the destination node's
        // DISPLAY NAME, not the transition slug — so the stored/historical view
        // matches the live rail (regression guard for the `transition_0` bug).
        let transcript = c
            .transcript_body
            .as_deref()
            .expect("transcript artifact was uploaded");
        assert!(
            transcript.contains("[transition: Booking Node]"),
            "marker must carry the node display name; got: {transcript}"
        );
        assert!(
            !transcript.contains("go_to_booking"),
            "marker must NOT carry the transition slug; got: {transcript}"
        );
    }

    // ---- is_transition_ack: don't swallow MCP results that carry a `status` -----
    // Regression guard for the cascaded counterpart of `s2s::mcp_result_struct`:
    // MCP tool results are now JSON objects, and an object with `{"status": "ok"}`
    // must still be fed back to the LLM, not misread as a `{status: moved}` ack.

    #[test]
    fn is_transition_ack_matches_only_brain_vocabulary() {
        // The brain's three real ACK envelopes:
        assert!(is_transition_ack(&json!({ "status": "moved" })));
        assert!(is_transition_ack(&json!({ "status": "stay" })));
        assert!(is_transition_ack(&json!({ "status": "ended" })));

        // MCP/workflow results that happen to carry a `status` are NOT acks:
        assert!(!is_transition_ack(&json!({ "status": "ok" })));
        assert!(!is_transition_ack(
            &json!({ "status": "confirmed", "booking_id": "B1" })
        ));
        // …nor a tool result with no status, nor a non-object/non-string status:
        assert!(!is_transition_ack(&json!({ "booking_id": "B1" })));
        assert!(!is_transition_ack(&json!({ "status": 200 })));
        assert!(!is_transition_ack(&json!("moved")));
    }
}
