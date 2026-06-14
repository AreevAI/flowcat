// SPDX-License-Identifier: Apache-2.0
//
//! The live Gemini-Live **S2S path as a processor pipeline** — the realtime call
//! orchestration (PROCESSOR-DESIGN §7). Assembled by [`build_s2s_task`].
//!
//! Each of the four trait seams becomes a [`FrameProcessor`]:
//!
//! ```text
//!   [transport pump] → TransportInput → RealtimeServiceProcessor → BrainProcessor → TransportOutput
//!                                              ▲                        │
//!                                        (tool results          (RecorderProcessor taps both legs)
//!                                         upstream)              (TranscriptProcessor taps text)
//!                                                                (FinalizeProcessor on End)
//! ```
//!
//! Every conversation event maps to a `process_frame` (PROCESSOR-DESIGN §7.1):
//! - carrier audio in → [`TransportInput`] emits `InputAudio` → the realtime
//!   service `send_audio`s it (after carrier→16k resample);
//! - `RealtimeEvent::AudioOut` → the realtime service emits `OutputAudio`(24k) →
//!   [`TransportOutput`] resamples 24k→carrier + plays;
//! - a workflow (MCP) `ToolCall` → [`BrainProcessor`] relays via [`ToolRelay`] and
//!   pushes a `FunctionCallResult` straight back upstream (no transition);
//! - a transition/end `ToolCall` → [`BrainProcessor`] pushes `UpdateSettings`
//!   (new prompt+tools → `update_system`) / an upstream `End`;
//! - `RealtimeEvent::Interrupted` → an `Interruption` broadcast → `TransportOutput`
//!   clears;
//! - `Usage` → folded into the shared [`LiveState`];
//! - terminal `End` → [`FinalizeProcessor`] runs the artifact-upload + `complete`
//!   (the `LiveState`/`finalize` logic).
//!
//! **Generic over the flowcat-core seams** — `T: MediaTransport`, `R: RealtimeLlm`,
//! `B: AgentBrain`, `S: SessionSource` — so flowcat-core stays embedder-agnostic.
//! The host wires the concrete brain/session implementations in.
//!
//! ## A note on the transport *pump* (the codified source-emit pattern)
//!
//! A pure *source* processor cannot self-emit frames from the framework's frozen
//! [`FrameProcessor::start`] hook (it receives no [`Link`]). Per the
//! source-emit ruling (PROCESSOR-DESIGN §10 Q6), the framework **codifies the
//! external-pump pattern** as the standard via the [`SourcePump`] helper rather than
//! growing the just-frozen trait: the transport's `recv()` loop runs in a reader task
//! that `emit`s `InputAudio`/`StreamStart`/`Stop` into the pipeline **head** — exactly
//! pipecat's `BaseInputTransport` reader-task model, hosted beside the
//! [`PipelineTask`]. Feeding the head (not a mid-chain `push`) is what preserves the
//! Start→ready handshake: queued frames can't reach any `process_frame` before every
//! processor's `start()` ran. [`TransportInput`] is then the head marker stage the
//! pump feeds. Every *downstream* seam (realtime/brain/output) reacts in
//! `process_frame` (where the `Link` exists) and lazily spawns its own reader task on
//! first frame — the pattern §6.1 describes.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::json;
use tokio::time::Instant;

use crate::audio::AudioRecorder;
use crate::brain::AgentBrain;
use crate::codec::Resampler;
use crate::error::{FlowcatError, Result};
use crate::processor::frame::{AudioFrame, Frame, FrameClass, StartParams};
use crate::processor::{Envelope, FrameProcessor, Link, ProcessorSetup, StopReason};
use crate::realtime::{RealtimeKickoff, RealtimeLlm};
use crate::session::SessionSource;
use crate::transcript::Transcript;
use crate::transport::{MediaIn, MediaTransport};
use crate::types::{AudioChunk, BrainAction, Finalize, RealtimeEvent, ToolDecl, Usage};

use super::{Pipeline, PipelineTask, PipelineTaskParams, SourceHandle, SourcePump};

/// Gemini Live's output sample rate (PCM). Bot audio arrives at this rate
/// (mirrors `call.rs::GEMINI_OUTPUT_RATE`). The *input* rate is no longer a constant
/// here — it comes from `RealtimeLlm::input_sample_rate()` (16 kHz Gemini / 24 kHz
/// OpenAI) so the model's required rate drives both the session config and the
/// resampler.
const GEMINI_OUTPUT_RATE: u32 = 24_000;

// ===========================================================================
// Custom frames — the S2S-specific signals that ride the Frame::Custom hatch.
//
// These are not in the v1 named-variant set (PROCESSOR-DESIGN §1.2 keeps the
// long tail in Custom). The S2S path needs to
// carry a *tool call from the realtime model* downstream to the brain and the
// brain's *tool result* back upstream to the realtime service; we model both with
// Custom frames so flowcat-core gains no new named variant for this private path.
// ===========================================================================

/// A tool/function call the realtime model emitted (model → brain, downstream).
/// Carried in [`Frame::Custom`]. The [`BrainProcessor`] consumes it.
///
/// `pub(crate)` (constructor + fields) so the cascaded task assembler
/// ([`super::cascaded_task`]) can feed the *cascaded LLM's* function-calls to the
/// **same** [`BrainProcessor`] via this frame — the integration that lets the
/// cascaded path reuse the realtime path's brain/transition wiring verbatim. The
/// realtime path is unchanged.
#[derive(Debug, Clone)]
pub(crate) struct ModelToolCall {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) args: serde_json::Value,
}

impl crate::processor::frame::CustomFrame for ModelToolCall {
    fn frame_class(&self) -> FrameClass {
        // System: tool calls are control signals that must reach the brain even
        // through a backlog (mirrors `RealtimeEvent::ToolCall` jumping the select).
        FrameClass::System
    }
    fn name(&self) -> &'static str {
        "ModelToolCall"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// A tool result the brain produced (brain → realtime service, upstream). Carried
/// in [`Frame::Custom`]. The [`RealtimeServiceProcessor`] consumes it and calls
/// `send_tool_result`. Uninterruptible (a produced result must always be delivered).
///
/// `pub(crate)` so the cascaded task ([`super::cascaded_task`]) can consume the
/// **same** brain output upstream of the cascaded LLM (feeding an MCP result back
/// into the rolling LLM context). The realtime path is unchanged.
#[derive(Debug, Clone)]
pub(crate) struct ToolResult {
    pub(crate) id: String,
    pub(crate) result: serde_json::Value,
}

impl crate::processor::frame::CustomFrame for ToolResult {
    fn frame_class(&self) -> FrameClass {
        FrameClass::System
    }
    fn uninterruptible(&self) -> bool {
        true
    }
    fn name(&self) -> &'static str {
        "ToolResult"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// A re-prompt instruction the brain produced on a transition (brain → realtime
/// service, upstream). Carried in [`Frame::Custom`]. The
/// [`RealtimeServiceProcessor`] consumes it and calls `update_system`.
///
/// `pub(crate)` so the cascaded task ([`super::cascaded_task`]) can consume the
/// **same** transition re-prompt upstream of the cascaded LLM (swapping the rolling
/// context's system prompt + tools, then re-running the LLM). Realtime is unchanged.
#[derive(Debug, Clone)]
pub(crate) struct Reprompt {
    pub(crate) prompt: String,
    pub(crate) tools: Vec<ToolDecl>,
}

impl crate::processor::frame::CustomFrame for Reprompt {
    fn frame_class(&self) -> FrameClass {
        FrameClass::System
    }
    fn uninterruptible(&self) -> bool {
        true
    }
    fn name(&self) -> &'static str {
        "Reprompt"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ===========================================================================
// Shared live-call state — the faithful analogue of call.rs::LiveState.
//
// In `Call::run` one `LiveState` is owned by the single select! task. In the
// processor model the recorder/transcript/usage taps + finalize live in different
// processor tasks, so the state is shared behind an Arc<Mutex<…>> — the same data,
// accumulated by the same code paths, consumed by FinalizeProcessor on End.
// ===========================================================================

/// Per-call accumulated state, shared across the recorder/transcript/finalize
/// processors (recorder, transcript, usage accumulation, disposition +
/// collected_vars fold).
///
/// `pub(crate)` so the cascaded task ([`super::cascaded_task`]) reuses the **same**
/// shared-state type + recorder/transcript/finalize processors (no parallel impl).
pub(crate) struct LiveState {
    pub(crate) recorder: AudioRecorder,
    pub(crate) transcript: Transcript,
    pub(crate) disposition: Option<String>,
    pub(crate) collected_vars: serde_json::Value,
    usage: Usage,
    /// Count of REAL workflow tool calls relayed this call (not transitions) —
    /// reported as `usage_metrics.tool_calls` for the Run-detail "Tool Calls" metric.
    pub(crate) tool_calls: u64,
    error: Option<FlowcatError>,
    /// Set once a terminal transport error is seen (peer gone). De-dupes the log
    /// (warn once, not per buffered frame) and lets the sink stop sending / end the call.
    pub(crate) transport_dead: bool,
}

impl LiveState {
    pub(crate) fn new(carrier_rate: u32) -> Self {
        Self {
            recorder: AudioRecorder::new(carrier_rate),
            transcript: Transcript::new(),
            disposition: None,
            collected_vars: serde_json::Value::Null,
            usage: Usage::default(),
            tool_calls: 0,
            error: None,
            transport_dead: false,
        }
    }

    pub(crate) fn record_error(&mut self, e: FlowcatError) {
        // A terminal transport error repeats on every buffered frame — log it once.
        let is_transport = matches!(e, FlowcatError::Transport(_));
        if is_transport && self.transport_dead {
            return;
        }
        tracing::warn!(error = %e, "live-phase error (s2s pipeline)");
        if is_transport {
            self.transport_dead = true;
        }
        if self.error.is_none() {
            self.error = Some(e);
        }
    }

    /// Fold one provider [`Usage`] report into the running totals.
    fn accumulate_usage(&mut self, u: &Usage) {
        self.usage.input_tokens = add_opt(self.usage.input_tokens, u.input_tokens);
        self.usage.output_tokens = add_opt(self.usage.output_tokens, u.output_tokens);
        self.usage.total_tokens = add_opt(self.usage.total_tokens, u.total_tokens);
        if u.extra.is_some() {
            self.usage.extra = u.extra.clone();
        }
    }

    /// Serialize the accumulated provider [`Usage`] plus the call's wall-clock
    /// `duration_seconds` (from the recorder timeline). The control plane reads
    /// `usage_metrics.duration_seconds` for the Interactions duration column and
    /// the duration-derived cost/cost-per-minute, so it must ride in this object.
    fn usage_json(&self) -> serde_json::Value {
        let mut v = serde_json::to_value(&self.usage).unwrap_or(serde_json::Value::Null);
        if let serde_json::Value::Object(ref mut map) = v {
            map.entry("duration_seconds".to_string())
                .or_insert_with(|| serde_json::json!(self.recorder.duration_seconds()));
            // Run-detail observability counts (the control plane reads these off
            // usage_metrics for "User/Bot Turns" + "Tool Calls").
            map.entry("user_turns".to_string())
                .or_insert_with(|| serde_json::json!(self.transcript.user_turns()));
            map.entry("bot_turns".to_string())
                .or_insert_with(|| serde_json::json!(self.transcript.bot_turns()));
            map.entry("tool_calls".to_string())
                .or_insert_with(|| serde_json::json!(self.tool_calls));
        }
        v
    }
}

/// Sum two optional counts, treating `None` as "not reported" (so `None + x = x`).
fn add_opt(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a + b),
        (Some(a), None) => Some(a),
        (None, b) => b,
    }
}

/// Fold an optional `disposition` into the brain's collected-vars JSON.
/// `pub(crate)` for the cascaded finalize reuse.
pub(crate) fn merge_disposition(
    vars: serde_json::Value,
    disposition: Option<String>,
) -> serde_json::Value {
    match (vars, disposition) {
        (serde_json::Value::Object(mut map), Some(d)) => {
            // Key matches the control plane's DISPOSITION_VAR ("call_disposition")
            // — the embedder reads `collected_vars->>'call_disposition'` to record
            // the call's disposition.
            map.entry("call_disposition".to_string())
                .or_insert(serde_json::Value::String(d));
            serde_json::Value::Object(map)
        }
        (other, _) => other,
    }
}

pub(crate) type SharedState = Arc<Mutex<LiveState>>;

// ===========================================================================
// TransportInput — the head marker stage the transport pump feeds.
// ===========================================================================

/// The head/source-side stage (PROCESSOR-DESIGN §7.1). The transport's `recv()`
/// loop is driven by the external [`spawn_transport_pump`] (see the module note);
/// this processor is the named head the pump injects into and forwards from. It is
/// a thin pass-through so `InputAudio`/lifecycle flows on to the realtime service.
pub struct TransportInput {
    name: &'static str,
}

impl TransportInput {
    pub(crate) fn new() -> Self {
        Self {
            name: "TransportInput",
        }
    }
}

#[async_trait]
impl FrameProcessor for TransportInput {
    fn name(&self) -> &str {
        self.name
    }
    // Default `process_frame` forwards — the pump feeds InputAudio/StreamStart/Stop
    // at the head and they pass straight through to the realtime service.
}

/// A [`MediaTransport`] shared (behind an async mutex) between the pump's `recv`
/// loop and [`TransportOutput`]'s `send_audio`/`send_clear`. In `Call::run` the
/// single `select!` task owns the whole transport and interleaves recv/send on it;
/// the processor split needs both a reader and a writer task, so the one transport
/// object is shared here. `recv` mostly awaits new media, releasing the lock so a
/// concurrent `send_audio` is never starved.
pub(crate) struct SharedTransport<T: MediaTransport>(Arc<tokio::sync::Mutex<T>>);

impl<T: MediaTransport> Clone for SharedTransport<T> {
    fn clone(&self) -> Self {
        SharedTransport(self.0.clone())
    }
}

impl<T: MediaTransport> SharedTransport<T> {
    pub(crate) fn new(t: T) -> Self {
        SharedTransport(Arc::new(tokio::sync::Mutex::new(t)))
    }
}

#[async_trait]
impl<T: MediaTransport> MediaTransport for SharedTransport<T> {
    async fn recv(&mut self) -> Option<MediaIn> {
        self.0.lock().await.recv().await
    }
    async fn send_audio(&mut self, chunk: AudioChunk) -> std::result::Result<(), FlowcatError> {
        self.0.lock().await.send_audio(chunk).await
    }
    async fn send_clear(&mut self) -> std::result::Result<(), FlowcatError> {
        self.0.lock().await.send_clear().await
    }
    fn carrier_rate(&self) -> u32 {
        // carrier_rate is a const property; read it once up front instead (the
        // assembler captures it before sharing). This path is not on the hot loop.
        // We cannot `.await` here, so callers must not rely on it post-share — the
        // assembler passes the rate explicitly to each processor instead.
        0
    }
}

/// Spawn the transport's `recv()` pump via the standard [`SourcePump`] helper (the
/// codified source-emit pattern, PROCESSOR-DESIGN §10 Q6): drive
/// [`MediaTransport::recv`] and `emit` each event into the pipeline head, mirroring
/// `call.rs`'s carrier→model select arm. Emits a `ClientConnected` kickoff-gate
/// marker on stream start (the realtime service kicks off on it), `InputAudio` per
/// audio chunk, and a terminal `End` on `Stop`/exhaustion. The returned
/// [`SourcePump`] aborts the reader on drop.
pub(crate) fn spawn_transport_pump<T: MediaTransport + 'static>(
    mut transport: SharedTransport<T>,
    head: tokio::sync::mpsc::UnboundedSender<Frame>,
) -> SourcePump {
    SourcePump::spawn(head, move |h: SourceHandle| async move {
        loop {
            match transport.recv().await {
                Some(MediaIn::StreamStart { call_id }) => {
                    tracing::debug!(call_id, "media stream started (s2s pipeline)");
                    // ClientConnected = the kickoff gate (the realtime service's
                    // transport_started analogue). System frame, jumps the queue.
                    let _ = h.emit(Frame::ClientConnected);
                }
                Some(MediaIn::Audio(chunk)) => {
                    let frame = Frame::InputAudio(Arc::new(AudioFrame::from(&chunk)));
                    if h.emit(frame).is_err() {
                        break; // pipeline gone
                    }
                }
                // Carrier stop or transport exhausted → drain the pipeline.
                Some(MediaIn::Stop) | None => {
                    h.end();
                    break;
                }
            }
        }
    })
}

// ===========================================================================
// RealtimeServiceProcessor — wraps a RealtimeLlm.
// ===========================================================================

/// Wraps a [`RealtimeLlm`] (the existing Gemini client through the trait — never a
/// rewrite). On first `InputAudio` (after the stream-start kickoff gate) it lazily
/// spawns the `next_event` reader task (the existing reader-task→mpsc bridge made a
/// processor-internal task, PROCESSOR-DESIGN §6.1) which emits downstream
/// `OutputAudio`/transcription/tool-call/`Interruption`/usage frames and a terminal
/// `End` on `Closed`. It feeds each `InputAudio` to `send_audio`; consumes the
/// brain's upstream [`ToolResult`]→`send_tool_result` and [`Reprompt`]→`update_system`.
pub struct RealtimeServiceProcessor<R: RealtimeLlm + RealtimeKickoff> {
    /// The realtime session, shared with the lazily-spawned reader task.
    realtime: Arc<tokio::sync::Mutex<R>>,
    /// Carrier→model-input resampler (one per call, preserves filter state — mirrors
    /// `call.rs`'s long-lived `in_resampler`). Targets [`input_rate`](Self::input_rate).
    in_resampler: Option<Resampler>,
    carrier_rate: u32,
    /// The model's required input sample rate (Hz) — the resampler's target (16 kHz
    /// for Gemini, 24 kHz for OpenAI Realtime; from `RealtimeLlm::input_sample_rate`).
    input_rate: u32,
    /// Whether the carrier stream has started (the kickoff gate).
    transport_started: bool,
    /// Whether bot-first kickoff has fired (once per call).
    kicked_off: bool,
    /// Whether the reader task has been spawned (once).
    reader_spawned: bool,
    /// Shared live state (for error recording).
    state: SharedState,
    /// The pipeline-head queue, so a terminal model event (`Closed`/`None`) ends
    /// the call by injecting `End` at the **head** — Source converts it to a
    /// downstream drain through the *whole* chain (so Source/TransportInput break
    /// too, a clean fast teardown). A `push_down(End)` from mid-chain would leave
    /// the head processors lingering on the grace window (PROCESSOR-DESIGN §4.1).
    end_tx: tokio::sync::mpsc::UnboundedSender<Frame>,
    /// Abort handle for the spawned model-event reader task. The reader reconnects
    /// the realtime session on transient drops (Gemini `1008` aborts), so once
    /// the call ends it would keep reconnecting in the background until the reconnect
    /// ceiling — burning provider quota long after teardown. We abort it in
    /// [`stop`](FrameProcessor::stop) so the reader (and its realtime session) die
    /// with the call. `None` until the reader is spawned.
    reader_handle: Option<tokio::task::AbortHandle>,
}

impl<R: RealtimeLlm + RealtimeKickoff + 'static> RealtimeServiceProcessor<R> {
    fn new(
        realtime: R,
        carrier_rate: u32,
        input_rate: u32,
        state: SharedState,
        end_tx: tokio::sync::mpsc::UnboundedSender<Frame>,
    ) -> Self {
        Self {
            realtime: Arc::new(tokio::sync::Mutex::new(realtime)),
            in_resampler: None,
            carrier_rate,
            input_rate,
            transport_started: false,
            kicked_off: false,
            reader_spawned: false,
            state,
            end_tx,
            reader_handle: None,
        }
    }

    /// Spawn the model-event reader: pump `next_event` and emit the matching
    /// downstream frames, mirroring `call.rs`'s model→carrier select arm. Runs
    /// until `Closed`/`None`, then pushes a terminal `End` so the pipeline drains.
    fn spawn_reader(&mut self, link: Link) {
        if self.reader_spawned {
            return;
        }
        self.reader_spawned = true;
        let realtime = self.realtime.clone();
        let end_tx = self.end_tx.clone();
        let handle = tokio::spawn(async move {
            // Readiness notify (if the provider supports the lock-free path). With
            // it, we never hold the session lock across the idle wait between bot
            // turns — so `send_audio` (caller audio in) is never starved and the
            // call doesn't deadlock after the greeting. Without it (other
            // providers / mocks), fall back to the blocking `next_event`.
            let notify = { realtime.lock().await.event_notify() };
            // Accumulates streaming user-transcription deltas into one growing
            // interim line; cleared when the matching `UserText` finalizes it.
            let mut user_partial = String::new();
            // Accumulates the bot's response transcript deltas. Emitted as ONE
            // finalized line at the response boundary (`response.done` → Usage,
            // barge-in, or stream end) — never per delta — so an interleaved user
            // partial can't split the bot's sentence into several bubbles, and the
            // bot line is finalized (the UI's "speaking…" clears) because nothing
            // else produces `BotStoppedSpeaking` on the realtime path.
            let mut bot_partial = String::new();
            // Flush the accumulated bot line (if any) + signal the turn ended.
            macro_rules! flush_bot {
                () => {
                    if !bot_partial.is_empty() {
                        link.push_down(Frame::Transcription {
                            text: std::mem::take(&mut bot_partial),
                            user_id: Arc::from("bot"),
                            language: None,
                            final_: true,
                        })
                        .await;
                        link.push_down(Frame::BotStoppedSpeaking).await;
                    }
                };
            }
            loop {
                let event = if let Some(notify) = &notify {
                    loop {
                        let polled = {
                            let mut rt = realtime.lock().await;
                            rt.poll_event().await
                        };
                        match polled {
                            crate::realtime::PollEvent::Ready(ev) => break ev,
                            // No event ready: await readiness WITHOUT the lock.
                            crate::realtime::PollEvent::Pending => notify.notified().await,
                        }
                    }
                } else {
                    let mut rt = realtime.lock().await;
                    rt.next_event().await
                };
                let Some(event) = event else {
                    // Realtime stream ended → flush any pending bot line, then drain
                    // via the head (Source breaks too).
                    flush_bot!();
                    let _ = end_tx.send(Frame::End { reason: None });
                    break;
                };
                match event {
                    RealtimeEvent::AudioOut(chunk) => {
                        // 24k bot audio — TransportOutput resamples + records it.
                        let af = Arc::new(AudioFrame::from(&chunk));
                        link.push_down(Frame::OutputAudio(af)).await;
                    }
                    RealtimeEvent::UserInterimText(delta) => {
                        // A streaming partial: grow the interim line and push it as
                        // a non-final transcription so the UI updates one bubble in
                        // place (rather than one bubble per word).
                        user_partial.push_str(&delta);
                        link.push_down(Frame::InterimTranscription {
                            text: user_partial.clone(),
                            user_id: Arc::from("user"),
                            language: None,
                        })
                        .await;
                    }
                    RealtimeEvent::UserText(text) => {
                        // The finalized utterance (provider "completed" event) is
                        // authoritative — emit it as the committed line and reset
                        // the interim accumulator for the next turn.
                        user_partial.clear();
                        link.push_down(Frame::Transcription {
                            text,
                            user_id: Arc::from("user"),
                            language: None,
                            final_: true,
                        })
                        .await;
                    }
                    RealtimeEvent::BotText(text) => {
                        // Accumulate; the full line is emitted at the response
                        // boundary (see `flush_bot!`), not per delta.
                        bot_partial.push_str(&text);
                    }
                    RealtimeEvent::ToolCall { id, name, args } => {
                        link.push_down(Frame::Custom(Arc::new(ModelToolCall { id, name, args })))
                            .await;
                    }
                    RealtimeEvent::Interrupted => {
                        // Barge-in: the bot was cut off — finalize whatever it had
                        // said, then broadcast both directions; TransportOutput
                        // clears the carrier's playback (call.rs's send_clear arm).
                        flush_bot!();
                        link.broadcast(Frame::Interruption).await;
                    }
                    RealtimeEvent::Usage(u) => {
                        // `response.done` → the bot turn ended: flush its line first,
                        // then carry usage downstream as a Custom frame for the
                        // recorder tap to fold into the shared LiveState (call.rs
                        // accumulates it on its single LiveState).
                        flush_bot!();
                        link.push_down(Frame::Custom(Arc::new(UsageReport(u))))
                            .await;
                    }
                    RealtimeEvent::Closed => {
                        // Flush any pending bot line, then drain via the head so
                        // Source/TransportInput break too.
                        flush_bot!();
                        let _ = end_tx.send(Frame::End { reason: None });
                        break;
                    }
                }
            }
        });
        // Keep the abort handle so `stop()` can kill the reader at teardown —
        // otherwise it keeps reconnecting the realtime session after the call ends.
        self.reader_handle = Some(handle.abort_handle());
    }
}

/// A usage report riding the Custom hatch (realtime service → recorder tap).
/// `pub(crate)` so the cascaded LLM-usage tap can fold usage into the **same**
/// shared `LiveState` via the reused [`RecorderProcessor`].
#[derive(Debug, Clone)]
pub(crate) struct UsageReport(pub(crate) Usage);

impl crate::processor::frame::CustomFrame for UsageReport {
    fn frame_class(&self) -> FrameClass {
        FrameClass::Data
    }
    fn name(&self) -> &'static str {
        "UsageReport"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[async_trait]
impl<R: RealtimeLlm + RealtimeKickoff + 'static> FrameProcessor for RealtimeServiceProcessor<R> {
    fn name(&self) -> &str {
        "RealtimeService"
    }

    async fn start(&mut self, _setup: &ProcessorSetup, _params: &StartParams) -> Result<()> {
        // The realtime session was already `connect`ed by the assembler (mirroring
        // `Call::run` connecting before its loop). Build the carrier→input-rate
        // resampler (16k for Gemini, 24k for OpenAI — the model's required input).
        self.in_resampler = Some(Resampler::new(self.carrier_rate, self.input_rate)?);
        Ok(())
    }

    /// Teardown: abort the model-event reader so it stops reconnecting the realtime
    /// session once the call has ended. Without this the reader (which transparently
    /// reconnects on Gemini's `1008` aborts) keeps re-opening the session in the
    /// background until the reconnect ceiling — burning provider quota and a
    /// connection for minutes after every call. Aborting drops the reader's clone of
    /// the session `Arc`, so the realtime WS closes with the call.
    async fn stop(&mut self, _reason: StopReason) -> Result<()> {
        if let Some(handle) = self.reader_handle.take() {
            handle.abort();
        }
        Ok(())
    }

    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        match &env.frame {
            // Stream-start gate (call.rs's MediaIn::StreamStart arm): kickoff once
            // BOTH the transport stream is up and the model is connected. The model
            // was connected synchronously by the assembler, so this is the gate.
            Frame::ClientConnected => {
                self.transport_started = true;
                // Lazily spawn the model-event reader now we have a Link.
                self.spawn_reader(link.clone());
                if self.transport_started && !self.kicked_off {
                    let mut rt = self.realtime.lock().await;
                    if let Err(e) = rt.kickoff().await {
                        drop(rt);
                        self.state.lock().unwrap().record_error(e);
                        let _ = self.end_tx.send(Frame::End { reason: None });
                        return Ok(());
                    }
                    self.kicked_off = true;
                }
                // Forward downstream (the recorder/etc. ignore it).
                link.push(env.meta, env.frame, env.direction).await;
            }

            // Caller audio in (call.rs's MediaIn::Audio arm): record at carrier rate,
            // resample carrier→16k, push to the model. Forward InputAudio downstream
            // so the RecorderProcessor taps the inbound leg.
            Frame::InputAudio(audio) => {
                // Ensure the reader is running even if audio precedes ClientConnected.
                self.spawn_reader(link.clone());
                let chunk = AudioChunk::from(audio.as_ref());
                let resampler = self
                    .in_resampler
                    .as_mut()
                    .expect("in_resampler set in start()");
                match resampler.process(&chunk) {
                    Ok(up) => {
                        if !up.is_empty() {
                            let mut rt = self.realtime.lock().await;
                            if let Err(e) = rt.send_audio(up).await {
                                drop(rt);
                                self.state.lock().unwrap().record_error(e);
                                let _ = self.end_tx.send(Frame::End { reason: None });
                                return Ok(());
                            }
                        }
                    }
                    Err(e) => {
                        self.state.lock().unwrap().record_error(e);
                        let _ = self.end_tx.send(Frame::End { reason: None });
                        return Ok(());
                    }
                }
                // Forward downstream for the RecorderProcessor tap.
                link.push(env.meta, env.frame, env.direction).await;
            }

            // Upstream tool result from the brain → send_tool_result (call.rs's
            // realtime.send_tool_result calls).
            Frame::Custom(c) if c.as_any().is::<ToolResult>() => {
                let tr = c.as_any().downcast_ref::<ToolResult>().unwrap();
                let mut rt = self.realtime.lock().await;
                if let Err(e) = rt.send_tool_result(tr.id.clone(), tr.result.clone()).await {
                    drop(rt);
                    self.state.lock().unwrap().record_error(e);
                }
            }

            // Upstream re-prompt from the brain → update_system (call.rs's
            // realtime.update_system call on a transition).
            Frame::Custom(c) if c.as_any().is::<Reprompt>() => {
                let rp = c.as_any().downcast_ref::<Reprompt>().unwrap();
                let mut rt = self.realtime.lock().await;
                if let Err(e) = rt.update_system(rp.prompt.clone(), rp.tools.clone()).await {
                    drop(rt);
                    self.state.lock().unwrap().record_error(e);
                }
            }

            _ => {
                link.push(env.meta, env.frame, env.direction).await;
            }
        }
        Ok(())
    }
}

// ===========================================================================
// BrainProcessor + ToolRelay.
// ===========================================================================

/// The seam that relays a *workflow* (MCP/HTTP) tool call to the control plane,
/// abstracting [`SessionSource::tool_call`] + [`SessionSource::node_tools`] so the
/// [`BrainProcessor`] is generic over the session (PROCESSOR-DESIGN §6.2: the
/// node-tools/tool-call relay becomes a `ToolRelay` injected into `BrainProcessor`).
#[async_trait]
pub trait ToolRelay: Send + Sync {
    /// The current node's MCP/HTTP workflow tools (degrades to empty on error).
    async fn node_tools(&self, node_id: &str) -> Vec<ToolDecl>;
    /// Relay a workflow tool call → the control plane runs the egress; returns the
    /// content string to feed back to the model verbatim.
    async fn relay(&self, node_id: &str, tool_name: &str, args: &serde_json::Value) -> String;
}

/// A [`ToolRelay`] over a [`SessionSource`] + the run id/token — the generic glue
/// that lets `BrainProcessor` stay session-agnostic (the embedder's concrete
/// `SessionSource` plugs in here).
pub struct SessionToolRelay<S: SessionSource> {
    session: Arc<S>,
    run_id: i64,
    token: String,
}

impl<S: SessionSource> SessionToolRelay<S> {
    pub(crate) fn new(session: Arc<S>, run_id: i64, token: String) -> Self {
        Self {
            session,
            run_id,
            token,
        }
    }
}

#[async_trait]
impl<S: SessionSource> ToolRelay for SessionToolRelay<S> {
    async fn node_tools(&self, node_id: &str) -> Vec<ToolDecl> {
        self.session
            .node_tools(self.run_id, &self.token, node_id)
            .await
            .unwrap_or_default()
    }
    async fn relay(&self, node_id: &str, tool_name: &str, args: &serde_json::Value) -> String {
        match self
            .session
            .tool_call(self.run_id, &self.token, node_id, tool_name, args)
            .await
        {
            Ok(c) => c,
            Err(_) => "the tool is temporarily unavailable".to_string(),
        }
    }
}

/// Shape an MCP tool's text result into the JSON **object** Gemini Live requires
/// for `functionResponses[].response`.
///
/// That field is typed as a protobuf `Struct`, so a bare string (or any non-object)
/// is rejected with `1007 "Invalid value at … response (Struct)"`, which tears down
/// the call. MCP returns a tool's output as text that is *usually itself JSON* — we
/// parse it through so the model receives structured data, and wrap any non-object
/// payload (plain text, a JSON array, a scalar) in `{result: …}` so the response is
/// always a valid `Struct`.
fn mcp_result_struct(content: &str) -> serde_json::Value {
    match serde_json::from_str::<serde_json::Value>(content) {
        Ok(v @ serde_json::Value::Object(_)) => v,
        Ok(other) => json!({ "result": other }),
        Err(_) => json!({ "result": content }),
    }
}

/// Holds the graph state and maps a model tool-call to either a **workflow-tool
/// relay** (emit `FunctionCallResult` straight back upstream, no transition —
/// mirrors `mcp_tool_call_is_relayed_not_treated_as_transition`) or a
/// [`BrainAction`] (`Transition`→`Reprompt`+result, `End`→upstream `End`, `Stay`).
///
/// Generic over [`AgentBrain`] + [`ToolRelay`]; consumes [`ModelToolCall`] frames,
/// pushes [`ToolResult`]/[`Reprompt`] upstream to the realtime service. Holds the
/// MCP-name set for the current node (the `mcp_names` branch set in call.rs).
pub struct BrainProcessor<B: AgentBrain, Rl: ToolRelay> {
    brain: B,
    relay: Arc<Rl>,
    /// The current node's workflow-tool names (call.rs's `mcp_names`).
    mcp_names: std::collections::HashSet<String>,
    /// Shared live state (transcript marker + disposition + collected_vars).
    state: SharedState,
    /// The pipeline-head queue, so `BrainAction::End` ends the call by injecting
    /// `End` at the head (Source → downstream drain through the whole chain). This
    /// is the "request to end" of PROCESSOR-DESIGN §4.1, routed through the head so
    /// every processor (incl. Source/TransportInput) breaks for a clean teardown.
    end_tx: tokio::sync::mpsc::UnboundedSender<Frame>,
}

impl<B: AgentBrain, Rl: ToolRelay> BrainProcessor<B, Rl> {
    pub(crate) fn new(
        brain: B,
        relay: Arc<Rl>,
        mcp_names: std::collections::HashSet<String>,
        state: SharedState,
        end_tx: tokio::sync::mpsc::UnboundedSender<Frame>,
    ) -> Self {
        Self {
            brain,
            relay,
            mcp_names,
            state,
            end_tx,
        }
    }

    /// Resolve the new node's tool set after a transition (transitions + node MCP),
    /// refreshing `mcp_names` — verbatim `call.rs::resolve_tools`'s ordering
    /// (transitions first, then MCP).
    async fn resolve_tools(&mut self) -> Vec<ToolDecl> {
        let node_id = self.brain.current_node_id();
        let mcp = self.relay.node_tools(&node_id).await;
        self.mcp_names = mcp.iter().map(|t| t.name.clone()).collect();
        let mut tools = self.brain.tools();
        tools.extend(mcp);
        tools
    }
}

#[async_trait]
impl<B: AgentBrain + 'static, Rl: ToolRelay + 'static> FrameProcessor for BrainProcessor<B, Rl> {
    fn name(&self) -> &str {
        "Brain"
    }

    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        let Frame::Custom(c) = &env.frame else {
            link.push(env.meta, env.frame, env.direction).await;
            return Ok(());
        };
        let Some(tc) = c.as_any().downcast_ref::<ModelToolCall>() else {
            link.push(env.meta, env.frame, env.direction).await;
            return Ok(());
        };
        let id = tc.id.clone();
        let name = tc.name.clone();
        let args = tc.args.clone();

        // ---- MCP/HTTP workflow tool (NOT a transition) ----  (call.rs:276)
        if self.mcp_names.contains(&name) {
            // A real tool call (not a transition) — tally for usage_metrics.tool_calls.
            self.state.lock().unwrap().tool_calls += 1;
            let node_id = self.brain.current_node_id();
            let content = self.relay.relay(&node_id, &name, &args).await;
            // Shape the MCP text result into the JSON object the model requires
            // (see `mcp_result_struct`): a bare string here is rejected by Gemini
            // Live with `1007 Invalid value at … response (Struct)`, killing the call.
            link.push_up(Frame::Custom(Arc::new(ToolResult {
                id,
                result: mcp_result_struct(&content),
            })))
            .await;
            return Ok(());
        }

        // ---- Otherwise: a transition / end / stay ----  (call.rs:307)
        match self.brain.on_tool_call(&name, &args) {
            BrainAction::Transition {
                system_prompt, say, ..
            } => {
                // Re-resolve the NEW node's tool set; re-prompt; ack {moved}; mark
                // the transition in the transcript (verbatim call.rs ordering:
                // update_system, then send_tool_result, then transcript marker).
                let tools = self.resolve_tools().await;
                link.push_up(Frame::Custom(Arc::new(Reprompt {
                    prompt: system_prompt,
                    tools,
                })))
                .await;
                link.push_up(Frame::Custom(Arc::new(ToolResult {
                    id,
                    result: json!({ "status": "moved" }),
                })))
                .await;
                let marker = match &say {
                    Some(s) if !s.is_empty() => format!("[transition: {name}] {s}"),
                    _ => format!("[transition: {name}]"),
                };
                self.state.lock().unwrap().transcript.push_bot(&marker);
            }
            BrainAction::Stay => {
                link.push_up(Frame::Custom(Arc::new(ToolResult {
                    id,
                    result: json!({ "status": "stay" }),
                })))
                .await;
            }
            BrainAction::End { disposition } => {
                // Ack {ended}, set disposition, request End (the Source converts the
                // upstream End to a downstream drain — call.rs breaks the loop here).
                link.push_up(Frame::Custom(Arc::new(ToolResult {
                    id,
                    result: json!({ "status": "ended" }),
                })))
                .await;
                self.state.lock().unwrap().disposition = disposition;
                // Capture collected vars now (the brain is consumed at end-of-call;
                // call.rs reads collected_vars after the loop while brain is alive).
                let vars = self.brain.collected_vars();
                self.state.lock().unwrap().collected_vars = vars;
                // Request the pipeline drain via the head (Source converts to a
                // downstream End through the whole chain — clean fast teardown).
                let _ = self.end_tx.send(Frame::End { reason: None });
            }
        }
        Ok(())
    }
}

// ===========================================================================
// TransportOutput — wraps a MediaTransport's send_audio/send_clear.
// ===========================================================================

/// Safety cap on the end-of-call playout drain — a runaway estimate must never
/// pin teardown open. No single bot utterance approaches this. (Shared with the
/// cascaded sink, which has the identical end-of-call drain.)
pub(super) const MAX_PLAYOUT_DRAIN: Duration = Duration::from_secs(15);

/// Advance the "all bot audio sent so far has finished playing at the carrier"
/// instant by a freshly-sent chunk's real-time duration.
///
/// The carrier plays audio in real time while we `send_audio` faster than real
/// time, so we track when playback will actually catch up. Back-to-back chunks
/// **queue** (the new chunk starts where the previous one ends); a chunk sent
/// after a gap (`until` already in the past) starts from `now`. Pure arithmetic
/// so the [`TransportOutput`] end-of-call drain is unit-testable without timing.
/// `pub(super)` so the cascaded sink reuses the exact same estimate.
pub(super) fn advance_playout(
    until: Option<Instant>,
    now: Instant,
    samples: usize,
    rate: u32,
) -> Instant {
    let dur = Duration::from_secs_f64(samples as f64 / rate.max(1) as f64);
    let base = match until {
        Some(t) if t > now => t,
        _ => now,
    };
    base + dur
}

/// The sink-side stage (PROCESSOR-DESIGN §7.1). Consumes `OutputAudio`/`TtsAudio`
/// → resamples 24k→carrier and `send_audio`s to the carrier; consumes
/// `Interruption` → `send_clear`. Records the outbound (bot) leg at 24k into the
/// shared recorder (call.rs records the bot chunk at 24k before the resample).
pub struct TransportOutput<T: MediaTransport> {
    /// The carrier transport (shared so the lazily-spawned… not needed — held here).
    transport: T,
    /// 24k→carrier resampler (one per call, preserves filter state — mirrors
    /// `call.rs`'s long-lived `out_resampler`).
    out_resampler: Option<Resampler>,
    carrier_rate: u32,
    /// Wall-clock instant by which all bot audio sent so far will have finished
    /// playing at the carrier (real-time playout). Advanced on every `send_audio`,
    /// cleared on barge-in. On `End` we wait out the remaining tail so the final
    /// utterance (e.g. the goodbye the model emits alongside `endCall`) isn't
    /// truncated when the carrier WS closes. `None` = nothing pending.
    bot_audio_until: Option<Instant>,
    state: SharedState,
}

impl<T: MediaTransport> TransportOutput<T> {
    fn new(transport: T, carrier_rate: u32, state: SharedState) -> Self {
        Self {
            transport,
            out_resampler: None,
            carrier_rate,
            bot_audio_until: None,
            state,
        }
    }
}

#[async_trait]
impl<T: MediaTransport + 'static> FrameProcessor for TransportOutput<T> {
    fn name(&self) -> &str {
        "TransportOutput"
    }

    async fn start(&mut self, _setup: &ProcessorSetup, _params: &StartParams) -> Result<()> {
        self.out_resampler = Some(Resampler::new(GEMINI_OUTPUT_RATE, self.carrier_rate)?);
        Ok(())
    }

    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        match &env.frame {
            // Bot audio out (call.rs's RealtimeEvent::AudioOut arm): record at 24k,
            // resample 24k→carrier, send to the carrier.
            Frame::OutputAudio(audio) | Frame::TtsAudio { audio, .. } => {
                let chunk = AudioChunk::from(audio.as_ref());
                // Record the bot leg at its source rate (24k) — call.rs does
                // `st.recorder.push_outbound(&chunk)` at 24k before resampling.
                self.state.lock().unwrap().recorder.push_outbound(&chunk);
                let resampler = self
                    .out_resampler
                    .as_mut()
                    .expect("out_resampler set in start()");
                match resampler.process(&chunk) {
                    Ok(down) => {
                        if !down.is_empty() {
                            // Track when this chunk finishes playing (carrier plays
                            // in real time) so `End` can wait out the tail.
                            let samples = down.pcm.len();
                            if let Err(e) = self.transport.send_audio(down).await {
                                self.state.lock().unwrap().record_error(e);
                            } else {
                                self.bot_audio_until = Some(advance_playout(
                                    self.bot_audio_until,
                                    Instant::now(),
                                    samples,
                                    self.carrier_rate,
                                ));
                            }
                        }
                    }
                    Err(e) => {
                        self.state.lock().unwrap().record_error(e);
                    }
                }
            }
            // Barge-in (call.rs's RealtimeEvent::Interrupted arm): flush the carrier.
            Frame::Interruption => {
                if let Err(e) = self.transport.send_clear().await {
                    self.state.lock().unwrap().record_error(e);
                }
                // The carrier dropped its queued bot audio — nothing left to drain.
                self.bot_audio_until = None;
                // Forward the interruption in its direction (framework also drains).
                link.push(env.meta, env.frame, env.direction).await;
            }
            // End-of-call: let the final bot utterance finish playing before the
            // carrier WS closes. The model emits the goodbye audio AND `endCall` in
            // one turn, so by the time `End` reaches us the goodbye is already
            // `send_audio`'d but still playing out — tearing down now truncates it
            // (cut off mid-"thank you for calling…"). Hold `End` here until the
            // real-time tail drains (bounded by `MAX_PLAYOUT_DRAIN`), then forward.
            Frame::End { .. } => {
                if let Some(until) = self.bot_audio_until.take() {
                    let now = Instant::now();
                    if until > now {
                        tokio::time::sleep((until - now).min(MAX_PLAYOUT_DRAIN)).await;
                    }
                }
                link.push(env.meta, env.frame, env.direction).await;
            }
            _ => {
                link.push(env.meta, env.frame, env.direction).await;
            }
        }
        Ok(())
    }
}

// ===========================================================================
// RecorderProcessor — taps both legs into the shared recorder + folds usage.
// ===========================================================================

/// Taps the inbound (caller) leg into the shared recorder and folds usage reports.
/// The outbound (bot) leg is recorded by [`TransportOutput`] (where the 24k chunk
/// is in hand), exactly mirroring `call.rs` which records inbound at carrier rate
/// and outbound at 24k. A pure observer-style tap: forwards every frame unchanged.
pub struct RecorderProcessor {
    state: SharedState,
}

impl RecorderProcessor {
    pub(crate) fn new(state: SharedState) -> Self {
        Self { state }
    }
}

#[async_trait]
impl FrameProcessor for RecorderProcessor {
    fn name(&self) -> &str {
        "Recorder"
    }

    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        match &env.frame {
            // Inbound caller audio at carrier rate — call.rs's push_inbound.
            Frame::InputAudio(audio) => {
                let chunk = AudioChunk::from(audio.as_ref());
                self.state.lock().unwrap().recorder.push_inbound(&chunk);
            }
            // Usage report — fold into the shared LiveState (call.rs accumulate_usage).
            Frame::Custom(c) if c.as_any().is::<UsageReport>() => {
                let u = c.as_any().downcast_ref::<UsageReport>().unwrap();
                self.state.lock().unwrap().accumulate_usage(&u.0);
            }
            _ => {}
        }
        link.push(env.meta, env.frame, env.direction).await;
        Ok(())
    }
}

// ===========================================================================
// TranscriptProcessor — taps user/bot transcription into the shared transcript.
// ===========================================================================

/// Taps `Transcription` frames (user/bot) into the shared transcript, mirroring
/// `call.rs`'s `RealtimeEvent::UserText`/`BotText` arms (`transcript.push_user/bot`).
/// Forwards every frame unchanged.
pub struct TranscriptProcessor {
    state: SharedState,
}

impl TranscriptProcessor {
    pub(crate) fn new(state: SharedState) -> Self {
        Self { state }
    }
}

#[async_trait]
impl FrameProcessor for TranscriptProcessor {
    fn name(&self) -> &str {
        "Transcript"
    }

    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        // Realtime path: user/bot transcription frames (user_id distinguishes). The
        // cascaded path taps its transcript at the aggregators instead (the user line
        // in `UserContextAggregator`, the bot reply in `AssistantContextAggregator`),
        // since there the transcription is consumed / the reply rides `LlmText`.
        if let Frame::Transcription { text, user_id, .. } = &env.frame {
            let mut st = self.state.lock().unwrap();
            if user_id.as_ref() == "bot" {
                st.transcript.push_bot(text);
            } else {
                st.transcript.push_user(text);
            }
        }
        link.push(env.meta, env.frame, env.direction).await;
        Ok(())
    }
}

// ===========================================================================
// FinalizeProcessor — on terminal End: artifact upload + complete.
// ===========================================================================

/// On terminal `End` (the `stop` hook), runs the `LiveState`/`finalize`
/// artifact-upload + `SessionSource::complete` logic verbatim from `call.rs`
/// (render recording + transcript, upload via presigned URL, persist the stored
/// keys, fold disposition into collected_vars, write the run result). Generic over
/// [`SessionSource`].
pub struct FinalizeProcessor<S: SessionSource> {
    session: Arc<S>,
    run_id: i64,
    token: String,
    state: SharedState,
    /// Guard so finalize runs at most once (End may arrive then Stop on teardown).
    finalized: Arc<AtomicBool>,
}

impl<S: SessionSource> FinalizeProcessor<S> {
    pub(crate) fn new(session: Arc<S>, run_id: i64, token: String, state: SharedState) -> Self {
        Self {
            session,
            run_id,
            token,
            state,
            finalized: Arc::new(AtomicBool::new(false)),
        }
    }

    /// The finalize body — a faithful port of `call.rs::finalize` (best-effort
    /// artifact upload + `complete`). Renders + uploads the recording and
    /// transcript, persists the stored keys, and writes the run result.
    async fn finalize(&self) {
        // Snapshot the shared state under the lock, then drop it for the awaits.
        let (recording_bytes, transcript_bytes, usage_json, collected_vars) = {
            let mut st = self.state.lock().unwrap();
            // Fold the final disposition into collected_vars (call.rs does this
            // after the loop). collected_vars was captured at End from the brain.
            let vars = merge_disposition(st.collected_vars.clone(), st.disposition.clone());
            st.collected_vars = vars.clone();
            let recording = st.recorder.render_wav();
            let transcript = st.transcript.render();
            (recording, transcript, st.usage_json(), vars)
        };

        let recording_url = match recording_bytes {
            Ok(bytes) => self.upload_artifact("recording", bytes, "audio/wav").await,
            Err(e) => {
                tracing::warn!(error = %e, "render recording wav failed; skipping recording artifact");
                None
            }
        };
        let transcript_url = self
            .upload_artifact("transcript", transcript_bytes, "application/json")
            .await;

        let fin = Finalize {
            usage: usage_json,
            collected_vars,
            recording_url,
            transcript_url,
        };
        if let Err(e) = self.session.complete(self.run_id, &self.token, fin).await {
            tracing::warn!(error = %e, run_id = self.run_id, "session.complete failed");
        }
    }

    /// Upload one artifact — verbatim `call.rs::upload_artifact` (presigned PUT,
    /// persist the stored key, never the secret-bearing URL).
    async fn upload_artifact(
        &self,
        kind: &str,
        bytes: Vec<u8>,
        content_type: &str,
    ) -> Option<String> {
        let target = match self
            .session
            .artifact_upload_url(self.run_id, &self.token, kind)
            .await
        {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!(error = %e, kind, "artifact_upload_url failed; skipping {kind}");
                return None;
            }
        };
        let ct = if target.content_type.is_empty() {
            content_type.to_string()
        } else {
            target.content_type.clone()
        };
        match self.session.put_bytes(&target.url, bytes, &ct).await {
            Ok(()) => Some(target.key),
            Err(e) => {
                tracing::warn!(error = %e, kind, "put_bytes failed; skipping {kind} url");
                None
            }
        }
    }
}

#[async_trait]
impl<S: SessionSource + 'static> FrameProcessor for FinalizeProcessor<S> {
    fn name(&self) -> &str {
        "Finalize"
    }

    async fn stop(&mut self, reason: StopReason) -> Result<()> {
        // Finalize on graceful End (and a Cancel teardown too — best-effort, once).
        if matches!(reason, StopReason::EndOfTask | StopReason::Cancelled)
            && !self.finalized.swap(true, Ordering::SeqCst)
        {
            self.finalize().await;
        }
        Ok(())
    }
}

// ===========================================================================
// The assembler — build the S2S PipelineTask.
// ===========================================================================

/// A built S2S task plus its transport pump handle (PROCESSOR-DESIGN §7.1).
///
/// Drive it with `task.run().await`; the pump feeds the transport's `recv()` into
/// the pipeline head and aborts when `run()` returns.
pub struct S2sTask {
    /// The wired pipeline task — run it to completion.
    pub task: PipelineTask,
    /// The transport-pump source reader (aborted on drop / after `run`).
    pump: SourcePump,
}

impl S2sTask {
    /// Run the S2S pipeline to completion, then stop the transport pump.
    pub async fn run(self) -> Result<()> {
        let res = self.task.run().await;
        self.pump.abort();
        res
    }
}

/// Assemble the S2S processor pipeline (PROCESSOR-DESIGN §7.1):
/// `[TransportInput → RealtimeService → Brain → Recorder → Transcript → TransportOutput → Finalize]`
/// plus the transport pump that feeds the head.
///
/// Connects the realtime model with the brain's opening prompt + tools (the start
/// node's transitions + MCP workflow tools), then wires the processor graph.
/// Generic over the four seams; the queue handle the pump uses is the task's own.
#[allow(clippy::too_many_arguments)]
/// Build the realtime (s2s) pipeline task. Delegates to
/// [`build_s2s_task_with_observers`] with no observers (the historical signature —
/// unchanged for every existing caller + test).
pub async fn build_s2s_task<T, R, B, S>(
    transport: T,
    realtime: R,
    brain: B,
    session: S,
    run_id: i64,
    token: String,
    model: String,
) -> Result<S2sTask>
where
    T: MediaTransport + 'static,
    R: RealtimeLlm + RealtimeKickoff + 'static,
    B: AgentBrain + 'static,
    S: SessionSource + 'static,
{
    build_s2s_task_with_observers(
        transport,
        realtime,
        brain,
        session,
        run_id,
        token,
        model,
        vec![],
    )
    .await
}

/// `build_s2s_task` with external pipeline `observers` (e.g. an `RtviObserver` that
/// streams live transcript/RTF events). `build_s2s_task` delegates here with no
/// observers — non-breaking for every existing caller.
#[allow(clippy::too_many_arguments)]
pub async fn build_s2s_task_with_observers<T, R, B, S>(
    transport: T,
    mut realtime: R,
    brain: B,
    session: S,
    run_id: i64,
    token: String,
    model: String,
    observers: Vec<Arc<dyn crate::observer::FrameObserver>>,
) -> Result<S2sTask>
where
    T: MediaTransport + 'static,
    R: RealtimeLlm + RealtimeKickoff + 'static,
    B: AgentBrain + 'static,
    S: SessionSource + 'static,
{
    let carrier_rate = transport.carrier_rate();
    let session = Arc::new(session);
    let relay = Arc::new(SessionToolRelay::new(
        session.clone(),
        run_id,
        token.clone(),
    ));

    // 1. Resolve the start node's tool set (transitions + node MCP tools) and the
    //    MCP-name branch set — verbatim call.rs::resolve_tools ordering.
    let node_id = brain.current_node_id();
    let mcp = relay.node_tools(&node_id).await;
    let mcp_names: std::collections::HashSet<String> = mcp.iter().map(|t| t.name.clone()).collect();
    let mut initial_tools = brain.tools();
    initial_tools.extend(mcp);

    // 2. Connect the realtime model with the opening prompt + tools (call.rs's
    //    pre-loop connect). On failure, finalize immediately (run is never built).
    // The model's required input rate (16k Gemini / 24k OpenAI Realtime). Captured
    // before `realtime` is moved into the processor; drives both the session config
    // we send the model AND the carrier→input resampler so the two never disagree.
    let input_rate = realtime.input_sample_rate();
    let setup = crate::types::RealtimeSetup {
        model,
        system_prompt: brain.system_prompt(),
        tools: initial_tools,
        input_sample_rate: input_rate,
        output_sample_rate: GEMINI_OUTPUT_RATE,
    };
    if let Err(e) = realtime.connect(setup).await {
        // Build a fresh state and finalize (mirrors call.rs's connect-fail branch).
        let state: SharedState = Arc::new(Mutex::new(LiveState::new(carrier_rate)));
        state.lock().unwrap().record_error(e);
        let fin = FinalizeProcessor::new(session, run_id, token, state);
        fin.finalize().await;
        return Err(FlowcatError::Realtime(
            "realtime connect failed; finalized".into(),
        ));
    }

    // 3. The shared live-call state (the LiveState analogue).
    let state: SharedState = Arc::new(Mutex::new(LiveState::new(carrier_rate)));

    // 4. Share the one transport between the pump (recv) and TransportOutput (send).
    let shared = SharedTransport::new(transport);

    // 5. The end-request channel. An inner processor (brain End / realtime Closed)
    //    requests a clean drain by sending `End` here; a forwarder injects it at the
    //    pipeline **head** (Source → downstream End through the whole chain). This is
    //    the §4.1 "request to end" routed through the head so every processor breaks.
    let (end_tx, mut end_rx) = tokio::sync::mpsc::unbounded_channel::<Frame>();

    // 6. Build the processor chain.
    let processors: Vec<Box<dyn FrameProcessor>> = vec![
        Box::new(TransportInput::new()),
        Box::new(RealtimeServiceProcessor::new(
            realtime,
            carrier_rate,
            input_rate,
            state.clone(),
            end_tx.clone(),
        )),
        Box::new(BrainProcessor::new(
            brain,
            relay,
            mcp_names,
            state.clone(),
            end_tx.clone(),
        )),
        Box::new(RecorderProcessor::new(state.clone())),
        Box::new(TranscriptProcessor::new(state.clone())),
        Box::new(TransportOutput::new(
            shared.clone(),
            carrier_rate,
            state.clone(),
        )),
        Box::new(FinalizeProcessor::new(
            session,
            run_id,
            token,
            state.clone(),
        )),
    ];
    let pipeline = Pipeline::new(processors);

    // 7. Build the task. Disable the idle timeout (call.rs has no idle gate; the
    //    call ends on brain End / transport Stop). Audio rates match Gemini.
    let params = PipelineTaskParams {
        audio_in_sample_rate: input_rate,
        audio_out_sample_rate: GEMINI_OUTPUT_RATE,
        idle_timeout: None,
        ..Default::default()
    };
    let task = PipelineTask::new(pipeline, params, observers);

    // 8. Forward end-requests into the task head (so Source breaks → clean drain).
    let head = task.queue_sender();
    tokio::spawn(async move {
        while let Some(f) = end_rx.recv().await {
            if head.send(f).is_err() {
                break;
            }
        }
    });

    // 9. Spawn the transport pump feeding the task head.
    let pump = spawn_transport_pump(shared, task.queue_sender());

    Ok(S2sTask { task, pump })
}

// ===========================================================================
// §7.2 — the S2S pipeline behaviour gate.
//
// Drive the S2S `PipelineTask` from the scripted mocks (the shared
// `pipeline::s2s_test_mocks`) and assert the 6 captured output sets match the
// golden values (PROCESSOR-DESIGN §7.2):
//   (a) count + order of `playAudio` frames to the carrier;
//   (b) the (id, status) tool-result sequence;
//   (c) fin.recording_url / fin.transcript_url stored keys;
//   (d) fin.collected_vars incl. the folded disposition;
//   (e) fin.usage.total_tokens;
//   (f) the relayed (node_id, tool_name, args) + verbatim MCP result.
// Covers BOTH scenarios (audio-both-ways + finalize, and MCP-relay-not-transition).
//
// NOTE: these golden values are exactly the outputs the now-deleted `Call::run`
// reference oracle produced — the differential test proved `build_s2s_task` is
// byte-for-byte equal to `Call::run` on both scenarios, so with `Call` removed
// the processor pipeline is asserted directly against that proven expected surface.
// ===========================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::s2s_test_mocks::{
        Captured, McpMockRealtime, MockBrain, MockRealtime, MockSession, MockSocket, CARRIER_RATE,
    };
    use crate::serializer::PlivoSerializer;
    use crate::transport::WsCarrierTransport;
    use crate::types::WsOut;
    use std::sync::atomic::AtomicBool;
    use std::time::Duration;

    const TEST_MODEL: &str = "models/test-realtime";

    /// Teardown must abort the model-event reader. Without it the reader keeps
    /// reconnecting the realtime session after the call ends (Gemini `1008` reconnect),
    /// burning provider quota for minutes. We stand a never-finishing
    /// task in for the reader, plant its handle, and assert `stop()` cancels it.
    #[tokio::test]
    async fn stop_aborts_the_reader_task() {
        let reader = tokio::spawn(async {
            loop {
                tokio::task::yield_now().await;
            }
        });
        let (end_tx, _end_rx) = tokio::sync::mpsc::unbounded_channel::<Frame>();
        let state: SharedState = Arc::new(Mutex::new(LiveState::new(CARRIER_RATE)));
        let mock = MockRealtime::new(
            Arc::new(AtomicBool::new(false)),
            Arc::new(Mutex::new(0)),
            Arc::new(Mutex::new(Vec::new())),
        );
        let mut proc = RealtimeServiceProcessor::new(mock, CARRIER_RATE, 16_000, state, end_tx);
        proc.reader_handle = Some(reader.abort_handle());
        assert!(!reader.is_finished(), "reader runs before stop()");

        proc.stop(StopReason::EndOfTask).await.unwrap();

        // After abort the task resolves to a cancelled JoinError.
        let joined = reader.await;
        assert!(
            joined.is_err() && joined.unwrap_err().is_cancelled(),
            "stop() must abort the reader task"
        );
        assert!(
            proc.reader_handle.is_none(),
            "the handle is taken on stop()"
        );
    }

    /// The captured outputs of one harness run — the §7.2 comparison surface.
    #[derive(Debug, PartialEq)]
    struct Captures {
        /// (a) The `playAudio` carrier frames, in order (full JSON payloads).
        play_audio: Vec<String>,
        /// (b) The (id, status) tool-result sequence (MockRealtime scenario).
        tool_results: Vec<(String, String)>,
        /// (c) recording + transcript stored keys.
        recording_url: Option<String>,
        transcript_url: Option<String>,
        /// (d) collected_vars incl. disposition.
        collected_vars: serde_json::Value,
        /// (e) usage total tokens.
        total_tokens: Option<u64>,
        /// (f) relayed (node_id, tool_name, args).
        relayed: Vec<(String, String, serde_json::Value)>,
        /// (f) the raw (id, result) sequence (MCP scenario — verbatim contents).
        raw_results: Vec<(String, serde_json::Value)>,
    }

    /// Extract the `playAudio` frames (in order) from the captured carrier output.
    fn play_audio_frames(sent: &[WsOut]) -> Vec<String> {
        sent.iter()
            .filter_map(|o| match o {
                WsOut::Text(t) if t.contains("playAudio") => Some(t.clone()),
                _ => None,
            })
            .collect()
    }

    /// Build the captures struct from the shared mock buffers.
    fn collect(
        sent: &Arc<Mutex<Vec<WsOut>>>,
        tool_results: &Arc<Mutex<Vec<(String, String)>>>,
        raw_results: &Arc<Mutex<Vec<(String, serde_json::Value)>>>,
        captured: &Arc<Mutex<Captured>>,
    ) -> Captures {
        let c = captured.lock().unwrap();
        let fin = c.finalize.as_ref();
        Captures {
            play_audio: play_audio_frames(&sent.lock().unwrap()),
            tool_results: tool_results.lock().unwrap().clone(),
            recording_url: fin.and_then(|f| f.recording_url.clone()),
            transcript_url: fin.and_then(|f| f.transcript_url.clone()),
            collected_vars: fin
                .map(|f| f.collected_vars.clone())
                .unwrap_or(serde_json::Value::Null),
            total_tokens: fin.and_then(|f| f.usage.get("total_tokens").and_then(|v| v.as_u64())),
            relayed: c.tool_calls.clone(),
            raw_results: raw_results.lock().unwrap().clone(),
        }
    }

    // ---- Scenario 1: audio both ways + finalize (the END_TOOL script) -------

    /// Run the S2S `PipelineTask` for scenario 1 and capture its outputs.
    async fn run_pipeline_scenario1() -> Captures {
        let sent = Arc::new(Mutex::new(Vec::<WsOut>::new()));
        let connected = Arc::new(AtomicBool::new(false));
        let audio_received = Arc::new(Mutex::new(0usize));
        let tool_results = Arc::new(Mutex::new(Vec::<(String, String)>::new()));
        let seen_tools = Arc::new(Mutex::new(Vec::<String>::new()));
        let captured = Arc::new(Mutex::new(Captured::default()));
        let raw_results = Arc::new(Mutex::new(Vec::<(String, serde_json::Value)>::new()));

        let transport = WsCarrierTransport::new(
            MockSocket::new(sent.clone()),
            PlivoSerializer::new(CARRIER_RATE),
        );
        let s2s = build_s2s_task(
            transport,
            MockRealtime::new(
                connected.clone(),
                audio_received.clone(),
                tool_results.clone(),
            ),
            MockBrain::new(seen_tools.clone()),
            MockSession::new(captured.clone()),
            4242,
            "tok-abc".into(),
            TEST_MODEL.into(),
        )
        .await
        .expect("build_s2s_task");
        tokio::time::timeout(Duration::from_secs(5), s2s.run())
            .await
            .expect("S2S pipeline timed out")
            .expect("S2S pipeline errored");
        collect(&sent, &tool_results, &raw_results, &captured)
    }

    #[tokio::test]
    async fn s2s_pipeline_matches_golden_audio_and_finalize() {
        let pipeline = run_pipeline_scenario1().await;

        // (a) The playAudio frames reached the carrier. The bot audio is fully
        // deterministic (one scripted AudioOut → resample 24k→8k → one Plivo
        // playAudio frame), so assert the exact count + that the frame is a
        // well-formed playAudio carrying a non-empty base64 payload (restores the
        // count+order constraint the differential oracle previously gave).
        assert_eq!(
            pipeline.play_audio.len(),
            1,
            "expected exactly one playAudio frame for the scripted bot audio"
        );
        let frame: serde_json::Value =
            serde_json::from_str(&pipeline.play_audio[0]).expect("playAudio frame is valid JSON");
        assert_eq!(frame["event"], "playAudio");
        assert!(
            frame["media"]["payload"]
                .as_str()
                .is_some_and(|p| !p.is_empty()),
            "playAudio frame must carry a non-empty base64 payload"
        );

        // (b) The (id, status) tool-result sequence (golden: the END_TOOL ack).
        assert_eq!(
            pipeline.tool_results,
            vec![("fc-end-1".into(), "ended".into())]
        );

        // (c) The stored recording/transcript keys.
        assert_eq!(
            pipeline.recording_url.as_deref(),
            Some("runs/4242/recording")
        );
        assert_eq!(
            pipeline.transcript_url.as_deref(),
            Some("runs/4242/transcript")
        );

        // (d) collected_vars incl. the folded disposition.
        assert_eq!(pipeline.collected_vars["call_disposition"], "completed");
        assert_eq!(pipeline.collected_vars["name"], "Ada");
        assert_eq!(pipeline.collected_vars["intent"], "support");

        // (e) Usage total tokens.
        assert_eq!(pipeline.total_tokens, Some(15));
    }

    /// Records user-side transcription frames as `(final, text)` at their ORIGIN
    /// (`source == "RealtimeService"`) so multi-hop traversal doesn't double-count.
    #[derive(Default)]
    struct UserTranscriptRecorder {
        seen: Mutex<Vec<(bool, String)>>,
    }
    #[async_trait::async_trait]
    impl crate::observer::FrameObserver for UserTranscriptRecorder {
        async fn on_push(&self, e: &crate::observer::FramePushEvent<'_>) {
            if e.source != "RealtimeService" {
                return;
            }
            match e.frame {
                Frame::InterimTranscription { text, user_id, .. } if user_id.as_ref() == "user" => {
                    self.seen.lock().unwrap().push((false, text.clone()));
                }
                Frame::Transcription {
                    text,
                    user_id,
                    final_,
                    ..
                } if user_id.as_ref() == "user" => {
                    self.seen.lock().unwrap().push((*final_, text.clone()));
                }
                _ => {}
            }
        }
    }

    /// The realtime user-transcription path emits streaming partials as a single
    /// GROWING interim line (`final:false`), then ONE finalized line (`final:true`)
    /// — never one committed bubble per word. (Scripted deltas "I'd " + "like to
    /// end now" → interim "I'd " then "I'd like to end now"; `.completed` →
    /// final "I'd like to end now".)
    #[tokio::test]
    async fn user_transcription_streams_one_interim_line_then_finalizes() {
        let sent = Arc::new(Mutex::new(Vec::<WsOut>::new()));
        let recorder = Arc::new(UserTranscriptRecorder::default());

        let transport = WsCarrierTransport::new(
            MockSocket::new(sent.clone()),
            PlivoSerializer::new(CARRIER_RATE),
        );
        let s2s = build_s2s_task_with_observers(
            transport,
            MockRealtime::new(
                Arc::new(AtomicBool::new(false)),
                Arc::new(Mutex::new(0)),
                Arc::new(Mutex::new(Vec::new())),
            ),
            MockBrain::new(Arc::new(Mutex::new(Vec::new()))),
            MockSession::new(Arc::new(Mutex::new(Captured::default()))),
            4242,
            "tok-abc".into(),
            TEST_MODEL.into(),
            vec![recorder.clone()],
        )
        .await
        .expect("build_s2s_task_with_observers");
        tokio::time::timeout(Duration::from_secs(5), s2s.run())
            .await
            .expect("S2S pipeline timed out")
            .expect("S2S pipeline errored");

        let seen = recorder.seen.lock().unwrap().clone();
        assert_eq!(
            seen,
            vec![
                (false, "I'd ".to_string()),
                (false, "I'd like to end now".to_string()),
                (true, "I'd like to end now".to_string()),
            ],
            "expected two growing interim lines then one final, got {seen:?}"
        );
    }

    /// Records bot-side frames at their origin (`source == "RealtimeService"`):
    /// every final bot transcript line + whether a `BotStoppedSpeaking` was emitted.
    #[derive(Default)]
    struct BotTranscriptRecorder {
        lines: Mutex<Vec<String>>,
        stopped: std::sync::atomic::AtomicUsize,
    }
    #[async_trait::async_trait]
    impl crate::observer::FrameObserver for BotTranscriptRecorder {
        async fn on_push(&self, e: &crate::observer::FramePushEvent<'_>) {
            if e.source != "RealtimeService" {
                return;
            }
            match e.frame {
                Frame::Transcription { text, user_id, .. } if user_id.as_ref() == "bot" => {
                    self.lines.lock().unwrap().push(text.clone());
                }
                Frame::BotStoppedSpeaking => {
                    self.stopped
                        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }
                _ => {}
            }
        }
    }

    /// The bot transcript is coalesced into ONE finalized line per response (flushed
    /// at `response.done`/Usage), and a `BotStoppedSpeaking` is emitted so the UI's
    /// "speaking…" indicator clears — never one bubble per delta. (MockRealtime
    /// scripts a single `BotText` then `Usage`.)
    #[tokio::test]
    async fn bot_transcript_coalesces_into_one_final_and_signals_stopped() {
        let sent = Arc::new(Mutex::new(Vec::<WsOut>::new()));
        let recorder = Arc::new(BotTranscriptRecorder::default());

        let transport = WsCarrierTransport::new(
            MockSocket::new(sent.clone()),
            PlivoSerializer::new(CARRIER_RATE),
        );
        let s2s = build_s2s_task_with_observers(
            transport,
            MockRealtime::new(
                Arc::new(AtomicBool::new(false)),
                Arc::new(Mutex::new(0)),
                Arc::new(Mutex::new(Vec::new())),
            ),
            MockBrain::new(Arc::new(Mutex::new(Vec::new()))),
            MockSession::new(Arc::new(Mutex::new(Captured::default()))),
            4242,
            "tok-abc".into(),
            TEST_MODEL.into(),
            vec![recorder.clone()],
        )
        .await
        .expect("build_s2s_task_with_observers");
        tokio::time::timeout(Duration::from_secs(5), s2s.run())
            .await
            .expect("S2S pipeline timed out")
            .expect("S2S pipeline errored");

        let lines = recorder.lines.lock().unwrap().clone();
        assert_eq!(
            lines,
            vec!["Hello, how can I help?".to_string()],
            "bot transcript should be one coalesced final line, got {lines:?}"
        );
        assert!(
            recorder.stopped.load(std::sync::atomic::Ordering::SeqCst) >= 1,
            "a BotStoppedSpeaking must be emitted so the UI finalizes the bubble"
        );
    }

    // ---- Scenario 2: MCP relay (not a transition) + end --------------------

    const MCP_TOOL: &str = "list_doctors";

    fn mcp_node_tools() -> Vec<ToolDecl> {
        vec![ToolDecl {
            name: MCP_TOOL.into(),
            description: "List available doctors.".into(),
            params: json!({
                "type": "object",
                "properties": { "specialty": { "type": "string" } }
            }),
        }]
    }

    /// Run the S2S `PipelineTask` for the MCP scenario and capture.
    async fn run_pipeline_scenario2() -> (Captures, Vec<Vec<String>>) {
        let sent = Arc::new(Mutex::new(Vec::<WsOut>::new()));
        let seen_tools = Arc::new(Mutex::new(Vec::<String>::new()));
        let raw_results = Arc::new(Mutex::new(Vec::<(String, serde_json::Value)>::new()));
        let advertised = Arc::new(Mutex::new(Vec::<Vec<String>>::new()));
        let captured = Arc::new(Mutex::new(Captured::default()));
        let tool_results = Arc::new(Mutex::new(Vec::<(String, String)>::new()));

        let session =
            MockSession::with_node_tools(captured.clone(), mcp_node_tools(), "Dr. Smith, Dr. Lee");
        let transport = WsCarrierTransport::new(
            MockSocket::new(sent.clone()),
            PlivoSerializer::new(CARRIER_RATE),
        );
        let s2s = build_s2s_task(
            transport,
            McpMockRealtime::new(raw_results.clone(), advertised.clone(), MCP_TOOL),
            MockBrain::new(seen_tools.clone()),
            session,
            4242,
            "tok-abc".into(),
            TEST_MODEL.into(),
        )
        .await
        .expect("build_s2s_task");
        tokio::time::timeout(Duration::from_secs(5), s2s.run())
            .await
            .expect("S2S pipeline timed out")
            .expect("S2S pipeline errored");
        let caps = collect(&sent, &tool_results, &raw_results, &captured);
        let adv = advertised.lock().unwrap().clone();
        (caps, adv)
    }

    #[tokio::test]
    async fn s2s_pipeline_matches_golden_mcp_relay_not_transition() {
        let (pipeline, pipe_adv) = run_pipeline_scenario2().await;

        // (f) The relayed (node_id, tool_name, args) to the control plane.
        assert_eq!(pipeline.relayed.len(), 1, "exactly one MCP tool relayed");
        let (node_id, tool_name, args) = &pipeline.relayed[0];
        assert_eq!(node_id, "start");
        assert_eq!(tool_name, MCP_TOOL);
        assert_eq!(args["specialty"], "dentistry");

        // (f) The RAW (id, result) sequence — the MCP result is shaped into a JSON
        // object (Gemini requires `response` to be a protobuf Struct; a bare string
        // is `1007`-rejected). The mock returns plain text, so it is wrapped as
        // `{result: …}`; endCall stays the {status: ended} envelope.
        let mcp = pipeline
            .raw_results
            .iter()
            .find(|(id, _)| id == "fc-mcp-1")
            .expect("MCP tool result sent back to the model");
        assert_eq!(mcp.1, json!({ "result": "Dr. Smith, Dr. Lee" }));
        let end = pipeline
            .raw_results
            .iter()
            .find(|(id, _)| id == "fc-end-1")
            .expect("endCall result sent");
        assert_eq!(end.1["status"], "ended");

        // The MCP tool was advertised at connect AFTER the transitions (stable
        // ordering: transitions first, then MCP tools).
        assert_eq!(
            pipe_adv.first(),
            Some(&vec!["end_call".to_string(), MCP_TOOL.to_string()]),
            "transitions first, then MCP tools"
        );
    }

    // ---- Pure-helper unit tests for the canonical finalize helpers ----------
    // These cover edge cases the golden integration tests don't exercise; they
    // target the S2S-path implementations (the only copies now that call.rs is
    // gone): `merge_disposition`, `add_opt`, `LiveState::accumulate_usage`.

    #[test]
    fn merge_disposition_inserts_into_object() {
        let merged = merge_disposition(json!({ "a": 1 }), Some("busy".into()));
        assert_eq!(merged["call_disposition"], "busy");
        assert_eq!(merged["a"], 1);
    }

    #[test]
    fn merge_disposition_does_not_clobber_existing() {
        let merged = merge_disposition(
            json!({ "call_disposition": "preset" }),
            Some("override".into()),
        );
        assert_eq!(merged["call_disposition"], "preset", "existing value wins");
    }

    #[test]
    fn merge_disposition_none_and_non_object_passthrough() {
        assert_eq!(
            merge_disposition(json!({ "a": 1 }), None),
            json!({ "a": 1 })
        );
        assert_eq!(
            merge_disposition(json!("scalar"), Some("x".into())),
            json!("scalar")
        );
    }

    #[test]
    fn add_opt_sums_and_treats_none_as_zero() {
        assert_eq!(add_opt(Some(2), Some(3)), Some(5));
        assert_eq!(add_opt(Some(2), None), Some(2));
        assert_eq!(add_opt(None, Some(3)), Some(3));
        assert_eq!(add_opt(None, None), None);
    }

    #[test]
    fn live_state_accumulates_usage_across_reports() {
        let mut st = LiveState::new(CARRIER_RATE);
        st.accumulate_usage(&Usage {
            input_tokens: Some(10),
            output_tokens: Some(2),
            total_tokens: Some(12),
            extra: None,
        });
        st.accumulate_usage(&Usage {
            input_tokens: Some(5),
            output_tokens: Some(1),
            total_tokens: Some(6),
            extra: Some(json!({ "cached": 3 })),
        });
        // Run-detail observability counts ride in the same usage object.
        st.transcript.push_user("hi");
        st.transcript.push_bot("hello");
        st.tool_calls = 2;
        let u = st.usage_json();
        assert_eq!(u["input_tokens"], 15);
        assert_eq!(u["output_tokens"], 3);
        assert_eq!(u["total_tokens"], 18);
        assert_eq!(u["extra"]["cached"], 3, "last non-null extra kept");
        // Wall-clock call duration rides alongside the token totals (0.0 here —
        // no audio pushed — but the control plane requires the field present).
        assert!(
            u["duration_seconds"].is_number(),
            "usage_json must surface duration_seconds"
        );
        assert_eq!(u["user_turns"], 1, "one user turn surfaced");
        assert_eq!(u["bot_turns"], 1, "one bot turn surfaced");
        assert_eq!(u["tool_calls"], 2, "tool calls tallied");
    }

    // ---- mcp_result_struct: Gemini `response` must be a protobuf Struct --------
    // Regression for the 1007 "Invalid value at … response (Struct)" that dropped
    // live calls the instant a tool (e.g. get_initial_prompt) returned JSON text.

    #[test]
    fn mcp_result_object_passes_through_as_struct() {
        // A JSON-object result (get_initial_prompt's `{a2ui: …}`) is already a
        // Struct → forwarded as-is so the model gets the structured fields.
        let v = mcp_result_struct(r#"{"a2ui":{"kind":"card"},"prompt":"hi"}"#);
        assert!(v.is_object(), "object result must stay an object");
        assert_eq!(v["a2ui"]["kind"], "card");
        assert_eq!(v["prompt"], "hi");
    }

    #[test]
    fn mcp_result_plain_text_is_wrapped() {
        // Non-JSON text → wrapped so `response` is still a Struct (the exact bug:
        // a bare string was sent verbatim and Gemini 1007-rejected it).
        let v = mcp_result_struct("Dr. Smith, Dr. Lee");
        assert_eq!(v, json!({ "result": "Dr. Smith, Dr. Lee" }));
        assert!(v.is_object());
    }

    #[test]
    fn mcp_result_json_non_object_is_wrapped() {
        // Valid JSON that is NOT an object (array / scalar) is not a Struct either,
        // so it must be wrapped rather than forwarded bare.
        assert_eq!(mcp_result_struct("[1,2,3]"), json!({ "result": [1, 2, 3] }));
        assert_eq!(mcp_result_struct("42"), json!({ "result": 42 }));
        assert_eq!(mcp_result_struct("true"), json!({ "result": true }));
        assert!(mcp_result_struct("[1,2,3]").is_object());
    }

    // ---- advance_playout: the end-of-call drain estimate ----------------------
    // Guards the fix for the goodbye being cut off — teardown must wait out the
    // bot audio still playing at the carrier when `endCall` fires.

    #[test]
    fn advance_playout_tracks_realtime_tail() {
        let now = Instant::now();
        // From nothing: one second of 8 kHz audio ⇒ plays until now + 1 s.
        let t = advance_playout(None, now, 8000, 8000);
        assert_eq!(t, now + Duration::from_secs(1));
        // Back-to-back: the next chunk queues after the previous one ends.
        let t2 = advance_playout(Some(t), now, 8000, 8000);
        assert_eq!(t2, now + Duration::from_secs(2));
        // A chunk sent after the tail already drained anchors to `now`, not the past.
        let gap = advance_playout(Some(now - Duration::from_secs(5)), now, 8000, 8000);
        assert_eq!(gap, now + Duration::from_secs(1));
        // Rate is respected: a 20 ms frame (160 samples @ 8 kHz).
        let frame = advance_playout(None, now, 160, 8000);
        assert_eq!(frame, now + Duration::from_millis(20));
    }
}
