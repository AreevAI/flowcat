# Flowcat — the composable FrameProcessor pipeline (FROZEN API)

> **Status: FROZEN.** This is the keystone API for the whole pipecat-parity program
> (see `ROADMAP.md`). Every later component (cascaded STT/TTS/LLM, VAD/turn,
> WebRTC/serializer transports, observability, transfer/DTMF) is implemented *against*
> the traits and types frozen here. Companion runtime
> doc: `DESIGN.md` (today's `Call::run`, the four seams, the audio path). Mirror source: pipecat
> `pipecat/src/pipecat/{frames/frames.py, processors/frame_processor.py,
> pipeline/{pipeline,parallel_pipeline,task,runner}.py, observers/base_observer.py,
> metrics/metrics.py}`.
>
> **Scope:** the *framework* only — `Frame`, `FrameProcessor`, `Pipeline`,
> `ParallelPipeline`, `PipelineTask`, `PipelineRunner`, `Observer`, metrics frames, the
> service-processor trait *signatures*, and the seam→processor mapping. **Provider impls,
> audio models, transports, and serializers are later work** and only their
> trait signatures are frozen here (so the fan-out can start against a stable surface).
>
> **Non-negotiable:** the live Gemini-Live S2S prod path keeps working on the current
> `Call::run` until the processor pipeline is *proven equivalent* (§7). It lands
> *alongside* `Call::run`, never as a rewrite-in-place.

---

## 0. Design goals & the constraints they come from

1. **Literal pipecat parity** in shape, so the ~80-provider fan-out is a mechanical port:
   a `Frame` taxonomy, a `FrameProcessor` with `process_frame(frame, direction)` +
   `push_frame`, prev/next linking, `Pipeline`/`ParallelPipeline`, `PipelineTask` +
   `PipelineRunner`, `Observer`. (pipecat `frame_processor.py:175`, `pipeline.py:91`,
   `task.py:142`.)
2. **Protect the p99 moat.** Today's `Call::run` is *one* `tokio::select!` loop
   (`pipeline.rs:195`) holding p99 ≤ 0.61 ms to 2,000 concurrent calls
   (`bench/RESULTS.md`). The channel-per-processor model adds per-frame hops; we must
   show the added cost stays in the sub-millisecond noise (§2.4).
3. **OSS-clean + compile-fast.** `flowcat-core` stays embedder-agnostic (`lib.rs`) and
   must build without pulling every provider; providers/transports live in sibling
   crates behind one cargo feature each (§8).
4. **Extensible frame set.** OSS users (and later components) must be able to
   add frame types *without editing `flowcat-core`*. This drives the **enum-core +
   `Frame::Custom(Arc<dyn CustomFrame>)` escape hatch** decision (§1.1).
5. **Zero ABI churn for the parallel fan-out.** Trait method shapes frozen now; adding a
   provider must never require touching the framework.

---

## 1. Frame taxonomy

### 1.1 Enum core with a trait escape hatch — and why

pipecat models frames as a Python class tree with `isinstance` dispatch
(`frames.py:54` `Frame` → `SystemFrame`/`DataFrame`/`ControlFrame`). The direct Rust
analogues are **(a) a closed `enum Frame`** or **(b) `trait Frame: Any` + downcast**.

| | closed `enum` | `trait Frame: Any` + downcast |
|---|---|---|
| Dispatch | `match` (no vtable, no alloc, branch-predicted) | `Any::downcast_ref` (type-id compare) per handler |
| Exhaustiveness | compiler-checked; adding a variant flags every `match` | none; missed types silently fall through |
| Per-frame cost | a stack enum move; the hot audio variant is `Arc<AudioFrame>` | `Box<dyn>` / `Arc<dyn>` heap alloc per frame |
| OSS extensibility | **closed** — users cannot add a variant | open — any type implementing the trait |
| Category (System/Data/Control) | one method `fn class()` | per-impl |

The moat (constraint 2) wants the cheap, branch-predicted `match` and the alloc-free
hot path; literal parity (constraint 1) wants exhaustiveness so the port is mechanical
and a new audio/turn frame can't be silently dropped. But constraint 4 (OSS users add
frame types) rules out a *purely* closed enum.

**Decision: a closed `enum Frame` for every pipecat frame, plus one `Custom` variant
carrying `Arc<dyn CustomFrame>` as the extension point.** Core processors `match` on
named variants with full exhaustiveness; OSS extensions ride in `Custom` and are
downcast only by the processors that care. This is the standard "enum + escape hatch"
pattern: 99% of frames are first-class and alloc-free; extensibility costs one
`Arc<dyn>` *only on the frames that use it*.

```rust
// flowcat-core/src/processor/frame.rs   (NOTE: distinct from today's data-shape
// `frame.rs`, which is renamed to `types.rs` — see §8.4 migration step M0.)

use std::any::Any;
use std::sync::Arc;

/// Direction of frame flow. Mirrors pipecat `FrameDirection`
/// (frame_processor.py:56). `Downstream` = source→sink; `Upstream` = sink→source
/// (errors, end-of-task requests, RTVI acks).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Downstream,
    Upstream,
}

/// Frame scheduling class — mirrors pipecat's three base classes
/// (frames.py:95/106/118). Drives queue priority and interruptibility:
/// `System` jumps the queue and survives interruption; `Data` is dropped on
/// interruption; `Control` is ordered like Data but also survives interruption
/// when `uninterruptible()` is set (e.g. `End`, `Stop`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameClass {
    System,
    Data,
    Control,
}

/// Per-frame metadata, present on every frame (mirrors the `Frame` base fields:
/// `id`, `name`, `pts`, `metadata` — frames.py:73-79). Cheap to clone (Arc the map).
#[derive(Debug, Clone)]
pub struct FrameMeta {
    /// Process-unique monotonic id (an `AtomicU64` bump; see `next_frame_id()`).
    pub id: u64,
    /// Human name for tracing/observers, e.g. "OutputAudio#42". Built lazily.
    pub name: &'static str,
    /// Presentation timestamp in **nanoseconds** on the pipeline clock, if set.
    pub pts: Option<i64>,
    /// Paired id when a frame was broadcast both directions (frames.py:76).
    pub broadcast_sibling_id: Option<u64>,
    /// Arbitrary sideband metadata (Arc so cloning a frame is cheap).
    pub extra: Option<Arc<serde_json::Map<String, serde_json::Value>>>,
    /// Transport source/destination track names (frames.py:78-79).
    pub transport_source: Option<Arc<str>>,
    pub transport_destination: Option<Arc<str>>,
}

/// OSS extension point: a frame type defined outside flowcat-core. Carried in
/// `Frame::Custom`. Processors that understand it `downcast_ref`; everyone else
/// forwards it unchanged in the direction it arrived.
pub trait CustomFrame: Any + Send + Sync + std::fmt::Debug {
    fn frame_class(&self) -> FrameClass;
    /// True if this frame must survive interruption (pipecat `UninterruptibleFrame`).
    fn uninterruptible(&self) -> bool { false }
    fn as_any(&self) -> &dyn Any;
}

/// The frame that flows through every processor. One closed enum mirroring the
/// pipecat frame tree (frames.py), plus `Custom` for OSS extensions.
///
/// The hot audio variants box their payload in `Arc<AudioFrame>` so cloning a
/// frame for the broadcast/observer paths never copies PCM.
#[derive(Debug, Clone)]
pub enum Frame {
    // ---- System frames (priority; survive interruption) — frames.py:846+ ----
    /// Pipeline init: carries sample rates + metric/trace toggles (StartFrame, :847).
    Start(StartParams),
    /// Immediate stop; flush nothing (CancelFrame, :873).
    Cancel { reason: Option<String> },
    /// Error notification pushed upstream (ErrorFrame, :890). `fatal` ⇒ task cancels.
    Error { message: String, fatal: bool, processor: Option<Arc<str>> },
    /// Barge-in (InterruptionFrame, :959). Broadcast both directions by the
    /// turn/VAD start strategy or any processor.
    Interruption,
    /// Raw caller audio from a transport input (InputAudioRawFrame, :1250).
    InputAudio(Arc<AudioFrame>),
    /// User-associated audio (UserAudioRawFrame, :1296) — carries `user_id`.
    UserAudio { audio: Arc<AudioFrame>, user_id: Arc<str> },
    /// Raw text input from a transport (InputTextRawFrame, :1282) — text-chat path.
    InputText(String),
    /// Inbound DTMF keypress (InputDTMFFrame, :1353).
    InputDtmf(KeypadEntry),
    /// VAD/turn lifecycle (frames.py:971-1104). One variant per pipecat frame.
    UserStartedSpeaking,
    UserStoppedSpeaking,
    UserSpeaking,
    BotStartedSpeaking,
    BotStoppedSpeaking,
    BotSpeaking,
    /// Definitive VAD edges with the deciding secs (VAD*SpeakingFrame, :1043/1058).
    VadUserStartedSpeaking { start_secs: f32 },
    VadUserStoppedSpeaking { stop_secs: f32 },
    /// Mute/unmute the STT service (STTMuteFrame, :1182).
    SttMute(bool),
    /// Performance metrics (MetricsFrame, :1108) — TTFB/processing/usage/turn.
    Metrics(Vec<MetricsData>),
    /// Transport-level message in/out urgent (Input/OutputTransportMessage*, :1193/1207).
    TransportMessage { payload: serde_json::Value, urgent: bool },
    /// SFU/transport lifecycle (BotConnected/ClientConnected, :1621/1633).
    ClientConnected,
    BotConnected,
    /// Function-call signalling (FunctionCallsStarted/InProgress/Cancel, :1155/1804/1169).
    FunctionCallsStarted(Vec<FunctionCall>),
    FunctionCallInProgress { call: FunctionCall, cancel_on_interruption: bool },
    FunctionCallCancel { function_name: String, tool_call_id: String },

    // ---- Data frames (ordered; dropped on interruption) — frames.py:190+ ----
    /// Output audio to a transport (OutputAudioRawFrame, :191).
    OutputAudio(Arc<AudioFrame>),
    /// TTS-generated audio, tagged with its context id (TTSAudioRawFrame, :231).
    TtsAudio { audio: Arc<AudioFrame>, context_id: Option<Arc<str>> },
    /// Generic text (TextFrame, :293) — flows LLM→aggregator→TTS.
    Text(String),
    /// LLM-generated text chunk (LLMTextFrame, :333).
    LlmText(String),
    /// Final transcription (TranscriptionFrame, :419).
    Transcription { text: String, user_id: Arc<str>, language: Option<Language>, final_: bool },
    /// Interim/partial transcription (InterimTranscriptionFrame, :445).
    InterimTranscription { text: String, user_id: Arc<str>, language: Option<Language> },
    /// Text the TTS should speak (TTSSpeakFrame, :744).
    TtsSpeak { text: String, append_to_context: Option<bool> },
    /// Word/segment text emitted by TTS with its context (TTSTextFrame, :400).
    TtsText { text: String, context_id: Option<Arc<str>> },
    /// Function-call result, fed back to the LLM (FunctionCallResultFrame, :719).
    /// Uninterruptible — once produced, context must always be updated.
    FunctionCallResult(FunctionCallResult),
    /// Trigger an LLM run over the current context (LLMRunFrame, :585).
    LlmRun,
    /// The universal LLM context to run (LLMContextFrame, :502).
    LlmContext(Arc<LlmContext>),
    /// Outbound DTMF (OutputDTMFFrame, :790).
    OutputDtmf(Vec<KeypadEntry>),

    // ---- Control frames (ordered; `End`/`Stop` survive interruption) — :1580+ ----
    /// Graceful shutdown after flush (EndFrame, :1581). Uninterruptible.
    End { reason: Option<String> },
    /// Stop but keep processors connected (StopFrame, :1605). Uninterruptible.
    Stop,
    /// LLM response framing (LLMFullResponseStart/End, :1699/1714).
    LlmResponseStart,
    LlmResponseEnd,
    /// TTS response framing (TTSStarted/Stopped, :1850/1867).
    TtsStarted { context_id: Option<Arc<str>> },
    TtsStopped { context_id: Option<Arc<str>> },
    /// Update a service's settings live (ServiceUpdateSettingsFrame, :1878).
    /// Uninterruptible. `target` = STT/TTS/LLM/Filter/Mixer/All.
    UpdateSettings { target: ServiceKind, settings: serde_json::Value },
    /// Speech-control params broadcast (SpeechControlParamsFrame, :1419) + VAD
    /// param updates (VADParamsUpdateFrame, :1939).
    SpeechControlParams { vad: Option<VadParams>, turn: Option<TurnParams> },
    /// Liveness probe (HeartbeatFrame, :1654).
    Heartbeat { timestamp_ns: i64 },
    /// Output transport ready (OutputTransportReadyFrame, :1644).
    OutputTransportReady,

    // ---- OSS extension point ----
    Custom(Arc<dyn CustomFrame>),
}
```

`Frame` carries its `FrameMeta` **out of band** to keep the enum small and `match`-cheap:
the channel item is `Envelope { meta: FrameMeta, frame: Frame, direction: Direction }`
(§2.1). (pipecat stuffs id/name/pts onto the frame object; we separate them so the
hot variant — `OutputAudio(Arc<AudioFrame>)` — stays a thin pointer move and the meta
travels alongside.)

```rust
impl Frame {
    /// Scheduling class — drives queue priority + interruptibility (§2.3).
    pub fn class(&self) -> FrameClass { /* match: System for Start/Cancel/Error/
        Interruption/Input*/Vad*/Metrics/FunctionCalls*/…; Control for End/Stop/
        Llm*Response/Tts*/UpdateSettings/Heartbeat/…; Data for the rest;
        Custom delegates to CustomFrame::frame_class() */ }

    /// True ⇒ kept in the queue and not cancelled on interruption (pipecat
    /// `UninterruptibleFrame`: End, Stop, FunctionCallResult, UpdateSettings — :1581/1605/719/1878).
    pub fn uninterruptible(&self) -> bool { /* match those; Custom delegates */ }
}
```

```rust
/// Mono 16-bit LE PCM with an explicit sample rate. **Unchanged** from today's
/// `frame.rs:14` AudioChunk — renamed `AudioFrame` and Arc-wrapped in the enum so
/// the hot path never copies PCM. (Keep an `AudioChunk` type alias for one release.)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioFrame { pub pcm: Vec<i16>, pub sample_rate: u32, pub num_channels: u16 }
```

`StartParams`, `FunctionCall`, `FunctionCallResult`, `LlmContext`, `KeypadEntry`,
`Language`, `VadParams`, `TurnParams`, `ServiceKind`, `MetricsData` are plain structs/
enums in `flowcat-core/src/processor/{frame,metrics}.rs` mirroring the pipecat fields
cited above. `StartParams` mirrors `StartFrame` (frames.py:847): `audio_in_sample_rate`,
`audio_out_sample_rate`, `enable_metrics`, `enable_usage_metrics`, `enable_tracing`,
`report_only_initial_ttfb`.

### 1.2 What we deliberately *omit* from the v1 enum

To keep the enum reviewable, the long tail of pipecat's 100+ frames that no current
component needs (vision/image frames `:171/209/1267`, sprite `:274`, summarization
`:1735/1751/1782`, prompt-caching `:671`, pause/resume `:928/1668`, idle-timeout-update
`:1925`, filter/mixer enable `:1971/2000`, service-switcher `:2011`) **map to `Custom`
until a component needs them first-class**, at which point that component promotes
them to a named variant (a non-breaking, additive change — adding an enum variant only
forces non-exhaustive `match` sites to add an arm, which the compiler points at). The
checklist (§9) names exactly which variants each early component promotes.

---

## 2. `FrameProcessor` trait + the channel runtime

### 2.1 The trait

```rust
// flowcat-core/src/processor/mod.rs
use async_trait::async_trait;

/// The frame envelope that travels a processor's input channel.
pub struct Envelope { pub meta: FrameMeta, pub frame: Frame, pub direction: Direction }

/// One-time per-task wiring handed to every processor at startup (mirrors pipecat
/// `FrameProcessorSetup`, frame_processor.py:71): the pipeline clock, the (optional)
/// observer fan-out, and the task's shared cancellation token.
#[derive(Clone)]
pub struct ProcessorSetup {
    pub clock: Clock,                       // monotonic ns; `clock.now_ns()`
    pub observer: Option<Observer>,         // Arc fan-out (§5)
    pub cancel: tokio_util::sync::CancellationToken,
    pub enable_metrics: bool,
    pub enable_usage_metrics: bool,
}

/// A processor's view of "downstream" / "upstream" — a `Sender` to each neighbour,
/// wired by `Pipeline::link`. Cloned into the processor's run loop.
#[derive(Clone)]
pub struct Link {
    next: Option<EnvelopeSender>,      // downstream neighbour input
    prev: Option<EnvelopeSender>,      // upstream neighbour input
    name: Arc<str>,                    // this processor's name (for observer events)
    clock: Clock,
    observer: Option<Observer>,
}

impl Link {
    /// Push a frame to the adjacent processor in `direction`. Mirrors pipecat
    /// `push_frame` (frame_processor.py:688): fires the observer `on_push_frame`
    /// hook, then enqueues onto the neighbour's input channel. Backpressure:
    /// `await`s if the neighbour's bounded channel is full (§2.2).
    pub async fn push(&self, meta: FrameMeta, frame: Frame, direction: Direction);
    /// Convenience: push a fresh frame downstream with new meta.
    pub async fn push_down(&self, frame: Frame);
    /// Push an `Error` frame upstream (pipecat `push_error`, :630).
    pub async fn push_error(&self, message: impl Into<String>, fatal: bool);
    /// Broadcast a frame both directions with paired sibling ids (pipecat
    /// `broadcast_frame`, :731) — used for `Interruption`.
    pub async fn broadcast(&self, frame: Frame);
}

/// The building block. Each processor runs in **its own tokio task** fed by a
/// bounded mpsc channel (§2.2). The framework owns the task loop; an impl only
/// writes `process_frame` (and optional `start`/`stop` hooks).
///
/// Mirrors pipecat `FrameProcessor` (frame_processor.py:175): `process_frame`,
/// prev/next links, system-frame priority, interruption handling — but the per-
/// processor task + queues are a *framework* concern here, not re-implemented per
/// processor as in Python.
#[async_trait]
pub trait FrameProcessor: Send + 'static {
    /// Stable, human-readable name (observer events, error attribution, tracing).
    fn name(&self) -> &str;

    /// Called once when the `Start` frame reaches this processor, before any data
    /// frame. Open sockets / spawn provider readers here. Default: no-op.
    async fn start(&mut self, _setup: &ProcessorSetup, _params: &StartParams) -> Result<()> { Ok(()) }

    /// Handle one frame. Push results via `link`. **Must not block**: long work
    /// (a provider round-trip) is driven by an internally-spawned task that feeds
    /// results back as frames (the Gemini reader-task pattern, gemini_live.rs:265).
    /// The default impl forwards the frame unchanged in its direction — so a
    /// pure observer/no-op processor is `process_frame` = default.
    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        link.push(env.meta, env.frame, env.direction).await; Ok(())
    }

    /// Called on `End`/`Stop`/`Cancel` after the terminal frame is forwarded.
    /// Flush + close. Default: no-op.
    async fn stop(&mut self, _reason: StopReason) -> Result<()> { Ok(()) }

    /// Whether this processor produces metrics (pipecat `can_generate_metrics`,
    /// :395). Services override to `true`.
    fn can_generate_metrics(&self) -> bool { false }
}
```

`Result<()>` is `flowcat_core::Result` (`error.rs`, `FlowcatError`); an `Err` returned
from `process_frame` is converted to an upstream `Frame::Error{fatal:false}` by the task
loop (pipecat `__process_frame` catch → `push_error`, :979).

### 2.2 The channel runtime — bounded mpsc, one task per processor

Each linked processor becomes a spawned task:

```rust
// Pseudocode of the framework-owned per-processor loop (processor/runtime.rs).
// Two channels per processor so System frames jump ahead of Data/Control —
// pipecat does this with a PriorityQueue (frame_processor.py:119); a second
// channel is the cheaper, branch-free Rust equivalent.
async fn run_processor(mut p: Box<dyn FrameProcessor>, mut rx: ProcessorRx, link: Link,
                       setup: ProcessorSetup) {
    loop {
        let env = tokio::select! {
            biased;                                   // system channel first
            Some(e) = rx.system.recv() => e,          // Start/Cancel/Interruption/…
            Some(e) = rx.normal.recv() => e,          // Data + Control
            else => break,
        };
        // observer on_process_frame hook (§5)
        if let Some(o) = &setup.observer { o.on_process(&link.name, &env, setup.clock.now_ns()); }
        match &env.frame {
            Frame::Start(p0)   => { p.start(&setup, p0).await.ok(); link.push(env.meta, env.frame, env.direction).await; }
            Frame::Interruption => { /* §2.5: drain `normal` of interruptible frames, keep uninterruptible; forward */ }
            Frame::Cancel{..} | Frame::End{..} | Frame::Stop
                               => { let _ = p.stop(reason(&env.frame)).await; link.push(env.meta, env.frame, env.direction).await; if terminal { break } }
            _                  => { if let Err(e) = p.process_frame(env, &link).await { link.push_error(e.to_string(), false).await; } }
        }
    }
}

struct ProcessorRx { system: mpsc::Receiver<Envelope>, normal: mpsc::Receiver<Envelope> }
type EnvelopeSender = ProcessorTx;   // holds both system+normal Senders; `Link::push`
                                     // routes by `frame.class()` (System→system chan)
```

**Bounded vs unbounded — the decision.** Pipecat uses unbounded `asyncio` queues.
Flowcat uses **bounded `tokio::mpsc` on the Data/Control (`normal`) channel and an
unbounded channel on the System path**:

- **Bounded normal channel (default capacity 64, the `bench-rs` value).** Real-time
  audio is rate-limited by the wall clock (~50 audio frames/s/leg); a bounded channel
  gives natural backpressure — if a slow processor (e.g. a TTS provider stalling) can't
  keep up, the producer `await`s instead of growing an unbounded queue and ballooning
  latency/RAM. This is the right behaviour for media: never buffer seconds of audio.
  Capacity 64 = ~1.3 s of audio headroom; a producer blocking on a full channel is the
  signal to interrupt/drop, not to buffer.
- **Unbounded system channel.** `Cancel`/`Interruption`/`Error`/`Start` must never block
  on a full queue (an interruption that can't be delivered defeats barge-in). The system
  path is low-volume (events, not a stream), so unbounded is safe and removes the one
  place backpressure would be a correctness bug. (This is the Rust equivalent of
  pipecat's `HIGH_PRIORITY` jumping the queue, frame_processor.py:128.)

Backpressure on the bounded channel is **per-hop** and bounded by capacity, so end-to-end
queueing is `O(processors × 64)` frames worst case — predictable, unlike an unbounded
chain. A producer that hits a full channel during an active interruption is unblocked
immediately because the interruption drains the consumer's queue (§2.5).

> **Scope of the backpressure guarantee.** The *inbound* audio
> leg — `InputAudio`/`UserAudio` — is classified `System` (`processor/frame.rs::class`),
> so it rides the **unbounded** channel, matching pipecat (input audio is a `SystemFrame`).
> That is deliberate: caller/transport capture is wall-clock-rate-limited (~50 fps) and must
> never be blocked. So the "bounded media backpressure / never buffer seconds of audio"
> property above applies to the **output/Data leg** (`OutputAudio`/`TtsAudio`, where a
> stalling TTS/realtime stage *is* the thing to backpressure), not the input leg. The moat
> argument is unaffected (input is producer-rate-limited), but don't overclaim bounded
> backpressure on the input path. A follow-up may add an explicit input-side drop/age
> policy if a downstream stall is ever observed to grow the inbound queue.

### 2.3 System-frame priority & interruptibility (parity with pipecat)

- **Priority:** `Link::push` routes a frame by `frame.class()`: `System` → the consumer's
  *system* channel; `Data`/`Control` → the *normal* channel. The `biased` select drains
  system first (pipecat `FrameProcessorQueue`, frame_processor.py:119-167).
- **Interruptibility:** on `Frame::Interruption`, the task loop drains the *normal*
  channel, **keeping** any frame whose `uninterruptible()` is true (End/Stop/
  FunctionCallResult/UpdateSettings), and cancels the in-flight `process_frame` only if
  the current frame is interruptible — exactly pipecat `_start_interruption`
  (frame_processor.py:828). The in-flight cancel is a `select!` between
  `process_frame(...)` and an interruption signal on the system channel.

### 2.4 The latency argument — the channel model stays inside the moat

The moat is **p99 ≤ 0.61 ms round-trip to 2,000 calls** on one process
(`bench/RESULTS.md`, Azure 16-vCPU). Two facts bound the cost of moving from one
`select!` loop to a channel-per-processor graph:

1. **`bench-rs` already measured exactly this model.** `bench-rs/src/main.rs` builds a
   **7-stage pipeline of tasks connected by bounded mpsc channels (`CHAN_CAP=64`)** —
   the literal analogue of the FrameProcessor graph designed here — and the *authoritative
   bench numbers in RESULTS.md are that channel pipeline*, not the monolithic loop. It
   reports **0.20 µs/frame, 0.029 µs/processor-hop** on one core, and the end-to-end
   real-I/O sweep (real WS + μ-law) holds **p99 ≤ 3 ms to 2,000 calls**. So the channel
   model's per-hop cost is *already inside* the published moat — adopting it is not a
   regression from the bench, it *is* the bench.
2. **Per-frame hop budget.** A live call is ~50–100 frames/s/leg. A cascaded pipeline is
   ≤ ~12 processors (transport-in, VAD, turn, STT, user-agg, LLM, assistant-agg, TTS,
   transport-out, + 2-3 filters/observers); S2S is ~5. At **0.029 µs/hop** (RESULTS.md),
   12 hops = **0.35 µs/frame** of pure framework routing — **three orders of magnitude**
   below the 0.61 ms p99 and four below the ~10–20 ms audio frame period. The dominant
   per-frame cost is, as RESULTS.md §"DOES NOT PROVE" states, the *shared* μ-law/resample/
   WS-syscall work — identical in both the loop and the graph model.

**Verdict: bounded-channel-per-processor adds ≤ ~0.4 µs/frame of routing, ~10⁻³ of the
p99 budget; the moat is preserved.** A later step extends `bench-rs` to run the *real*
`Pipeline` (not the standalone 7-stage mock) for both topologies and asserts p99 stays
≤ the current numbers — the gate that holds this claim honest (§9 step 11).

One nuance the design bakes in to keep this true: the **hot audio frame is
`Arc<AudioFrame>`**, so each hop moves a pointer (the bench moved `Bytes`, similarly
cheap); only `Vec<i16>` PCM produced by a codec/resample stage allocates, and that's
shared I/O cost, not framework cost.

### 2.5 Interruption end-to-end

Barge-in (`Frame::Interruption`) is produced by the turn/VAD start strategy or any
processor via `link.broadcast(Frame::Interruption)`. It travels the **system** channel
both directions; each processor drains its normal queue of interruptible frames and
cancels in-flight interruptible work, then forwards. The transport-output processor
additionally clears the carrier's playback buffer (today's `transport.send_clear()`,
pipeline.rs:370 → a `process_frame` arm on `TransportOutput`). This is the literal port
of pipecat `broadcast_interruption` (frame_processor.py:704).

---

## 3. `Pipeline` + `ParallelPipeline`

### 3.1 `Pipeline` — a linear chain of linked tasks

```rust
// flowcat-core/src/pipeline/mod.rs
pub struct Pipeline { processors: Vec<Box<dyn FrameProcessor>> }

impl Pipeline {
    pub fn new(processors: Vec<Box<dyn FrameProcessor>>) -> Self { Self { processors } }
}
```

`Pipeline` is itself a `FrameProcessor` (so it nests — pipecat `Pipeline(BasePipeline)`,
pipeline.py:91). It wraps the user processors with an internal **Source** and **Sink**
processor (pipecat `PipelineSource`/`PipelineSink`, pipeline.py:21/55) so the
`PipelineTask` can inject downstream frames at the head and observe upstream frames at the
head, and observe downstream frames at the tail. `link()` is the framework wiring step
run by `PipelineTask::setup`: it allocates the per-processor channels, builds each `Link`
(prev/next senders), and `tokio::spawn`s one `run_processor` task per element. The chain
order is `[Source, ...user, Sink]` (pipeline.py:119).

### 3.2 `ParallelPipeline` — fan-out / fan-in with lifecycle sync

```rust
pub struct ParallelPipeline { branches: Vec<Pipeline> }
impl ParallelPipeline { pub fn new(branches: Vec<Pipeline>) -> Self { /* ... */ } }
```

Also a `FrameProcessor`. Mirrors pipecat `ParallelPipeline` (parallel_pipeline.py:24):

- A frame entering the parallel block is queued into **every** branch's source.
- Each branch has its own Source/Sink; the Sink's downstream output funnels to the
  parallel block's single downstream, **de-duplicating by `meta.id`** (pipecat
  `_parallel_push_frame` + `_seen_ids`, parallel_pipeline.py:168) so a frame fanned to N
  branches is emitted once.
- **Lifecycle frames (`Start`/`End`/`Cancel`) are synchronized**: the block holds a
  per-frame counter = branch count, **buffers** non-lifecycle output while synchronizing,
  and only forwards the lifecycle frame (and flushes the buffer) once *all* branches have
  passed it (parallel_pipeline.py:158/182). This prevents a fast branch's `End` from
  shutting the transport down while a slow branch still has audio to flush — a correctness
  invariant we port verbatim.

ParallelPipeline is needed for the cascaded path's service-switcher / parallel STT
and for tee'd observers; it is *not* on the v1 Gemini S2S critical path, so it
lands with unit tests but is first exercised by the cascaded path.

---

## 4. `PipelineTask` + `PipelineRunner`

### 4.1 `PipelineTask` — one running pipeline's lifecycle

Mirrors pipecat `PipelineTask` (task.py:142). Owns: the wrapped pipeline (Source +
user + Sink), the push queue, the clock, the observer fan-out, idle detection,
heartbeat/watchdog, and the start/end/finished signalling.

```rust
// flowcat-core/src/pipeline/task.rs
pub struct PipelineTaskParams {
    pub audio_in_sample_rate: u32,    // default 16000
    pub audio_out_sample_rate: u32,   // default 24000 (S2S) / per-TTS (cascaded)
    pub enable_metrics: bool,
    pub enable_usage_metrics: bool,
    pub enable_tracing: bool,
    pub enable_heartbeats: bool,
    pub heartbeat_period: Duration,         // default 1s   (task.py:59)
    pub heartbeat_monitor: Duration,        // default 10s  (task.py:60)
    pub idle_timeout: Option<Duration>,     // default 300s (task.py:62)
    pub cancel_on_idle: bool,               // default true
    pub idle_timeout_frames: Vec<FrameKind>,// default [BotSpeaking, UserSpeaking]
}

pub struct PipelineTask { /* pipeline, clock, observers, channels, flags */ }

impl PipelineTask {
    pub fn new(pipeline: Pipeline, params: PipelineTaskParams,
               observers: Vec<Observer>) -> Self;

    /// Queue a downstream frame into the head of the pipeline.
    pub async fn queue_frame(&self, frame: Frame);
    pub async fn queue_frames(&self, frames: impl IntoIterator<Item = Frame>);

    /// Graceful: queue an `End` so the pipeline drains then shuts down (task.py:568).
    pub async fn stop_when_done(&self);
    /// Immediate: queue a `Cancel` (task.py:577).
    pub async fn cancel(&self, reason: Option<String>);

    pub fn has_finished(&self) -> bool;

    /// Run to completion: setup() spawns all processor tasks, inject `Start`, wait
    /// for it to reach the Sink (pipeline ready), pump queued frames, and exit when
    /// a terminal frame (`End`/`Stop`/`Cancel`) reaches the Sink. (task.py:586/818.)
    pub async fn run(self) -> Result<()>;

    /// Event hooks (pipecat task.py event handlers), each a registered async closure:
    pub fn on_started(&mut self, f: impl Fn() + ...);
    pub fn on_finished(&mut self, f: impl Fn(StopReason) + ...);
    pub fn on_error(&mut self, f: impl Fn(&str, bool) + ...);
    pub fn on_idle_timeout(&mut self, f: impl Fn() + ...);
    pub fn on_frame_reached_downstream(&mut self, types: &[FrameKind], f: ...);
    pub fn on_frame_reached_upstream(&mut self, types: &[FrameKind], f: ...);
}
```

**Lifecycle (task.py:818 `_process_push_queue` + :898 `_sink_push_frame`):**
1. `setup` builds channels, `spawn`s every processor task, starts the clock + idle task.
2. Inject `Frame::Start(params)` at the head; **block until it reaches the Sink** (every
   processor has run `start()`), then signal ready.
3. Pump `queue_frame`'d frames into the head.
4. The **Source** processor watches *upstream* frames: an upstream `End`/`Stop`/`Cancel`
   request (a processor wanting to end the call, today's `BrainAction::End`) is converted
   to the corresponding downstream lifecycle frame (task.py:859 `_source_push_frame`).
5. The **Sink** watches *downstream* frames: when `End`/`Stop`/`Cancel` reaches it, the
   task signals "ended" and exits the run loop; `Heartbeat` is timestamped for the monitor;
   `Error{fatal}` triggers `Cancel`.

**Idle detection (task.py:970):** a watcher resets on each `idle_timeout_frames` frame
seen by the observer; on timeout it fires `on_idle_timeout` and, if `cancel_on_idle`,
cancels. **Heartbeat/watchdog (task.py:941/950):** a task pushes `Heartbeat` every
`heartbeat_period`; the monitor warns if none returns within `heartbeat_monitor` — and
(Rust addition) doubles as a **per-task watchdog** since a wedged processor would block
the heartbeat from traversing. **Graceful shutdown:** `End` flushes; `Cancel` does not
(bounded by `cancel_timeout`, after which the task force-aborts the processor tasks via
the shared `CancellationToken`).

### 4.2 `PipelineRunner` — supervise tasks + signals

Mirrors pipecat `PipelineRunner` (runner.py:25). Runs one or many `PipelineTask`s, installs
**SIGINT/SIGTERM** handlers that `cancel()` all tasks for graceful drain (a deploy-cutover
drain story relies on SIGTERM grace), and joins them.

```rust
pub struct PipelineRunner { /* tasks, signal guard */ }
impl PipelineRunner {
    pub fn new(handle_sigint: bool, handle_sigterm: bool) -> Self;
    pub async fn run(&self, task: PipelineTask) -> Result<()>;
    pub async fn cancel_all(&self);
}
```

In the embedder, each inbound/outbound call constructs one `PipelineTask` and hands it to a
process-wide `PipelineRunner` — replacing today's "spawn `Call::run` per call".

---

## 5. `Observer` trait + metrics frames

### 5.1 Observer

Non-intrusive monitoring, mirroring pipecat `BaseObserver` (base_observer.py:70). An
`Observer` sees every processed/pushed frame without sitting in the chain — this is the
seam the observability layer (OpenTelemetry/Sentry/Langfuse/RTVI) plugs into.

```rust
// flowcat-core/src/observer.rs
pub struct FrameEvent<'a> {
    pub processor: &'a str,
    pub frame: &'a Frame,
    pub meta: &'a FrameMeta,
    pub direction: Direction,
    pub timestamp_ns: i64,    // pipeline clock
}
pub struct FramePushEvent<'a> {
    pub source: &'a str, pub destination: &'a str,
    pub frame: &'a Frame, pub meta: &'a FrameMeta,
    pub direction: Direction, pub timestamp_ns: i64,
}

#[async_trait]
pub trait FrameObserver: Send + Sync {
    /// A processor is about to handle a frame (base_observer.py:79).
    async fn on_process(&self, _e: &FrameEvent<'_>) {}
    /// A frame was pushed source→destination (base_observer.py:91).
    async fn on_push(&self, _e: &FramePushEvent<'_>) {}
    /// The pipeline finished starting (base_observer.py:103).
    async fn on_pipeline_started(&self) {}
}

/// Cheap clonable fan-out over many observers (pipecat `TaskObserver` proxy,
/// task.py:401). Hooks are invoked **synchronously on the hot path only when
/// enabled** — when no observer is registered the loop skips the call entirely
/// (the `if let Some(o)` in run_processor), so observation is zero-cost-when-off.
#[derive(Clone, Default)]
pub struct Observer(Arc<[Arc<dyn FrameObserver>]>);
```

Built-in observers shipped here (ported from pipecat): `TurnTrackingObserver`
(user/bot speaking edges → turn boundaries), `UserBotLatencyObserver` (TTFB of the bot's
first audio after the user stops), and the `IdleFrameObserver` that drives idle detection
(task.py:70). The observability layer adds the exporters.

### 5.2 Metrics frames

`Frame::Metrics(Vec<MetricsData>)` carries the same data as pipecat `MetricsFrame`
(frames.py:1108) + `metrics/metrics.py`:

```rust
// flowcat-core/src/processor/metrics.rs   (mirrors metrics.py)
pub enum MetricsData {
    Ttfb        { processor: String, model: Option<String>, seconds: f64 }, // metrics.py:29
    Processing  { processor: String, model: Option<String>, seconds: f64 }, // metrics.py:39
    LlmUsage    { processor: String, model: Option<String>, tokens: LlmTokenUsage }, // :68
    TtsUsage    { processor: String, characters: u64 },                     // :78
    TurnPrediction { processor: String, is_complete: bool, probability: f32,
                     e2e_processing_ms: f64 },                              // :101
}
pub struct LlmTokenUsage {                                                  // metrics.py:49
    pub prompt_tokens: u64, pub completion_tokens: u64, pub total_tokens: u64,
    pub cache_read_input_tokens: Option<u64>, pub cache_creation_input_tokens: Option<u64>,
    pub reasoning_tokens: Option<u64>,
}
```

A `FrameProcessor` produces metrics via helper methods on `Link` that gate on
`setup.enable_metrics` and emit a `Frame::Metrics` downstream: `start_ttfb`/`stop_ttfb`,
`start_processing`/`stop_processing`, `report_llm_usage`, `report_tts_usage` — the literal
port of `frame_processor.py:411-489`. `report_only_initial_ttfb` (StartParams) is honored.

---

## 6. Seam → processor mapping

Today's five trait seams (`MediaTransport`, `RealtimeLlm`, `AgentBrain`, `SessionSource`,
`MediaSerializer`) and `Call::run`'s inline logic become processors. The mapping:

| Today (seam / inline logic) | becomes | crate | notes |
|---|---|---|---|
| `MediaTransport::recv` (media.rs:49) | **`TransportInput` processor** — a *source*: reads the transport, emits `Frame::InputAudio`/`UserStartedSpeaking`/lifecycle downstream | `flowcat-transports` (trait stays in core) | pipecat `BaseInputTransport` |
| `MediaTransport::send_audio`/`send_clear` (media.rs:53/57) | **`TransportOutput` processor** — a *sink*: consumes `OutputAudio`/`TtsAudio`, plays to carrier; on `Interruption` clears playback (was pipeline.rs:370) | `flowcat-transports` | pipecat `BaseOutputTransport`; emits `BotStarted/StoppedSpeaking` |
| `RealtimeLlm` (realtime/mod.rs:22) | **`RealtimeLlmService` processor** — consumes `InputAudio`, emits `TtsAudio`(bot)/`Transcription`/`FunctionCallsStarted`/`Interruption`/`Metrics`; the reader-task→mpsc bridge (gemini_live.rs:265) becomes the processor's internal task feeding `link` | `flowcat-services` (`realtime-gemini` feature) | the trait below; Gemini is one impl |
| `AgentBrain` (brain.rs:22) | **`BrainProcessor`** — consumes `FunctionCallsStarted`/tool-call frames, emits `UpdateSettings`(new prompt+tools) on transition / `End` on terminal; holds the graph state | the embedder (its glue) | pipecat has no peer; this is the embedder's engine adapter |
| `SessionSource` (session.rs:21) | **stays embedder glue, NOT a processor** — a service the `BrainProcessor` + a `FinalizeProcessor` *call*; bootstrap/finalize/artifact-upload is control-plane I/O, not a media frame stage | the embedder | see §6.2 — it leaves flowcat-core for OSS cleanliness |
| `MediaSerializer` (serializer/mod.rs) | **stays a pure `FrameSerializer`** — no change of shape; `TransportInput`/`Output` for a WS carrier compose a `FrameSerializer` exactly as `WsCarrierTransport` does today | `flowcat-telephony` | pipecat `FrameSerializer` |
| `Call::run` orchestration (pipeline.rs:130) | **the `Pipeline` graph + `PipelineTask`** — the `select!` loop's arms become each processor's `process_frame`; `LiveState`/`finalize` become `FinalizeProcessor`+`SessionSource` | core + the embedder | the whole point of this framework |
| `LiveState` recorder/transcript (pipeline.rs:451) | **`RecorderProcessor` + `TranscriptProcessor`** — observers/sinks that tap audio + text frames | `flowcat-core` (recorder), the embedder (finalize) | pipecat recorder-as-processor |

### 6.1 New service-processor traits the cascaded path needs (signatures only)

These are frozen here so the provider implementations (the fan-out) build against them. Each is a
`FrameProcessor` *plus* a service-specific async contract; the framework's
`process_frame` arm calls the contract and emits the result frames. **Impls are later
work** — this framework ships only the trait + a no-op/mock impl for the integration test.

```rust
// flowcat-core/src/service/mod.rs

/// Streaming speech→text. Consumes `InputAudio`, emits `InterimTranscription` then
/// final `Transcription`. Mirrors pipecat `STTService`. (22 providers.)
#[async_trait]
pub trait SttService: Send {
    fn name(&self) -> &str;
    async fn start(&mut self, params: &StartParams) -> Result<()>;
    /// Feed one audio chunk; transcripts arrive asynchronously via the returned
    /// stream of `Frame`s (the processor forwards them downstream).
    async fn run_stt(&mut self, audio: Arc<AudioFrame>) -> Result<Vec<Frame>>;
    async fn set_muted(&mut self, muted: bool);
}

/// Streaming text→speech. Consumes `TtsSpeak`/`Text`, emits `TtsStarted`,
/// `TtsAudio`*, `TtsStopped`. Mirrors pipecat `TTSService`. (31 providers.)
#[async_trait]
pub trait TtsService: Send {
    fn name(&self) -> &str;
    fn sample_rate(&self) -> u32;
    async fn start(&mut self, params: &StartParams) -> Result<()>;
    async fn run_tts(&mut self, text: &str) -> Result<Vec<Frame>>;
}

/// Context-driven LLM. Consumes `LlmContext`/`LlmRun`, emits `LlmResponseStart`,
/// `LlmText`*, optional `FunctionCallsStarted`, `LlmResponseEnd`. (26 providers.)
#[async_trait]
pub trait LlmService: Send {
    fn name(&self) -> &str;
    async fn start(&mut self, params: &StartParams) -> Result<()>;
    async fn run_llm(&mut self, ctx: &LlmContext) -> Result<BoxStream<'_, Frame>>;
    fn set_tools(&mut self, tools: Vec<ToolDecl>);
}

/// The realtime S2S contract (today's RealtimeLlm, realtime/mod.rs:22 — UNCHANGED
/// shape, restated here as the canonical service trait the processor wraps).
#[async_trait]
pub trait RealtimeLlmService: Send {
    async fn connect(&mut self, setup: RealtimeSetup) -> Result<()>;
    async fn send_audio(&mut self, chunk: Arc<AudioFrame>) -> Result<()>;
    async fn update_system(&mut self, prompt: String, tools: Vec<ToolDecl>) -> Result<()>;
    async fn send_tool_result(&mut self, id: String, result: serde_json::Value) -> Result<()>;
    async fn next_event(&mut self) -> Option<RealtimeEvent>;
}

// ---- audio-intelligence traits — signatures only ----

/// Voice-activity detector. Mirrors pipecat `VADAnalyzer`. Silero (ONNX/`ort`)
/// is the reference impl.
pub trait VadAnalyzer: Send {
    fn sample_rate(&self) -> u32;
    /// Classify a frame of audio: Quiet / Starting / Speaking / Stopping.
    fn analyze(&mut self, audio: &AudioFrame) -> VadState;
    fn set_params(&mut self, params: VadParams);
}

/// End-of-turn / semantic-completion analyzer. Mirrors pipecat `BaseTurnAnalyzer`
/// (Smart-Turn v2/v3 is the reference impl).
pub trait TurnAnalyzer: Send {
    /// Given accumulated speech + VAD edges, predict whether the turn is complete.
    fn analyze_turn(&mut self, audio: &AudioFrame, vad: VadState) -> TurnPrediction;
    fn set_params(&mut self, params: TurnParams);
}
```

`RealtimeSetup`, `RealtimeEvent`, `ToolDecl`, `BrainAction` are **unchanged** from today's
`frame.rs` (renamed `types.rs`, §8.4) — the existing Gemini client and an embedder `AgentBrain` impl
satisfy them with thin wrapping into the processor shape.

### 6.2 Where `SessionSource` lives — and why it leaves flowcat-core

`SessionSource` (session.rs) is embedder-specific control-plane I/O (resolve a run+token,
upload artifacts, finalize over HTTP, node-tools/tool-call relay). It is **not
a media-frame stage** and it is the one seam that is *inherently* about the embedder's contract.
For OSS cleanliness it **lives in the embedder** as a plain service the
`BrainProcessor` and a `FinalizeProcessor` call — flowcat-core keeps only the
media-pipeline framework. The `node_tools`/`tool_call` relay (session.rs:53/66) becomes a
`ToolRelay` dependency injected into `BrainProcessor`. This is a clean cut: flowcat-core
has zero embedder knowledge (honoring `lib.rs` and the DESIGN.md OSS boundary), and the
embedder's glue owns bootstrap/finalize.

---

## 7. Migration / no-regression strategy

The prod Gemini-Live S2S path **must not regress** (live-verified runs per the
voice-live memory notes). Strategy: build the processor pipeline *beside* `Call::run`,
prove equivalence, then cut over.

### 7.1 The S2S processor pipeline (what the migration assembles)

```
TransportInput → RealtimeLlmService(Gemini) → BrainProcessor → TransportOutput
                                ▲                    │
                          (tool calls)      RecorderProcessor (taps both legs)
                          ToolRelay (embedder)       TranscriptProcessor
                                                     FinalizeProcessor (on End → SessionSource)
```

Every arm of today's `select!` (pipeline.rs:195-380) maps to a `process_frame`:
- carrier audio in (pipeline.rs:221) → `TransportInput` emits `InputAudio` → Gemini
  service `send_audio`.
- `RealtimeEvent::AudioOut` (pipeline.rs:249) → Gemini service emits `TtsAudio` →
  `TransportOutput` plays.
- `RealtimeEvent::ToolCall` MCP branch (pipeline.rs:276) → `BrainProcessor` recognizes
  the node's workflow tools (via `ToolRelay`) and emits a `FunctionCallResult` straight
  back to the Gemini service (no transition).
- `ToolCall` transition/end (pipeline.rs:307) → `BrainProcessor` emits `UpdateSettings`
  (new prompt+tools → Gemini `update_system`) or `End`.
- `RealtimeEvent::Interrupted` (pipeline.rs:367) → `Frame::Interruption` broadcast →
  `TransportOutput` clears.
- `Usage` (pipeline.rs:375) → `Frame::Metrics` → `RecorderProcessor`/finalize accumulate.
- loop break → terminal `End` → `FinalizeProcessor` runs the `LiveState`/`finalize`
  artifact-upload + `SessionSource.complete` logic (pipeline.rs:538).

### 7.2 The equivalence test (the gate that authorizes cutover)

This migration is **not done** until this passes. Reuse the *exact* mocks already in
`pipeline.rs:634` (MockSocket scripted Plivo frames, MockRealtime scripted event script,
MockBrain, MockSession capturing the finalize payload). Build **two harnesses driven by
the same scripted inputs**:

1. `Call::run` (today) → capture: outbound WS frames, `send_audio` count, tool-result
   statuses+ids, the `Finalize` payload (recording/transcript keys, `collected_vars`
   incl. folded disposition, `usage` totals), and the relayed `tool_call`s.
2. The processor `PipelineTask` (new) → capture the same set off the same mocks.

**Assert byte-for-byte equality of the captured outputs** (the same 6 assertion blocks
that `call_run_bridges_audio_both_ways_and_finalizes` and
`mcp_tool_call_is_relayed_not_treated_as_transition` already check, run against *both*
harnesses). Specifically equal: the count + order of `playAudio` frames sent to the
carrier; the `(id, status)` tool-result sequence; `fin.recording_url`/`transcript_url`
(stored keys, not presigned URLs); `fin.collected_vars` incl. `disposition`;
`fin.usage.total_tokens`; the relayed `(node_id, tool_name, args)` and verbatim MCP result.
Plus: a **timing assertion** that the processor pipeline's per-frame routing p99 ≤ the
`bench-rs` channel-pipeline number (§2.4) so cutover can't regress the moat.

Only when this differential test is green (CI, no network) does a later step rewire the embedder
to construct the `PipelineTask` instead of `Call::run`, and `Call::run` is **deleted**
(cleanup mandate — no parallel implementations). A live one-call smoke on a carrier dev
number (never prod) confirms the live path post-cutover.

---

## 8. Crate split & feature matrix

### 8.1 Crates (the target crate layout)

| Crate | Contents | Deps it may pull | License |
|---|---|---|---|
| `flowcat-core` | framework (frame, processor, pipeline, task, runner, observer, metrics) + audio (codec/resample/recorder) + native SIP UA + **all trait seams** (Transport/Stt/Tts/Llm/RealtimeLlm/Vad/Turn/FrameSerializer/Brain) | tokio, tokio-util, async-trait, serde(_json), bytes, thiserror, tracing, rubato, audio-codec-algorithms, hound, rsipstack, rand | Apache-2.0 |
| `flowcat-services` | every STT/TTS/LLM/realtime provider, **one cargo feature each** | per-feature: reqwest/tonic/tokio-tungstenite/ort/whisper-rs | Apache-2.0 |
| `flowcat-transports` | str0m WebRTC + Opus, WebSocket, Daily, LiveKit, local, avatars | str0m, opus/audiopus, per-feature SDKs | Apache-2.0 |
| `flowcat-telephony` | carrier `FrameSerializer`s (Twilio/Telnyx/Plivo/…) + DTMF (RFC2833 + Goertzel) | base64, serde_json | Apache-2.0 |
| `flowcat-cli` | demos/examples (parity with pipecat `examples/`) | the above | Apache-2.0 |
| the embedder | the host's glue: `BrainProcessor` (its engine adapter), `SessionSource`, `ToolRelay`, `FinalizeProcessor`, routing | flowcat-* + the host's engine | the host's license |

`flowcat-core` **must not** depend on `reqwest`/`tonic`/`ort`/`str0m` — those live in the
sibling crates so core stays compile-fast and dependency-light (constraint 3). The Gemini
Live client moves from `flowcat-core/src/realtime/` into `flowcat-services` behind the
`realtime-gemini` feature (the trait `RealtimeLlmService` stays in core).

### 8.2 Feature-flag matrix (pipecat "extras" parity)

- `flowcat-services`: `stt-deepgram`, `stt-whisper-local`, `tts-cartesia`, `tts-elevenlabs`,
  `llm-openai`, `llm-anthropic`, `realtime-gemini`, `realtime-openai`, … — **one feature per
  provider**, each pulling only its client dep. Umbrella features `stt-all`/`tts-all`/
  `llm-all` for the CLI/tests.
- `flowcat-core`: `sip` (native SIP UA; embedders that need telephony enable it), `recorder`, `vad-ort`
  (the ONNX runtime is heavy → opt-in even though the *trait* is always present).
- `flowcat-transports`: `webrtc-str0m`, `ws`, `daily`, `livekit`, `local`.
- `flowcat-telephony`: `twilio`, `telnyx`, `plivo`, `dtmf-inband`.

### 8.3 Dependency choices (locked here so the fan-out doesn't relitigate)

| Need | Crate | Why |
|---|---|---|
| ONNX (Silero VAD, Smart-Turn) | **`ort`** (ONNX Runtime) | the mature Rust ONNX binding; pipecat ships ONNX models; behind `vad-ort` |
| WebRTC | **`str0m`** (sans-I/O) | the chosen WebRTC stack; sans-I/O fits the tokio task model |
| Opus | **`audiopus`** (libopus binding) | WebRTC codec; `opus` pure-Rust is immature |
| gRPC (Google STT/TTS) | **`tonic`** | the tokio-native gRPC stack |
| HTTP/WS providers | **`reqwest`** + **`tokio-tungstenite`** | already in the tree (tungstenite is the Gemini socket) |
| local Whisper | **`whisper-rs`** | the mature whisper.cpp binding (toolchain note in §5) |

### 8.4 Module-rename housekeeping (step M0, done before any new code)

Today's `flowcat-core/src/frame.rs` holds *data shapes* (`AudioChunk`, `RealtimeEvent`,
`ToolDecl`, …), not pipeline frames. To avoid a name collision with the new
`processor/frame.rs` (the `Frame` enum), **rename the existing `frame.rs` → `types.rs`**
(pure mechanical, update `lib.rs` re-exports, keep `pub use` aliases for one release).
`AudioChunk` gets a type alias to the new `AudioFrame`. This is the first checklist step.

---

## 9. Implementation checklist (execute in order — engineer-ready)

Each step says **what to build, where, and the tests that gate it.** Steps 1–8 are
`flowcat-core` (single PR each or a small stack); step 9 is the migration; 10–12 are
cross-cutting. The framework itself does **not** touch the security-sensitive surfaces
(those come with the later work on transfer/DTMF/WebRTC-signaling/serializer-sigs).

0. **M0 — rename `frame.rs`→`types.rs`** (§8.4). No behavior change. Test: workspace
   builds + existing suite green.
1. **`processor/frame.rs` — the `Frame` enum + `FrameMeta` + `Direction` + `FrameClass`
   + `CustomFrame` + `AudioFrame`** (§1). Tests: `class()`/`uninterruptible()` table tests
   for every variant; a `Custom` frame round-trips through a no-op processor unchanged; a
   broadcast pairs sibling ids.
2. **`processor/metrics.rs` — `MetricsData` + `LlmTokenUsage`** (§5.2). Tests: serde
   round-trip; mirrors metrics.py field-for-field.
3. **`processor/mod.rs` + `processor/runtime.rs` — `FrameProcessor` trait, `Link`,
   `Envelope`, `ProcessorSetup`, the bounded/unbounded dual-channel `run_processor` loop,
   system-frame priority, interruption drain** (§2). Tests: a 3-processor hand-wired chain
   forwards frames in order; a `System` frame overtakes a backlog of `Data` frames; an
   `Interruption` drops interruptible queued frames but keeps an `End`; a `process_frame`
   `Err` becomes an upstream `Error`. **Promotes** to named variants only what these tests
   need.
4. **`observer.rs` — `FrameObserver`, `Observer` fan-out, `FrameEvent`/`FramePushEvent`,
   `TurnTrackingObserver`, `UserBotLatencyObserver`, `IdleFrameObserver`** (§5.1). Tests:
   an observer sees every push; zero-cost-when-none (no observer ⇒ hook not called);
   turn-tracking emits boundaries off scripted speaking edges.
5. **`pipeline/mod.rs` — `Pipeline` (Source/Sink wrap, `link()` spawns tasks),
   nesting** (§3.1). Tests: a `Pipeline` is a `FrameProcessor` and nests; downstream
   injected at head reaches the Sink; upstream observed at head.
6. **`pipeline/parallel.rs` — `ParallelPipeline` (fan-out, id-dedup, lifecycle sync)**
   (§3.2). Tests: a frame fans to N branches and emits once; a fast branch's `End` is held
   until the slow branch passes it (the sync invariant).
7. **`pipeline/task.rs` — `PipelineTask` (Start→ready handshake, push pump, Source/Sink
   upstream/downstream handling, idle, heartbeat/watchdog, graceful vs cancel, event
   hooks)** (§4.1). Tests: `Start` reaches Sink before any data frame; `stop_when_done`
   drains then ends; `cancel` skips flush; idle timeout fires + cancels; an upstream `End`
   request from a processor converts to a downstream `End`.
8. **`pipeline/runner.rs` — `PipelineRunner` (SIGINT/SIGTERM → cancel_all, join)** (§4.2).
   Tests: a simulated signal cancels a running task; multiple tasks join cleanly.
9. **The migration — the S2S processor pipeline + the equivalence test** (§7). Build
   `TransportInput`/`TransportOutput` (core trait + the WS-carrier composition reusing
   `FrameSerializer`), wrap the **existing Gemini client** as `RealtimeLlmService` behind a
   processor, `BrainProcessor` + `ToolRelay` + `FinalizeProcessor` in the embedder. The
   gate: the **differential test** (§7.2) asserting the processor `PipelineTask` produces
   byte-identical outputs to `Call::run` off the shared scripted mocks, plus the p99 timing
   assertion. **Cutover (delete `Call::run`) only after this is green** — a separate
   PR.
10. **Service-processor trait stubs** (§6.1): land `SttService`/`TtsService`/`LlmService`/
    `VadAnalyzer`/`TurnAnalyzer` traits + a no-op mock impl of each in `flowcat-core/src/
    service/` and `flowcat-core/src/audio/`. Test: a mock cascaded pipeline
    (mock-STT→mock-LLM→mock-TTS) runs a turn end-to-end through a real `Pipeline` — the
    fixture the provider implementations build against.
11. **Extend `bench-rs` to drive the *real* `Pipeline`** (not the standalone
    7-stage mock) for both S2S and cascaded topologies and assert p99 ≤ the published
    moat. Wire `cargo test --manifest-path flowcat/Cargo.toml --locked` into
    `.woodpecker/test.yaml` (flowcat is a separate workspace, excluded from `--workspace`).
12. **Crate split** (§8.1): add `flowcat-services`/`-transports`/
    `-telephony` skeletons + the feature matrix; move the Gemini client into
    `flowcat-services` behind `realtime-gemini`; Apache headers/NOTICE. (Can run in
    parallel with 9–11.)

**Build order rationale:** the enum (1) and the processor runtime (3) are the spine;
everything else composes them. The equivalence test (9) is the *single* gate that makes
the whole multi-quarter program safe — nothing in the later provider waves starts until 1–9
are frozen and green (see `ROADMAP.md`).

---

## 10. Open questions (need an architect/user call before/at cutover)

1. **Custom-frame downcast ergonomics for OSS users.** The `Frame::Custom(Arc<dyn
   CustomFrame>)` escape hatch is the agreed extensibility model, but a provider that
   needs a *first-class, hot-path* frame (rare) must promote a variant in core — i.e. a
   PR to flowcat-core. Acceptable? (Recommendation: yes — the long tail rides `Custom`;
   only genuinely-hot frames get variants, which is also true of pipecat's own evolution.)
2. **`BrainProcessor` location for the OSS demo.** flowcat-core ships a *demo* brain today
   (`lib.rs`); the embedder's engine adapter is the host's own code. Confirm the demo
   `BrainProcessor` stays in `flowcat-core` (so the OSS pipeline is runnable end-to-end) while
   the engine adapter lives in the embedder. (Recommendation: yes — matches the DESIGN.md OSS
   boundary.)
3. **Heartbeat-as-watchdog default.** Pipecat ships heartbeats *off* by default. An embedder
   with a deploy-cutover drain story likely wants them *on* with the 1s/10s defaults — it can
   flip `enable_heartbeats` true in its own `PipelineTaskParams` without changing the core default.
4. **str0m vs a thin WebRTC sidecar for Daily.** Not a decision for this framework, but the
   `TransportInput`/`Output` trait shape frozen here must not assume the transport owns its own
   event loop — confirmed sans-I/O-friendly above.
5. **Lifecycle frames bypass `process_frame` (RESOLVED).** The framework
   loop (`run_processor`, §2.2) routes `Start` and *downstream* `End`/`Stop`/`Cancel` to the
   `start`/`stop` hooks and **never** to `process_frame`; an *upstream* terminal is a
   "request to end" that DOES reach `process_frame` so the `Source` can convert it (§4.1).
   This is the single most surprising property for processor authors and was the root of two
   bring-up hangs (the internal `Sink` and `Source` each needed an edge wired for it). **Ruled
   correct and kept** (the framework — not the author — owns the lifecycle; routing lifecycle
   through `process_frame` would force per-processor lifecycle boilerplate, the exact thing the
   design eliminates). The contract is now stated on the `FrameProcessor` trait doc itself
   (`processor/mod.rs`) so every later / OSS author hits it. Every later component
   depends on knowing it.
6. **Source-emit affordance for transports (RESOLVED).** A *source* processor (a transport reading `recv()`) cannot self-emit
   from the frozen `FrameProcessor::start` hook (it gets `&ProcessorSetup` + `&StartParams`
   but **no `Link`**). The migration worked around this with a bespoke external pump feeding
   `PipelineTask::queue_sender()`. The open question was whether the frozen trait should grow a
   first-class source affordance before more transports proliferate. **Decision: (a) codify
   the external-pump pattern — NO trait change.** flowcat-core ships a small
   [`SourcePump`]/[`SourceHandle`] helper (`pipeline/source_pump.rs`) that wraps a transport's
   reader task and `emit`s frames at the **pipeline head** via the task's
   `queue_sender()`; the helper owns the spawn + abort lifecycle so every transport author gets a
   one-liner instead of re-deriving the pump. **Why (a) over a `run_source` trait hook:**
   (i) the head-injection path is the *only* way to preserve the Start→ready ordering guarantee
   — `PipelineTask::run` blocks on the Start→Sink handshake **before** it drains the head queue,
   so a pumped `InputAudio` provably cannot reach any `process_frame` before that processor's
   `start()` ran (the invariant that matters for `.expect()`-in-`process_frame`
   processors). A runtime-spawned `run_source(link, …)` would emit *before* `Start` traverses
   downstream, re-introducing exactly the ordering hazard the bring-up fixed — making it safe
   would require threading a new start-barrier through the frozen runtime, the most delicate
   part of the runtime. (ii) This is *literally pipecat's own model* — `BaseInputTransport` spawns a
   reader task and pushes; it has no synchronous source-emit method either. (iii) Backpressure
   stays correct: the head queue is unbounded (input must never block — §2.2) and the first
   bounded *normal* channel one hop in applies natural backpressure. (iv) Zero churn on a
   just-frozen + reviewed trait; the helper is additive and re-uses the proven handshake.
   **Code landed** alongside this ruling: `SourcePump`/`SourceHandle` + 2 unit tests (one
   asserts the Start handshake holds for pump-injected frames, one asserts abort-on-drop);
   `s2s.rs`'s bespoke `spawn_transport_pump` refactored onto it (the §7.2 differential test
   stays green). 157 flowcat tests green (`--locked`), clippy-clean. **Note: this adds to
   the frozen public surface (new `pipeline::{SourcePump, SourceHandle}` re-exports).**
