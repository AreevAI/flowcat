// SPDX-License-Identifier: Apache-2.0
//
//! The [`FrameProcessor`] trait + the channel-runtime building blocks
//! (PROCESSOR-DESIGN §2).
//!
//! Each processor runs in **its own tokio task** fed by a bounded mpsc channel
//! (Data/Control) plus an unbounded mpsc channel (System frames jump the queue).
//! The framework owns the task loop ([`runtime::run_processor`]); an impl only
//! writes [`FrameProcessor::process_frame`] (and optional `start`/`stop` hooks).

pub mod frame;
pub mod metrics;
pub mod runtime;

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;

use crate::error::Result;
use crate::observer::Observer;
use frame::{next_frame_id, Direction, Frame, FrameMeta, StartParams};
use metrics::MetricsData;
use runtime::EnvelopeSender;

pub use runtime::{run_processor, ProcessorRx, ProcessorTx};

/// The frame envelope that travels a processor's input channel
/// (PROCESSOR-DESIGN §2.1). Carries the [`FrameMeta`] out of band so the hot
/// [`Frame`] variant stays a thin pointer move.
#[derive(Debug, Clone)]
pub struct Envelope {
    pub meta: FrameMeta,
    pub frame: Frame,
    pub direction: Direction,
}

impl Envelope {
    /// Wrap `frame` with fresh meta flowing `direction`.
    pub fn new(frame: Frame, direction: Direction) -> Self {
        Self {
            meta: FrameMeta::new(&frame),
            frame,
            direction,
        }
    }
}

/// Why a processor's [`FrameProcessor::stop`] was called.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Graceful drain-then-shutdown (`End`).
    EndOfTask,
    /// Stop processing but keep links connected (`Stop`).
    Stopped,
    /// Immediate, no flush (`Cancel`).
    Cancelled,
}

/// A monotonic nanosecond clock shared across a task's processors. Mirrors
/// pipecat's pipeline `Clock` (`task.py`): one `Instant` base captured at setup,
/// `now_ns()` returns elapsed-since-base in ns.
#[derive(Clone)]
pub struct Clock {
    base: Instant,
}

impl Clock {
    /// Start a clock at "now".
    pub fn new() -> Self {
        Self {
            base: Instant::now(),
        }
    }

    /// Nanoseconds elapsed since the clock's base instant.
    pub fn now_ns(&self) -> i64 {
        self.base.elapsed().as_nanos() as i64
    }
}

impl Default for Clock {
    fn default() -> Self {
        Self::new()
    }
}

/// One-time per-task wiring handed to every processor at startup (mirrors pipecat
/// `FrameProcessorSetup`, `frame_processor.py:71`): the pipeline clock, the
/// (optional) observer fan-out, the shared cancellation token, and the metric
/// toggles.
#[derive(Clone)]
pub struct ProcessorSetup {
    pub clock: Clock,
    pub observer: Option<Observer>,
    pub cancel: tokio_util::sync::CancellationToken,
    pub enable_metrics: bool,
    pub enable_usage_metrics: bool,
}

/// A processor's view of "downstream" / "upstream" — a sender to each neighbour,
/// wired by [`crate::pipeline::Pipeline`]. Cloned into the processor's run loop.
#[derive(Clone)]
pub struct Link {
    pub(crate) next: Option<EnvelopeSender>,
    pub(crate) prev: Option<EnvelopeSender>,
    pub(crate) name: Arc<str>,
    pub(crate) clock: Clock,
    pub(crate) observer: Option<Observer>,
    /// Whether metrics emission is enabled for this task (gates the helpers).
    pub(crate) enable_metrics: bool,
    /// Whether usage metrics emission is enabled (gates `report_*_usage`).
    pub(crate) enable_usage_metrics: bool,
    /// Per-link in-flight TTFB start timestamps keyed by nothing (single value):
    /// pipecat tracks one TTFB at a time per service. We store the `ns` start.
    pub(crate) ttfb_start: Arc<AtomicI64>,
    /// Per-link processing-time start.
    pub(crate) processing_start: Arc<AtomicI64>,
}

impl Link {
    /// This processor's name (observer/error attribution).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Push a frame to the adjacent processor in `direction`. Mirrors pipecat
    /// `push_frame` (`frame_processor.py:688`): fires the observer `on_push` hook,
    /// then enqueues onto the neighbour's input channel. Backpressure: `await`s if
    /// the neighbour's bounded (normal) channel is full (PROCESSOR-DESIGN §2.2).
    pub async fn push(&self, meta: FrameMeta, frame: Frame, direction: Direction) {
        let tx = match direction {
            Direction::Downstream => self.next.as_ref(),
            Direction::Upstream => self.prev.as_ref(),
        };
        let Some(tx) = tx else {
            // No neighbour in this direction — the frame falls off the end of the
            // chain (the Source/Sink wrap normally observes it before this).
            return;
        };
        if let Some(o) = &self.observer {
            let dest = tx.name();
            let ev = crate::observer::FramePushEvent {
                source: &self.name,
                destination: dest,
                frame: &frame,
                meta: &meta,
                direction,
                timestamp_ns: self.clock.now_ns(),
            };
            o.on_push(&ev).await;
        }
        let env = Envelope {
            meta,
            frame,
            direction,
        };
        tx.send(env).await;
    }

    /// Convenience: push a fresh frame downstream with new meta.
    pub async fn push_down(&self, frame: Frame) {
        let meta = FrameMeta::new(&frame);
        self.push(meta, frame, Direction::Downstream).await;
    }

    /// Convenience: push a fresh frame upstream with new meta.
    pub async fn push_up(&self, frame: Frame) {
        let meta = FrameMeta::new(&frame);
        self.push(meta, frame, Direction::Upstream).await;
    }

    /// Push an [`Frame::Error`] upstream (pipecat `push_error`, :630).
    pub async fn push_error(&self, message: impl Into<String>, fatal: bool) {
        let frame = Frame::Error {
            message: message.into(),
            fatal,
            processor: Some(self.name.clone()),
        };
        self.push_up(frame).await;
    }

    /// Broadcast a frame **both** directions with paired sibling ids (pipecat
    /// `broadcast_frame`, :731) — used for [`Frame::Interruption`].
    pub async fn broadcast(&self, frame: Frame) {
        let down_id = next_frame_id();
        let up_id = next_frame_id();
        let mut down_meta = FrameMeta::new(&frame);
        down_meta.id = down_id;
        down_meta.broadcast_sibling_id = Some(up_id);
        let mut up_meta = FrameMeta::new(&frame);
        up_meta.id = up_id;
        up_meta.broadcast_sibling_id = Some(down_id);
        self.push(down_meta, frame.clone(), Direction::Downstream)
            .await;
        self.push(up_meta, frame, Direction::Upstream).await;
    }

    // ---- metrics helpers (PROCESSOR-DESIGN §5.2 / frame_processor.py:411-489) ----

    /// Mark the start of a TTFB measurement (no-op when metrics disabled).
    pub fn start_ttfb(&self) {
        if self.enable_metrics {
            self.ttfb_start
                .store(self.clock.now_ns(), Ordering::Relaxed);
        }
    }

    /// Emit a [`MetricsData::Ttfb`] downstream for the elapsed time since
    /// [`Link::start_ttfb`]. No-op when metrics are disabled or no start was set.
    pub async fn stop_ttfb(&self, model: Option<String>) {
        if !self.enable_metrics {
            return;
        }
        let start = self.ttfb_start.swap(0, Ordering::Relaxed);
        if start == 0 {
            return;
        }
        let seconds = (self.clock.now_ns() - start).max(0) as f64 / 1e9;
        self.emit_metric(MetricsData::Ttfb {
            processor: self.name.to_string(),
            model,
            seconds,
        })
        .await;
    }

    /// Mark the start of a processing-time measurement.
    pub fn start_processing(&self) {
        if self.enable_metrics {
            self.processing_start
                .store(self.clock.now_ns(), Ordering::Relaxed);
        }
    }

    /// Emit a [`MetricsData::Processing`] downstream for the elapsed time.
    pub async fn stop_processing(&self, model: Option<String>) {
        if !self.enable_metrics {
            return;
        }
        let start = self.processing_start.swap(0, Ordering::Relaxed);
        if start == 0 {
            return;
        }
        let seconds = (self.clock.now_ns() - start).max(0) as f64 / 1e9;
        self.emit_metric(MetricsData::Processing {
            processor: self.name.to_string(),
            model,
            seconds,
        })
        .await;
    }

    /// Emit an [`MetricsData::LlmUsage`] downstream (gated on usage metrics).
    pub async fn report_llm_usage(&self, model: Option<String>, tokens: metrics::LlmTokenUsage) {
        if !self.enable_usage_metrics {
            return;
        }
        self.emit_metric(MetricsData::LlmUsage {
            processor: self.name.to_string(),
            model,
            tokens,
        })
        .await;
    }

    /// Emit a [`MetricsData::TtsUsage`] downstream (gated on usage metrics).
    pub async fn report_tts_usage(&self, characters: u64) {
        if !self.enable_usage_metrics {
            return;
        }
        self.emit_metric(MetricsData::TtsUsage {
            processor: self.name.to_string(),
            characters,
        })
        .await;
    }

    async fn emit_metric(&self, data: MetricsData) {
        self.push_down(Frame::Metrics(vec![data])).await;
    }
}

/// The building block (PROCESSOR-DESIGN §2.1). Each processor runs in its own
/// tokio task fed by a bounded mpsc channel; the framework owns the task loop, an
/// impl writes only `process_frame` (and optional `start`/`stop`).
///
/// # Lifecycle frames bypass `process_frame` (read this before writing a processor)
///
/// The framework — not the processor — owns the lifecycle. The per-processor task
/// loop (`run_processor`) intercepts lifecycle frames and routes them to the hooks
/// instead of `process_frame`:
/// - [`Frame::Start`] → calls [`start`](FrameProcessor::start), then forwards.
/// - A **downstream** [`Frame::End`]/[`Frame::Stop`]/[`Frame::Cancel`] → calls
///   [`stop`](FrameProcessor::stop), forwards, and (End/Cancel) ends the task.
/// - [`Frame::Interruption`] → drains the interruptible backlog, then forwards.
///
/// So **your `process_frame` never sees these frames** — observe lifecycle via the
/// `start`/`stop` hooks (this is why the internal `Sink` taps from the hooks). An
/// *upstream* End/Stop/Cancel is the exception: it is a "request to end" and DOES
/// reach `process_frame` (the default forwards it upstream so the `Source` can
/// convert it to a downstream drain — pipecat's `EndTaskFrame` vs `EndFrame`). Keep
/// `process_frame` about data; do not try to handle Start/terminal frames there.
///
/// Mirrors pipecat `FrameProcessor` (`frame_processor.py:175`).
#[async_trait]
pub trait FrameProcessor: Send + 'static {
    /// Stable, human-readable name (observer events, error attribution, tracing).
    fn name(&self) -> &str;

    /// Called once when the [`Frame::Start`] frame reaches this processor, before
    /// any data frame. Open sockets / spawn provider readers here. Default: no-op.
    async fn start(&mut self, _setup: &ProcessorSetup, _params: &StartParams) -> Result<()> {
        Ok(())
    }

    /// Handle one frame. Push results via `link`. **Must not block**: long work
    /// (a provider round-trip) is driven by an internally-spawned task that feeds
    /// results back as frames. The default impl forwards the frame unchanged in
    /// its direction — so a pure observer/no-op processor is `process_frame` =
    /// default.
    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        link.push(env.meta, env.frame, env.direction).await;
        Ok(())
    }

    /// Called on `End`/`Stop`/`Cancel` after the terminal frame is forwarded.
    /// Flush + close. Default: no-op.
    async fn stop(&mut self, _reason: StopReason) -> Result<()> {
        Ok(())
    }

    /// Whether this processor produces metrics (pipecat `can_generate_metrics`,
    /// :395). Services override to `true`.
    fn can_generate_metrics(&self) -> bool {
        false
    }

    /// Framework-internal: a composite processor
    /// ([`Pipeline`](crate::pipeline::Pipeline) /
    /// [`ParallelPipeline`](crate::pipeline::ParallelPipeline)) exposes its internal
    /// wiring [`Topology`] here so it links transparently (pipecat nests
    /// `Pipeline(BasePipeline)`). Leaf processors return `None` (the default) and
    /// are spawned as one task. Called once at link time; a composite moves its
    /// children out (via `mem::take`) and must not be linked twice. **Object-safe**
    /// (`&mut self`) so it can be invoked on a `Box<dyn FrameProcessor>`.
    #[doc(hidden)]
    fn explode(&mut self) -> Option<Topology> {
        None
    }
}

/// Convert a boxed processor into a wiring [`Topology`]: a composite explodes into
/// its internal graph; a leaf becomes [`Topology::Leaf`]. The single entry point
/// the pipeline linker uses (object-safe, no `Sized` coercion in a default body).
pub fn into_topology(mut p: Box<dyn FrameProcessor>) -> Topology {
    match p.explode() {
        Some(topo) => topo,
        None => Topology::Leaf(p),
    }
}

/// The wiring shape a processor expands to when a [`crate::pipeline::Pipeline`]
/// links its members (PROCESSOR-DESIGN §3). A leaf is one spawned task; a `Chain`
/// is wired head→tail in order; a `Parallel` fans one input into every branch and
/// funnels/dedups the outputs. This is how `Pipeline`/`ParallelPipeline` nest
/// without a downcast.
pub enum Topology {
    /// A single processor → one spawned task.
    Leaf(Box<dyn FrameProcessor>),
    /// A linear chain wired head→tail (a `Pipeline`'s members).
    Chain(Vec<Topology>),
    /// A fan-out/fan-in block (a `ParallelPipeline`'s branches).
    Parallel(Vec<Topology>),
}
