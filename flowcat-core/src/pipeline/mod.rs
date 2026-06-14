// SPDX-License-Identifier: Apache-2.0
//
//! `Pipeline` + `ParallelPipeline` + `PipelineTask` + `PipelineRunner`
//! (PROCESSOR-DESIGN §3/§4).
//!
//! A [`Pipeline`] is a linear chain of [`FrameProcessor`] tasks connected by
//! bounded/unbounded mpsc channels (the channel runtime, `processor::runtime`).
//! It wraps the user processors with an internal **Source** (head) and **Sink**
//! (tail) so a [`PipelineTask`] can inject downstream frames at the head and
//! observe terminal frames at the tail. `Pipeline` is itself a `FrameProcessor`
//! (so it nests, pipecat `Pipeline(BasePipeline)`).
//!
//! The live call orchestration is the S2S/cascaded processor pipelines —
//! [`build_s2s_task`] (realtime) and [`build_cascaded_task`] (STT→LLM→TTS) — both
//! assembled on this framework.

pub mod cascaded;
pub mod parallel;
pub mod runner;
pub mod s2s;
pub mod source_pump;
pub mod task;

/// Shared scripted mock seams for the S2S pipeline tests (test-only). Drive the
/// processor [`PipelineTask`] (and the equivalence/fixture tests) off scripted
/// inputs. Gated `#[cfg(test)]` so it never ships.
#[cfg(test)]
pub(crate) mod s2s_test_mocks;

use std::sync::Arc;

use async_trait::async_trait;
use tokio::task::JoinSet;

use crate::error::Result;
use crate::processor::frame::{Direction, Frame, StartParams};
use crate::processor::runtime::{channel, ProcessorTx, NORMAL_CHAN_CAP};
use crate::processor::{
    into_topology, run_processor, Envelope, FrameProcessor, Link, ProcessorSetup, StopReason,
    Topology,
};

pub use cascaded::{
    build_cascaded_call_with_observers, build_cascaded_pipeline, build_cascaded_task,
    build_cascaded_task_with_observers, CascadedConfig, CascadedTask, ContextSummarizer,
    SummarizerConfig,
};
pub use parallel::ParallelPipeline;
pub use runner::PipelineRunner;
pub use source_pump::{SourceHandle, SourcePump};
pub use task::{PipelineTask, PipelineTaskParams};

/// A linear chain of linked processor tasks (PROCESSOR-DESIGN §3.1).
///
/// `Pipeline::new([a, b, c])` builds the chain `[Source, a, b, c, Sink]` when
/// linked. It is a [`FrameProcessor`] so it nests inside another `Pipeline` or a
/// [`ParallelPipeline`].
pub struct Pipeline {
    processors: Vec<Box<dyn FrameProcessor>>,
}

impl Pipeline {
    /// Build a pipeline over `processors` (wrapped with Source/Sink at link time).
    pub fn new(processors: Vec<Box<dyn FrameProcessor>>) -> Self {
        Self { processors }
    }

    /// Number of user processors (excludes the Source/Sink wrap).
    pub fn len(&self) -> usize {
        self.processors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.processors.is_empty()
    }
}

#[async_trait]
impl FrameProcessor for Pipeline {
    fn name(&self) -> &str {
        "Pipeline"
    }

    fn explode(&mut self) -> Option<Topology> {
        let procs = std::mem::take(&mut self.processors);
        Some(Topology::Chain(
            procs.into_iter().map(into_topology).collect(),
        ))
    }
}

// ---------------------------------------------------------------------------
// The linking machinery: turn a `Topology` into spawned, wired tasks.
// ---------------------------------------------------------------------------

/// A spawned, linked subgraph's external wiring handles.
pub(crate) struct Linked {
    /// Send a frame INTO this subgraph flowing downstream (its head input).
    pub head: ProcessorTx,
    /// Send a frame INTO this subgraph flowing upstream (its tail input). Part of
    /// the wiring contract for upstream injection at a composite's tail; not yet
    /// consumed by `PipelineTask` (which injects only at the head), so allow it.
    #[allow(dead_code)]
    pub tail: ProcessorTx,
}

/// Build a [`Link`] for one processor task.
fn make_link(
    name: Arc<str>,
    next: Option<ProcessorTx>,
    prev: Option<ProcessorTx>,
    setup: &ProcessorSetup,
) -> Link {
    Link {
        next,
        prev,
        name,
        clock: setup.clock.clone(),
        observer: setup.observer.clone(),
        enable_metrics: setup.enable_metrics,
        enable_usage_metrics: setup.enable_usage_metrics,
        ttfb_start: Arc::new(std::sync::atomic::AtomicI64::new(0)),
        processing_start: Arc::new(std::sync::atomic::AtomicI64::new(0)),
    }
}

/// Recursively spawn `topo` into wired tasks. `prev`/`next` are the senders to the
/// upstream/downstream neighbours *outside* this subgraph. Returns the subgraph's
/// external head/tail input senders. All spawned tasks are tracked in `tasks`.
pub(crate) fn link_topology(
    topo: Topology,
    prev: Option<ProcessorTx>,
    next: Option<ProcessorTx>,
    setup: &ProcessorSetup,
    tasks: &mut JoinSet<()>,
) -> Linked {
    match topo {
        Topology::Leaf(p) => {
            let name: Arc<str> = Arc::from(p.name());
            let (tx, rx) = channel(name.clone(), NORMAL_CHAN_CAP);
            let link = make_link(name, next, prev, setup);
            let setup2 = setup.clone();
            tasks.spawn(async move {
                run_processor(p, rx, link, setup2).await;
            });
            Linked {
                head: tx.clone(),
                tail: tx,
            }
        }
        Topology::Chain(elems) => link_chain(elems, prev, next, setup, tasks),
        Topology::Parallel(branches) => parallel::link_parallel(branches, prev, next, setup, tasks),
    }
}

/// Wire `elems` as a head→tail chain (PROCESSOR-DESIGN §3.1). Empty chain
/// degenerates to a pass-through no-op processor so the channels still connect.
///
/// Each element is linked first (so its external head/tail input senders exist),
/// then neighbours are wired in a second pass by re-spawning each element with the
/// adjacent head sender. To make wiring uniform across leaves and nested
/// composites, every element exposes a single *input* sender via a thin per-joint
/// forwarding channel — the chain wires those, and a forwarder bridges each into
/// the element's real head and the element's tail output back out.
fn link_chain(
    elems: Vec<Topology>,
    prev: Option<ProcessorTx>,
    next: Option<ProcessorTx>,
    setup: &ProcessorSetup,
    tasks: &mut JoinSet<()>,
) -> Linked {
    if elems.is_empty() {
        return link_topology(
            Topology::Leaf(Box::new(PassThrough)),
            prev,
            next,
            setup,
            tasks,
        );
    }
    let n = elems.len();

    // Per-element *input* sender that neighbours target. For a leaf this is the
    // processor's own channel; for a nested composite it is a forwarding channel
    // bridged to the composite's real head. We create them up front so that
    // neighbour wiring (each element's prev/next) is fully known before spawning.
    let mut inputs: Vec<ProcessorTx> = Vec::with_capacity(n);
    let mut pending: Vec<PendingElem> = Vec::with_capacity(n);
    for elem in elems {
        match elem {
            Topology::Leaf(p) => {
                let name: Arc<str> = Arc::from(p.name());
                let (tx, rx) = channel(name.clone(), NORMAL_CHAN_CAP);
                inputs.push(tx);
                pending.push(PendingElem::Leaf { name, p, rx });
            }
            other => {
                // Forwarding input channel for the composite; bridged below.
                let name: Arc<str> = Arc::from("nested");
                let (tx, rx) = channel(name, NORMAL_CHAN_CAP);
                inputs.push(tx);
                pending.push(PendingElem::Composite {
                    topo: other,
                    in_rx: rx,
                });
            }
        }
    }

    // Second pass: resolve neighbours and spawn.
    for (i, elem) in pending.into_iter().enumerate() {
        let next_tx = if i + 1 < n {
            Some(inputs[i + 1].clone())
        } else {
            next.clone()
        };
        let prev_tx = if i > 0 {
            Some(inputs[i - 1].clone())
        } else {
            prev.clone()
        };
        match elem {
            PendingElem::Leaf { name, p, rx } => {
                let link = make_link(name, next_tx, prev_tx, setup);
                let setup2 = setup.clone();
                tasks.spawn(async move {
                    run_processor(p, rx, link, setup2).await;
                });
            }
            PendingElem::Composite { topo, in_rx } => {
                // Link the composite with the resolved neighbours, then bridge the
                // forwarding input channel into its real head.
                let linked = link_topology(topo, prev_tx, next_tx, setup, tasks);
                let real_head = linked.head;
                tasks.spawn(bridge(in_rx, real_head));
            }
        }
    }

    Linked {
        head: inputs[0].clone(),
        tail: inputs[n - 1].clone(),
    }
}

/// A chain element awaiting its second-pass neighbour wiring.
enum PendingElem {
    Leaf {
        name: Arc<str>,
        p: Box<dyn FrameProcessor>,
        rx: crate::processor::runtime::ProcessorRx,
    },
    Composite {
        topo: Topology,
        in_rx: crate::processor::runtime::ProcessorRx,
    },
}

/// Forward every envelope from `rx` into `dst` (bridges a forwarding input channel
/// to a nested composite's real head input). Exits after a terminal frame so the
/// task tree winds down.
pub(crate) async fn bridge(mut rx: crate::processor::runtime::ProcessorRx, dst: ProcessorTx) {
    loop {
        let env = tokio::select! {
            biased;
            Some(e) = rx.system.recv() => e,
            Some(e) = rx.normal.recv() => e,
            else => break,
        };
        let terminal = env.frame.is_terminal();
        dst.send(env).await;
        if terminal {
            break;
        }
    }
}

/// A no-op pass-through processor (default `process_frame` forwards).
struct PassThrough;

#[async_trait]
impl FrameProcessor for PassThrough {
    fn name(&self) -> &str {
        "PassThrough"
    }
}

// ---------------------------------------------------------------------------
// Source / Sink — the internal head/tail wrap (PROCESSOR-DESIGN §3.1).
// ---------------------------------------------------------------------------

/// The pipeline head wrapper (pipecat `PipelineSource`, `pipeline.py:21`).
///
/// The `PipelineTask` injects downstream frames here; it converts an **upstream**
/// `End`/`Stop`/`Cancel` *request* from an inner processor into the corresponding
/// downstream lifecycle frame (PROCESSOR-DESIGN §4.1 step 4, `_source_push_frame`).
pub(crate) struct Source;

#[async_trait]
impl FrameProcessor for Source {
    fn name(&self) -> &str {
        "Source"
    }

    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        // Upstream lifecycle request → convert to a downstream lifecycle frame so
        // a processor wanting to end the call drives the whole chain to drain.
        if env.direction == Direction::Upstream {
            match &env.frame {
                Frame::End { .. } | Frame::Stop | Frame::Cancel { .. } => {
                    link.push_down(env.frame).await;
                    return Ok(());
                }
                _ => {}
            }
        }
        link.push(env.meta, env.frame, env.direction).await;
        Ok(())
    }
}

/// The pipeline tail wrapper (pipecat `PipelineSink`, `pipeline.py:55`).
///
/// Mirrors downstream frames out to the `PipelineTask`'s observation channel so it
/// can detect ready/terminal/heartbeat frames. Forwards everything else.
pub(crate) struct Sink {
    /// A clone of the task's "sink observation" sender.
    pub(crate) tap: tokio::sync::mpsc::UnboundedSender<Frame>,
}

#[async_trait]
impl FrameProcessor for Sink {
    fn name(&self) -> &str {
        "Sink"
    }

    /// Lifecycle frames (`Start`, `End`/`Stop`/`Cancel`) are handled by the
    /// framework loop (`run_processor`) via the `start`/`stop` hooks and **never
    /// reach `process_frame`** (PROCESSOR-DESIGN §2.2). So the Sink must tap them
    /// here, from the hooks — otherwise the `PipelineTask` ready-handshake (it
    /// blocks until `Start` reaches the Sink) and its terminal-frame detection
    /// would never fire. Tapping `Start` from the hook makes the handshake work;
    /// tapping the reconstructed terminal frame from `stop` gives the task the
    /// correct `StopReason` (End→EndOfTask, Stop→Stopped, Cancel→Cancelled).
    async fn start(&mut self, _setup: &ProcessorSetup, params: &StartParams) -> Result<()> {
        let _ = self.tap.send(Frame::Start(params.clone()));
        Ok(())
    }

    async fn stop(&mut self, reason: StopReason) -> Result<()> {
        let frame = match reason {
            StopReason::EndOfTask => Frame::End { reason: None },
            StopReason::Stopped => Frame::Stop,
            StopReason::Cancelled => Frame::Cancel { reason: None },
        };
        let _ = self.tap.send(frame);
        Ok(())
    }

    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        if env.direction == Direction::Downstream {
            let _ = self.tap.send(env.frame.clone());
        }
        link.push(env.meta, env.frame, env.direction).await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processor::frame::{Frame, StartParams};
    use crate::processor::Clock;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc as StdArc;
    use tokio::sync::mpsc;

    // A processor that records the order of Text frames it sees, then forwards.
    struct Recorder {
        name: &'static str,
        log: StdArc<std::sync::Mutex<Vec<String>>>,
    }
    #[async_trait]
    impl FrameProcessor for Recorder {
        fn name(&self) -> &str {
            self.name
        }
        async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
            if let Frame::Text(t) = &env.frame {
                self.log
                    .lock()
                    .unwrap()
                    .push(format!("{}:{}", self.name, t));
            }
            link.push(env.meta, env.frame, env.direction).await;
            Ok(())
        }
    }

    fn setup() -> ProcessorSetup {
        ProcessorSetup {
            clock: Clock::new(),
            observer: None,
            cancel: tokio_util::sync::CancellationToken::new(),
            enable_metrics: false,
            enable_usage_metrics: false,
        }
    }

    #[tokio::test]
    async fn pipeline_is_a_processor_and_nests_reaching_the_sink() {
        let log = StdArc::new(std::sync::Mutex::new(Vec::new()));
        // Outer pipeline contains an inner pipeline → tests nesting.
        let inner = Pipeline::new(vec![Box::new(Recorder {
            name: "inner",
            log: log.clone(),
        })]);
        let outer = Pipeline::new(vec![
            Box::new(Recorder {
                name: "a",
                log: log.clone(),
            }),
            Box::new(inner),
            Box::new(Recorder {
                name: "b",
                log: log.clone(),
            }),
        ]);

        let (sink_tap_tx, mut sink_tap_rx) = mpsc::unbounded_channel::<Frame>();
        // Build [Source, outer..., Sink] and link.
        let topo = Topology::Chain(vec![
            Topology::Leaf(Box::new(Source)),
            into_topology(Box::new(outer)),
            Topology::Leaf(Box::new(Sink { tap: sink_tap_tx })),
        ]);
        let st = setup();
        let mut tasks = JoinSet::new();
        let linked = link_topology(topo, None, None, &st, &mut tasks);

        // Inject a downstream Text at the head.
        linked
            .head
            .send(Envelope::new(
                Frame::Text("hi".into()),
                Direction::Downstream,
            ))
            .await;
        // It should reach the Sink (tapped) after passing a, inner, b in order.
        let got = tokio::time::timeout(std::time::Duration::from_secs(2), sink_tap_rx.recv())
            .await
            .expect("sink frame timed out")
            .expect("sink closed");
        assert!(matches!(got, Frame::Text(t) if t == "hi"));

        // Send End to drain.
        linked
            .head
            .send(Envelope::new(
                Frame::End { reason: None },
                Direction::Downstream,
            ))
            .await;
        let _ = tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while tasks.join_next().await.is_some() {}
        })
        .await;

        let order = log.lock().unwrap().clone();
        assert_eq!(order, vec!["a:hi", "inner:hi", "b:hi"]);
    }

    // A leaf that counts Start frames it received before any data frame.
    struct StartCounter {
        seen_data_before_start: StdArc<AtomicUsize>,
        started: StdArc<AtomicUsize>,
    }
    #[async_trait]
    impl FrameProcessor for StartCounter {
        fn name(&self) -> &str {
            "StartCounter"
        }
        async fn start(&mut self, _s: &ProcessorSetup, _p: &StartParams) -> Result<()> {
            self.started.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
        async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
            if matches!(env.frame, Frame::Text(_)) && self.started.load(Ordering::Relaxed) == 0 {
                self.seen_data_before_start.fetch_add(1, Ordering::Relaxed);
            }
            link.push(env.meta, env.frame, env.direction).await;
            Ok(())
        }
    }

    #[tokio::test]
    async fn start_runs_before_data_in_a_linked_chain() {
        let started = StdArc::new(AtomicUsize::new(0));
        let bad = StdArc::new(AtomicUsize::new(0));
        let (sink_tap_tx, _rx) = mpsc::unbounded_channel::<Frame>();
        let topo = Topology::Chain(vec![
            Topology::Leaf(Box::new(Source)),
            Topology::Leaf(Box::new(StartCounter {
                seen_data_before_start: bad.clone(),
                started: started.clone(),
            })),
            Topology::Leaf(Box::new(Sink { tap: sink_tap_tx })),
        ]);
        let st = setup();
        let mut tasks = JoinSet::new();
        let linked = link_topology(topo, None, None, &st, &mut tasks);
        linked
            .head
            .send(Envelope::new(
                Frame::Start(StartParams::default()),
                Direction::Downstream,
            ))
            .await;
        // small yield so Start propagates
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        linked
            .head
            .send(Envelope::new(
                Frame::Text("data".into()),
                Direction::Downstream,
            ))
            .await;
        linked
            .head
            .send(Envelope::new(
                Frame::End { reason: None },
                Direction::Downstream,
            ))
            .await;
        let _ = tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while tasks.join_next().await.is_some() {}
        })
        .await;
        assert_eq!(started.load(Ordering::Relaxed), 1);
        assert_eq!(
            bad.load(Ordering::Relaxed),
            0,
            "data must not precede start"
        );
    }
}
