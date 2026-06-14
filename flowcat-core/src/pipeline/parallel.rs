// SPDX-License-Identifier: Apache-2.0
//
//! `ParallelPipeline` — fan-out / fan-in with lifecycle sync
//! (PROCESSOR-DESIGN §3.2).
//!
//! A frame entering the parallel block is queued into **every** branch's head.
//! Each branch's tail output funnels into a single downstream, **de-duplicated by
//! `meta.id`** (a frame fanned to N branches is emitted once). Lifecycle frames
//! (`Start`/`End`/`Cancel`) are **synchronized**: the block buffers non-lifecycle
//! output while a lifecycle frame is in flight and only forwards the lifecycle
//! frame (and flushes the buffer) once *all* branches have passed it — so a fast
//! branch's `End` can't shut the transport down while a slow branch still has
//! audio to flush. Mirrors pipecat `ParallelPipeline` (`parallel_pipeline.py:24`).

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::task::JoinSet;

use crate::processor::frame::Frame;
use crate::processor::runtime::{channel, ProcessorRx, ProcessorTx, NORMAL_CHAN_CAP};
use crate::processor::{into_topology, Envelope, FrameProcessor, ProcessorSetup, Topology};

use super::{link_topology, Linked, Pipeline};

/// A fan-out / fan-in block of parallel [`Pipeline`] branches
/// (PROCESSOR-DESIGN §3.2). Itself a [`FrameProcessor`] (so it nests).
pub struct ParallelPipeline {
    branches: Vec<Pipeline>,
}

impl ParallelPipeline {
    /// Build a parallel block over `branches`.
    pub fn new(branches: Vec<Pipeline>) -> Self {
        Self { branches }
    }

    /// Number of branches.
    pub fn len(&self) -> usize {
        self.branches.len()
    }

    pub fn is_empty(&self) -> bool {
        self.branches.is_empty()
    }
}

#[async_trait]
impl FrameProcessor for ParallelPipeline {
    fn name(&self) -> &str {
        "ParallelPipeline"
    }

    fn explode(&mut self) -> Option<Topology> {
        let branches = std::mem::take(&mut self.branches);
        Some(Topology::Parallel(
            branches
                .into_iter()
                .map(|b| into_topology(Box::new(b)))
                .collect(),
        ))
    }
}

/// Spawn a parallel block: fan `prev→branches→next` with id-dedup + lifecycle
/// sync. Returns the block's external head/tail input senders.
pub(crate) fn link_parallel(
    branches: Vec<Topology>,
    prev: Option<ProcessorTx>,
    next: Option<ProcessorTx>,
    setup: &ProcessorSetup,
    tasks: &mut JoinSet<()>,
) -> Linked {
    let branch_count = branches.len().max(1);

    // The block's downstream head: a forwarding channel the fan-out task reads.
    let (head_tx, head_rx) = channel(Arc::from("ParallelIn"), NORMAL_CHAN_CAP);
    // The block's upstream tail: branches push upstream here; we forward to `prev`.
    let (tail_tx, tail_rx) = channel(Arc::from("ParallelOut"), NORMAL_CHAN_CAP);

    // Fan-in collector: every branch tail pushes its DOWNSTREAM output here.
    let (fanin_tx, fanin_rx) = channel(Arc::from("ParallelFanIn"), NORMAL_CHAN_CAP);
    // Upstream collector: every branch head pushes UPSTREAM here.
    let (upstream_tx, upstream_rx) = channel(Arc::from("ParallelUpstream"), NORMAL_CHAN_CAP);

    // Link each branch with prev = upstream_collector, next = fanin_collector.
    let mut branch_heads: Vec<ProcessorTx> = Vec::with_capacity(branch_count);
    for branch in branches {
        let linked = link_topology(
            branch,
            Some(upstream_tx.clone()),
            Some(fanin_tx.clone()),
            setup,
            tasks,
        );
        branch_heads.push(linked.head);
    }
    drop(fanin_tx);
    drop(upstream_tx);

    // Fan-out task: read head_rx, push each frame into EVERY branch head.
    tasks.spawn(fan_out(head_rx, branch_heads));

    // Fan-in task: dedup by id + lifecycle sync, then push to `next`.
    tasks.spawn(fan_in(fanin_rx, next, branch_count));

    // Upstream forwarder: branches → `prev` (no dedup needed; pass through).
    tasks.spawn(forward(upstream_rx, prev));

    // Tail forwarder: upstream injection at a parallel block's tail is not wired
    // (a `PipelineTask` only injects at the head — see `Linked::tail`'s
    // `#[allow(dead_code)]` in `pipeline/mod.rs`). We drain `tail_rx` to `None` so a
    // stray tail injection is dropped deliberately, not silently queued forever. If
    // tail-upstream injection is needed in the future, route this into the upstream
    // collector path instead of `None`.
    tasks.spawn(forward(tail_rx, None));

    Linked {
        head: head_tx,
        tail: tail_tx,
    }
}

/// Push every frame from `rx` into every branch head (the fan-out).
async fn fan_out(mut rx: ProcessorRx, branch_heads: Vec<ProcessorTx>) {
    loop {
        let env = tokio::select! {
            biased;
            Some(e) = rx.system.recv() => e,
            Some(e) = rx.normal.recv() => e,
            else => break,
        };
        let terminal = env.frame.is_terminal();
        for head in &branch_heads {
            head.send(env.clone()).await;
        }
        if terminal {
            break;
        }
    }
}

/// Collect downstream output from all branches; dedup by `meta.id`; synchronize
/// lifecycle frames (forward once all branches passed them, buffering other
/// output meanwhile); forward to `next`.
async fn fan_in(mut rx: ProcessorRx, next: Option<ProcessorTx>, branch_count: usize) {
    let mut seen_ids: HashSet<u64> = HashSet::new();
    // For the *currently-syncing* lifecycle frame: how many branches have passed.
    let mut lifecycle_pending: Option<(Frame, usize)> = None;
    let mut buffer: Vec<Envelope> = Vec::new();

    loop {
        let env = tokio::select! {
            biased;
            Some(e) = rx.system.recv() => e,
            Some(e) = rx.normal.recv() => e,
            else => break,
        };

        if is_synced_lifecycle(&env.frame) {
            // A lifecycle frame from one branch. Count distinct sibling arrivals
            // by id-dedup: each branch emits its own copy (distinct id). Track by
            // a per-lifecycle counter keyed on the frame kind.
            match &mut lifecycle_pending {
                Some((pending, count)) if same_lifecycle(pending, &env.frame) => {
                    *count += 1;
                    if *count >= branch_count {
                        // All branches passed it: forward the lifecycle frame, then
                        // flush the buffer, then reset.
                        forward_one(&next, env).await;
                        for buffered in buffer.drain(..) {
                            forward_one(&next, buffered).await;
                        }
                        lifecycle_pending = None;
                    }
                }
                Some(_) => {
                    // A different lifecycle while one is pending — buffer it.
                    buffer.push(env);
                }
                None => {
                    lifecycle_pending = Some((env.frame.clone(), 1));
                    if branch_count <= 1 {
                        // Single branch: forward immediately.
                        forward_one(&next, env).await;
                        lifecycle_pending = None;
                    }
                }
            }
            continue;
        }

        // Non-lifecycle frame: dedup by id.
        if !seen_ids.insert(env.meta.id) {
            continue; // already emitted this fanned frame
        }
        if lifecycle_pending.is_some() {
            // Buffer non-lifecycle output until the lifecycle sync completes.
            buffer.push(env);
        } else {
            forward_one(&next, env).await;
        }
    }
}

/// Plain forwarder: drain `rx` into `dst` (or drop if `dst` is `None`).
async fn forward(mut rx: ProcessorRx, dst: Option<ProcessorTx>) {
    loop {
        let env = tokio::select! {
            biased;
            Some(e) = rx.system.recv() => e,
            Some(e) = rx.normal.recv() => e,
            else => break,
        };
        forward_one(&dst, env).await;
    }
}

async fn forward_one(dst: &Option<ProcessorTx>, env: Envelope) {
    if let Some(tx) = dst {
        tx.send(env).await;
    }
}

/// Lifecycle frames whose fan-in must be synchronized across all branches.
fn is_synced_lifecycle(frame: &Frame) -> bool {
    matches!(
        frame,
        Frame::Start(_) | Frame::End { .. } | Frame::Cancel { .. }
    )
}

/// Whether two lifecycle frames are "the same" event for sync counting.
fn same_lifecycle(a: &Frame, b: &Frame) -> bool {
    matches!(
        (a, b),
        (Frame::Start(_), Frame::Start(_))
            | (Frame::End { .. }, Frame::End { .. })
            | (Frame::Cancel { .. }, Frame::Cancel { .. })
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result;
    use crate::processor::frame::{Direction, Frame};
    use crate::processor::{Clock, Link};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    use std::time::Duration;

    fn setup() -> ProcessorSetup {
        ProcessorSetup {
            clock: Clock::new(),
            observer: None,
            cancel: tokio_util::sync::CancellationToken::new(),
            enable_metrics: false,
            enable_usage_metrics: false,
        }
    }

    // Counts how many Text frames it processed.
    struct Counter {
        name: &'static str,
        count: Arc<AtomicUsize>,
    }
    #[async_trait]
    impl FrameProcessor for Counter {
        fn name(&self) -> &str {
            self.name
        }
        async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
            if matches!(env.frame, Frame::Text(_)) {
                self.count.fetch_add(1, Ordering::Relaxed);
            }
            link.push(env.meta, env.frame, env.direction).await;
            Ok(())
        }
    }

    // A branch element that delays a configurable time on its terminal End, to
    // exercise the lifecycle sync (a fast branch must wait for the slow one).
    struct SlowEnd {
        name: &'static str,
        delay_ms: u64,
        end_order: Arc<Mutex<Vec<&'static str>>>,
    }
    #[async_trait]
    impl FrameProcessor for SlowEnd {
        fn name(&self) -> &str {
            self.name
        }
        async fn stop(&mut self, _r: crate::processor::StopReason) -> Result<()> {
            if self.delay_ms > 0 {
                tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            }
            self.end_order.lock().unwrap().push(self.name);
            Ok(())
        }
    }

    #[tokio::test]
    async fn frame_fans_to_n_branches_and_emits_once() {
        let c1 = Arc::new(AtomicUsize::new(0));
        let c2 = Arc::new(AtomicUsize::new(0));
        let b1 = Pipeline::new(vec![Box::new(Counter {
            name: "b1",
            count: c1.clone(),
        })]);
        let b2 = Pipeline::new(vec![Box::new(Counter {
            name: "b2",
            count: c2.clone(),
        })]);

        let topo = into_topology(Box::new(ParallelPipeline::new(vec![b1, b2])));

        // Capture the single de-duplicated downstream output.
        let (out_tx, out_rx) = channel(Arc::from("out"), NORMAL_CHAN_CAP);
        let st = setup();
        let mut tasks = JoinSet::new();
        let linked = link_topology(topo, None, Some(out_tx), &st, &mut tasks);

        // One Text frame fanned to both branches.
        linked
            .head
            .send(Envelope::new(
                Frame::Text("x".into()),
                Direction::Downstream,
            ))
            .await;

        // It should emit exactly once downstream (deduped).
        let mut out_rx = out_rx;
        let first = tokio::time::timeout(Duration::from_secs(2), out_rx.normal.recv())
            .await
            .expect("timed out")
            .expect("closed");
        assert!(matches!(first.frame, Frame::Text(t) if t == "x"));
        // No second copy within a short window.
        let second = tokio::time::timeout(Duration::from_millis(200), out_rx.normal.recv()).await;
        assert!(second.is_err(), "fanned frame must emit once, got a dup");

        // Both branches processed it.
        linked
            .head
            .send(Envelope::new(
                Frame::End { reason: None },
                Direction::Downstream,
            ))
            .await;
        let _ = tokio::time::timeout(Duration::from_secs(2), async {
            while tasks.join_next().await.is_some() {}
        })
        .await;
        assert_eq!(c1.load(Ordering::Relaxed), 1);
        assert_eq!(c2.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn fast_branch_end_is_held_until_slow_branch_passes() {
        let order = Arc::new(Mutex::new(Vec::new()));
        let fast = Pipeline::new(vec![Box::new(SlowEnd {
            name: "fast",
            delay_ms: 0,
            end_order: order.clone(),
        })]);
        let slow = Pipeline::new(vec![Box::new(SlowEnd {
            name: "slow",
            delay_ms: 150,
            end_order: order.clone(),
        })]);
        let topo = into_topology(Box::new(ParallelPipeline::new(vec![fast, slow])));

        let (out_tx, out_rx) = channel(Arc::from("out"), NORMAL_CHAN_CAP);
        let st = setup();
        let mut tasks = JoinSet::new();
        let linked = link_topology(topo, None, Some(out_tx), &st, &mut tasks);

        linked
            .head
            .send(Envelope::new(
                Frame::End { reason: None },
                Direction::Downstream,
            ))
            .await;

        // The synchronized End should only reach `next` after BOTH branches'
        // stop() ran — i.e. after the slow branch.
        let mut out_rx = out_rx;
        let got = tokio::time::timeout(Duration::from_secs(3), out_rx.normal.recv())
            .await
            .expect("End never synced")
            .expect("closed");
        assert!(matches!(got.frame, Frame::End { .. }));

        let seq = order.lock().unwrap().clone();
        assert!(
            seq.contains(&"fast") && seq.contains(&"slow"),
            "both branches must pass End, got {seq:?}"
        );

        let _ = tokio::time::timeout(Duration::from_secs(2), async {
            while tasks.join_next().await.is_some() {}
        })
        .await;
    }
}
