// SPDX-License-Identifier: Apache-2.0
//
//! [`SourcePump`] — the standard, ordering-safe way for a **source** processor (a
//! transport/socket reader that has no upstream frame to react to) to inject frames
//! into a running [`PipelineTask`](super::PipelineTask)'s head.
//!
//! ## Why a helper and not a trait hook (the source-emit ruling, §10 Q6)
//!
//! A [`FrameProcessor`](crate::processor::FrameProcessor)'s
//! [`start`](crate::processor::FrameProcessor::start) hook receives no
//! [`Link`](crate::processor::Link), so a *source* (a `recv()` loop) cannot
//! self-emit from the frozen trait. Rather than grow the just-frozen trait with a
//! runtime-spawned `run_source`, the framework **codifies the external-pump
//! pattern**: a source reads its transport in its own task and feeds the pipeline
//! head via [`PipelineTask::queue_sender`](super::PipelineTask::queue_sender). This
//! is exactly pipecat's `BaseInputTransport` reader-task model — the source's
//! reader is hosted *beside* the pipeline, not inside a lifecycle method.
//!
//! Feeding the **head** (not a mid-chain `push`) is what preserves the
//! Start→ready ordering guarantee every processor relies on:
//! [`PipelineTask::run`](super::PipelineTask::run) injects `Start` and **blocks on
//! the Start→Sink handshake before it drains the head queue**, so a pumped
//! `InputAudio` can never reach any processor's `process_frame` before that
//! processor's `start()` ran (the invariant the code-reviewer flagged for
//! `.expect()`-in-`process_frame` processors). `SourcePump` is the one-liner that
//! gives every transport author this behaviour without re-deriving the
//! spawn + abort + ordering dance.
//!
//! ## Backpressure
//!
//! The head queue is unbounded (a source must never block delivering capture —
//! input audio is wall-clock-rate-limited, PROCESSOR-DESIGN §2.2). Backpressure
//! lives one hop later: the head's bounded *normal* channel to the first real
//! processor `await`s when full, so a stalled downstream still applies natural
//! backpressure to the producer via that channel — the source pump itself does not
//! buffer unboundedly in practice because each `emit` returns immediately and the
//! `PipelineTask` head pump forwards into the bounded channel.
//!
//! ## Usage
//!
//! ```ignore
//! let task = PipelineTask::new(pipeline, params, vec![]);
//! // Spawn a reader that drives a transport's recv() into the head:
//! let pump = SourcePump::spawn(task.queue_sender(), |head| async move {
//!     while let Some(media) = transport.recv().await {
//!         match media {
//!             MediaIn::Audio(chunk) => {
//!                 if head.emit(Frame::InputAudio(Arc::new((&chunk).into()))).is_err() {
//!                     break; // pipeline gone
//!                 }
//!             }
//!             MediaIn::Stop | /* exhausted */ _ => { head.end(); break; }
//!         }
//!     }
//! });
//! task.run().await?;       // pump is aborted on drop after run returns
//! ```

use tokio::sync::mpsc::UnboundedSender;
use tokio::task::JoinHandle;

use crate::processor::frame::Frame;

/// A clonable handle a source's reader task uses to inject frames into the pipeline
/// head. Thin wrapper over the [`PipelineTask`](super::PipelineTask) head queue so
/// transport authors call `emit`/`end` instead of touching the raw channel.
#[derive(Clone)]
pub struct SourceHandle {
    head: UnboundedSender<Frame>,
}

impl SourceHandle {
    /// Wrap a [`PipelineTask::queue_sender`](super::PipelineTask::queue_sender)
    /// handle.
    pub fn new(head: UnboundedSender<Frame>) -> Self {
        Self { head }
    }

    /// Inject one frame at the pipeline head (flows downstream). Returns `Err` if
    /// the pipeline has shut down (the reader should then stop). Mirrors pipecat
    /// `BaseInputTransport.push_frame` from the reader task.
    pub fn emit(&self, frame: Frame) -> Result<(), Frame> {
        self.head.send(frame).map_err(|e| e.0)
    }

    /// Request a graceful drain: inject a downstream `End` at the head so every
    /// processor (incl. the internal `Source`) flushes and the task winds down.
    /// Idempotent and a no-op once the pipeline is gone.
    pub fn end(&self) {
        let _ = self.head.send(Frame::End { reason: None });
    }

    /// Request an immediate, no-flush teardown: inject a downstream `Cancel`.
    pub fn cancel(&self, reason: Option<String>) {
        let _ = self.head.send(Frame::Cancel { reason });
    }
}

/// Owns a source's spawned reader task and aborts it on drop. Holds the pump+abort
/// lifecycle so a transport author never re-derives it (the source-emit
/// ruling — see the module docs / PROCESSOR-DESIGN §10 Q6).
pub struct SourcePump {
    handle: JoinHandle<()>,
}

impl SourcePump {
    /// Spawn a source reader. `f` receives a [`SourceHandle`] and runs the
    /// transport's `recv()` loop, `emit`ing frames into the pipeline head until the
    /// transport is exhausted (then it should call [`SourceHandle::end`]). The
    /// returned [`SourcePump`] aborts the reader on drop, so dropping it after
    /// [`PipelineTask::run`](super::PipelineTask::run) returns cleans the reader up.
    pub fn spawn<F, Fut>(head: UnboundedSender<Frame>, f: F) -> Self
    where
        F: FnOnce(SourceHandle) -> Fut + Send + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        let handle = tokio::spawn(f(SourceHandle::new(head)));
        Self { handle }
    }

    /// Wrap an already-spawned reader `handle` (for sources that build their own
    /// task — e.g. one that needs the [`JoinHandle`] before the loop body). The
    /// reader is still expected to feed the head via a [`SourceHandle`].
    pub fn from_handle(handle: JoinHandle<()>) -> Self {
        Self { handle }
    }

    /// Stop the reader task immediately (used after `run()` returns).
    pub fn abort(&self) {
        self.handle.abort();
    }
}

impl Drop for SourcePump {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{Pipeline, PipelineTask, PipelineTaskParams};
    use crate::processor::frame::{Frame, StartParams};
    use crate::processor::{Envelope, FrameProcessor, Link, ProcessorSetup};
    use crate::Result;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// A processor that asserts its `start()` ran before any data frame, and counts
    /// the `Text` frames the source pump fed it — proving the Start→ready handshake
    /// holds for pump-injected frames (the ordering invariant the helper preserves).
    struct Guard {
        started: Arc<AtomicUsize>,
        data_before_start: Arc<AtomicUsize>,
        data_after_start: Arc<AtomicUsize>,
    }
    #[async_trait]
    impl FrameProcessor for Guard {
        fn name(&self) -> &str {
            "Guard"
        }
        async fn start(&mut self, _s: &ProcessorSetup, _p: &StartParams) -> Result<()> {
            self.started.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
            if matches!(env.frame, Frame::Text(_)) {
                if self.started.load(Ordering::SeqCst) == 0 {
                    self.data_before_start.fetch_add(1, Ordering::SeqCst);
                } else {
                    self.data_after_start.fetch_add(1, Ordering::SeqCst);
                }
            }
            link.push(env.meta, env.frame, env.direction).await;
            Ok(())
        }
    }

    #[tokio::test]
    async fn source_pump_feeds_head_and_respects_start_handshake() {
        let started = Arc::new(AtomicUsize::new(0));
        let before = Arc::new(AtomicUsize::new(0));
        let after = Arc::new(AtomicUsize::new(0));
        let pipeline = Pipeline::new(vec![Box::new(Guard {
            started: started.clone(),
            data_before_start: before.clone(),
            data_after_start: after.clone(),
        })]);
        let task = PipelineTask::new(pipeline, PipelineTaskParams::default(), vec![]);

        // A source reader that emits three Text frames then ends the call.
        let pump = SourcePump::spawn(task.queue_sender(), |head| async move {
            for i in 0..3 {
                if head.emit(Frame::Text(format!("m{i}"))).is_err() {
                    return;
                }
            }
            head.end();
        });

        tokio::time::timeout(std::time::Duration::from_secs(5), task.run())
            .await
            .expect("task timed out")
            .expect("run ok");
        drop(pump); // aborts the (already-finished) reader

        assert_eq!(started.load(Ordering::SeqCst), 1, "start ran once");
        assert_eq!(
            before.load(Ordering::SeqCst),
            0,
            "no pumped data reached process_frame before start()"
        );
        assert_eq!(
            after.load(Ordering::SeqCst),
            3,
            "all 3 pumped frames processed"
        );
    }

    #[tokio::test]
    async fn source_pump_aborts_reader_on_drop() {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel::<Frame>();
        let ticks = Arc::new(AtomicUsize::new(0));
        let t2 = ticks.clone();
        let pump = SourcePump::spawn(tx, |_head| async move {
            // Loop forever until aborted.
            loop {
                t2.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        });
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        drop(pump);
        let seen = ticks.load(Ordering::SeqCst);
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        assert_eq!(
            ticks.load(Ordering::SeqCst),
            seen,
            "reader task stopped ticking after the pump was dropped"
        );
    }
}
