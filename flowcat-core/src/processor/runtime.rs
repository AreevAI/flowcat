// SPDX-License-Identifier: Apache-2.0
//
//! The framework-owned per-processor task loop (PROCESSOR-DESIGN §2.2/§2.3/§2.5).
//!
//! Two channels per processor so `System` frames jump ahead of `Data`/`Control`
//! (pipecat does this with a `PriorityQueue`, `frame_processor.py:119`; a second
//! channel is the cheaper, branch-free Rust equivalent):
//!
//! - **bounded** normal channel (Data/Control) → natural backpressure for media;
//! - **unbounded** system channel (Start/Cancel/Interruption/Error/…) → an
//!   interruption can never block on a full queue.
//!
//! The loop biases the system channel, runs the processor's lifecycle hooks on
//! lifecycle frames, drains interruptible frames on [`Frame::Interruption`], and
//! converts a `process_frame` `Err` into an upstream [`Frame::Error`].

use tokio::sync::mpsc;

use super::frame::{Direction, Frame, FrameClass};
use super::{Envelope, FrameProcessor, Link, ProcessorSetup, StopReason};

/// Default capacity of the bounded normal (Data/Control) channel — the
/// `bench-rs` value (~1.3 s of audio headroom @ 50 fps). PROCESSOR-DESIGN §2.2.
pub const NORMAL_CHAN_CAP: usize = 64;

/// The receive half held by a processor task: a bounded normal channel and an
/// unbounded system channel.
pub struct ProcessorRx {
    pub(crate) system: mpsc::UnboundedReceiver<Envelope>,
    pub(crate) normal: mpsc::Receiver<Envelope>,
}

/// The send half: routes a frame by [`Frame::class`] (System → the unbounded
/// system channel, Data/Control → the bounded normal channel). Cheaply cloned
/// into each neighbour's [`Link`].
#[derive(Clone)]
pub struct ProcessorTx {
    system: mpsc::UnboundedSender<Envelope>,
    normal: mpsc::Sender<Envelope>,
    name: std::sync::Arc<str>,
}

impl ProcessorTx {
    /// The destination processor's name (for observer push events).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Enqueue `env` onto the right channel for its class. Backpressure (`await`)
    /// only on the bounded normal channel; the system channel never blocks. A
    /// closed channel is a no-op (the consumer task already exited).
    pub async fn send(&self, env: Envelope) {
        match env.frame.class() {
            FrameClass::System => {
                let _ = self.system.send(env);
            }
            FrameClass::Data | FrameClass::Control => {
                let _ = self.normal.send(env).await;
            }
        }
    }
}

/// `Link`'s sender alias for a neighbour (PROCESSOR-DESIGN §2.2).
pub type EnvelopeSender = ProcessorTx;

/// Build a processor channel pair (the bounded normal + unbounded system halves)
/// tagged with the owning processor's `name`.
pub fn channel(name: std::sync::Arc<str>, normal_cap: usize) -> (ProcessorTx, ProcessorRx) {
    let (sys_tx, sys_rx) = mpsc::unbounded_channel();
    let (norm_tx, norm_rx) = mpsc::channel(normal_cap);
    (
        ProcessorTx {
            system: sys_tx,
            normal: norm_tx,
            name,
        },
        ProcessorRx {
            system: sys_rx,
            normal: norm_rx,
        },
    )
}

fn stop_reason(frame: &Frame) -> StopReason {
    match frame {
        Frame::End { .. } => StopReason::EndOfTask,
        Frame::Stop => StopReason::Stopped,
        Frame::Cancel { .. } => StopReason::Cancelled,
        _ => StopReason::Stopped,
    }
}

/// The framework-owned per-processor loop. Runs until the input channels close or
/// a terminal frame (`End`/`Cancel`) is forwarded.
///
/// PROCESSOR-DESIGN §2.2 pseudocode, made real:
/// - `biased` select drains the **system** channel first (priority);
/// - `Start` runs `start()` then forwards;
/// - `Interruption` drains the normal queue of *interruptible* frames (keeping
///   uninterruptible ones — End/Stop/FunctionCallResult/UpdateSettings) then
///   forwards;
/// - `End`/`Stop`/`Cancel` run `stop()`, forward, and (End/Cancel) break;
/// - any other frame goes to `process_frame`; an `Err` becomes an upstream
///   `Error{fatal:false}`.
pub async fn run_processor(
    mut p: Box<dyn FrameProcessor>,
    mut rx: ProcessorRx,
    link: Link,
    setup: ProcessorSetup,
) {
    loop {
        let env = tokio::select! {
            biased;
            Some(e) = rx.system.recv() => e,
            Some(e) = rx.normal.recv() => e,
            else => break,
        };

        // Observer `on_process` hook (zero-cost when no observer registered).
        if let Some(o) = &setup.observer {
            let ev = crate::observer::FrameEvent {
                processor: &link.name,
                frame: &env.frame,
                meta: &env.meta,
                direction: env.direction,
                timestamp_ns: setup.clock.now_ns(),
            };
            o.on_process(&ev).await;
        }

        match &env.frame {
            Frame::Start(params) => {
                let params = params.clone();
                if let Err(e) = p.start(&setup, &params).await {
                    link.push_error(e.to_string(), false).await;
                }
                link.push(env.meta, env.frame, env.direction).await;
            }
            Frame::Interruption => {
                // Drain the normal queue: drop interruptible frames, keep
                // uninterruptible ones. A kept *downstream terminal* (End/Stop) is
                // returned rather than blindly forwarded — see below.
                let kept_terminal = drain_on_interruption(&mut rx, &link).await;
                // Forward the interruption in the direction it arrived.
                link.push(env.meta, env.frame, env.direction).await;
                // A downstream End/Stop that was buffered when the interruption hit
                // must go through the normal terminal handling so a `Sink` taps it via
                // `stop()` (its `next` is `None`, so a raw `link.push` would silently
                // drop it and the task would hang waiting for a terminal that never
                // surfaces). This was a latent interruption-vs-End race.
                if let Some(term) = kept_terminal {
                    let reason = stop_reason(&term.frame);
                    let is_terminal = !matches!(&term.frame, Frame::Stop);
                    if let Err(e) = p.stop(reason).await {
                        link.push_error(e.to_string(), false).await;
                    }
                    link.push(term.meta, term.frame, term.direction).await;
                    if is_terminal {
                        break;
                    }
                }
            }
            // Framework lifecycle handling applies only to **downstream** lifecycle
            // frames (the actual drain signal). An *upstream* End/Stop/Cancel is a
            // "request to end" from an inner processor — it must fall through to
            // `process_frame` so the `Source` can convert it into a downstream
            // lifecycle frame (PROCESSOR-DESIGN §4.1; pipecat keeps these distinct as
            // EndTaskFrame vs EndFrame). Intercepting it here would strand the request
            // and the task would never terminate.
            Frame::Cancel { .. } | Frame::End { .. } | Frame::Stop
                if env.direction == Direction::Downstream =>
            {
                let reason = stop_reason(&env.frame);
                let terminal = !matches!(&env.frame, Frame::Stop);
                if let Err(e) = p.stop(reason).await {
                    link.push_error(e.to_string(), false).await;
                }
                link.push(env.meta, env.frame, env.direction).await;
                if terminal {
                    break;
                }
            }
            _ => {
                if let Err(e) = p.process_frame(env, &link).await {
                    link.push_error(e.to_string(), false).await;
                }
            }
        }
    }
}

/// On an interruption, drain the **normal** channel of currently-queued frames,
/// dropping interruptible ones and forwarding uninterruptible ones (End/Stop/
/// FunctionCallResult/UpdateSettings) so they are never lost (pipecat
/// `_start_interruption`, `frame_processor.py:828`).
///
/// Returns a kept **downstream terminal** (End/Stop) if one was buffered, so the
/// caller can run it through the normal terminal handling (`stop()` + break) — at a
/// `Sink` that hook is what taps the terminal for `PipelineTask`; forwarding it via
/// `link.push` here would drop it (Sink `next` is `None`) and hang the task.
///
/// Ordering note: non-terminal kept frames (UpdateSettings/FunctionCallResult, or
/// upstream terminals) are forwarded here, *before* the caller forwards the
/// `Interruption` marker. Since the marker is a `System` frame (unbounded,
/// biased-first) and those frames are Data/Control, the marker may overtake them at
/// the next processor. That is intentional and safe: they are exactly the frames
/// that *survive* an interruption by definition, so the next hop keeps them
/// regardless of arrival order.
async fn drain_on_interruption(rx: &mut ProcessorRx, link: &Link) -> Option<Envelope> {
    // `try_recv` only the frames already queued — do not await new ones.
    while let Ok(env) = rx.normal.try_recv() {
        if env.frame.uninterruptible() {
            // A downstream terminal goes back to the caller for normal terminal
            // handling (so a Sink taps it via `stop()`); stop draining (nothing
            // queued after a terminal matters — the task is ending).
            if env.direction == Direction::Downstream && env.frame.is_terminal() {
                return Some(env);
            }
            // Other uninterruptible frames forward in their direction.
            link.push(env.meta, env.frame, env.direction).await;
        }
        // else: drop the interruptible frame.
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result;
    use crate::processor::{Clock, FrameProcessor};
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    /// A tail/sink-like processor: `next` is `None`, and it records whether the
    /// framework ran its `stop()` hook (the hook a `Sink` uses to tap a terminal).
    struct TapSink {
        stopped: Arc<AtomicBool>,
    }
    #[async_trait]
    impl FrameProcessor for TapSink {
        fn name(&self) -> &str {
            "TapSink"
        }
        async fn stop(&mut self, _reason: StopReason) -> Result<()> {
            self.stopped.store(true, Ordering::Relaxed);
            Ok(())
        }
    }

    fn sink_link(name: Arc<str>, clock: Clock) -> Link {
        // A tail Link: no `next` (the condition that made the bug drop the End).
        Link {
            next: None,
            prev: None,
            name,
            clock,
            observer: None,
            enable_metrics: false,
            enable_usage_metrics: false,
            ttfb_start: Arc::new(std::sync::atomic::AtomicI64::new(0)),
            processing_start: Arc::new(std::sync::atomic::AtomicI64::new(0)),
        }
    }

    /// Regression: a downstream `End` buffered on the normal channel when an
    /// `Interruption` arrives at a `Sink` (no `next`) must still terminate the task
    /// via the `stop()` hook — not be silently dropped by the interruption drain
    /// (which would hang `PipelineTask::run`).
    #[tokio::test]
    async fn interruption_does_not_lose_a_buffered_downstream_end_at_a_sink() {
        let name: Arc<str> = Arc::from("TapSink");
        let (tx, rx) = channel(name.clone(), 8);
        let clock = Clock::new();
        let setup = ProcessorSetup {
            clock: clock.clone(),
            observer: None,
            cancel: tokio_util::sync::CancellationToken::new(),
            enable_metrics: false,
            enable_usage_metrics: false,
        };
        let stopped = Arc::new(AtomicBool::new(false));
        let p = Box::new(TapSink {
            stopped: stopped.clone(),
        });

        // The race: a downstream End is already buffered (Control → normal channel)
        // when the Interruption (System → system channel, biased-first) is handled.
        tx.send(Envelope::new(
            Frame::End { reason: None },
            Direction::Downstream,
        ))
        .await;
        tx.send(Envelope::new(Frame::Interruption, Direction::Downstream))
            .await;
        drop(tx); // close channels so the loop can't block awaiting more input

        let h = tokio::spawn(run_processor(p, rx, sink_link(name, clock), setup));
        let res = tokio::time::timeout(std::time::Duration::from_secs(2), h).await;
        assert!(
            res.is_ok(),
            "run_processor hung on the interruption-vs-buffered-End race"
        );
        assert!(
            stopped.load(Ordering::Relaxed),
            "the Sink's stop() hook must run for the buffered End (else PipelineTask hangs)"
        );
    }
}
