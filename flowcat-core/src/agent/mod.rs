// SPDX-License-Identifier: Apache-2.0
//
//! Higher-level **pure-logic** conversation features (no network).
//!
//! These are frame-logic processors that decide things from the existing frame
//! stream without any outbound network call — so they live in core (the
//! networked pieces, MCP-as-processor + the carrier-coordinated transfer, are
//! out: MCP → `flowcat-services`, transfer → the embedder).
//!
//! Module map:
//! - [`voicemail`] — voicemail detection over VAD/transcript frames.
//! - [`ivr`] — IVR-navigator state machine.
//! - [`wakeword`] — wake-word gating.
//! - [`text_filter`] — text aggregators / output text filters.
//! - [`fcall_filter`] — function-call filters.

pub mod fcall_filter;
pub mod ivr;
pub mod text_filter;
pub mod voicemail;
pub mod wakeword;

/// A tiny single-processor test harness shared by the agent unit tests. Drives a
/// processor through one capture hop without spinning a full `PipelineTask`, so
/// each feature's `process_frame` can be asserted deterministically (no network,
/// no provider keys).
#[cfg(test)]
pub(crate) mod test_harness {
    use std::sync::atomic::AtomicI64;
    use std::sync::Arc;
    use std::time::Duration;

    use crate::processor::frame::{Direction, Frame};
    use crate::processor::runtime::{channel, NORMAL_CHAN_CAP};
    use crate::processor::{Envelope, FrameProcessor, Link};

    /// Build a [`Link`] whose downstream/upstream neighbour is a single capture
    /// channel, run `proc` over `frames` in `direction`, and return every frame
    /// it pushed onto the capture channel (both directions funnel to the same
    /// sink so tests can assert upstream emissions too).
    pub(crate) async fn drive(
        mut proc: Box<dyn FrameProcessor>,
        frames: Vec<Frame>,
        direction: Direction,
    ) -> Vec<Frame> {
        // One capture channel acts as both the downstream and upstream neighbour.
        let (cap_tx, mut cap_rx) = channel(Arc::from("capture"), NORMAL_CHAN_CAP);
        let link = Link {
            next: Some(cap_tx.clone()),
            prev: Some(cap_tx),
            name: Arc::from(proc.name().to_string()),
            clock: crate::processor::Clock::new(),
            observer: None,
            enable_metrics: false,
            enable_usage_metrics: false,
            ttfb_start: Arc::new(AtomicI64::new(0)),
            processing_start: Arc::new(AtomicI64::new(0)),
        };

        for frame in frames {
            let env = Envelope::new(frame, direction);
            proc.process_frame(env, &link)
                .await
                .expect("process_frame errored");
        }

        // Drain whatever the processor emitted (system frames on the unbounded
        // half, data/control on the bounded half), with a short settle window.
        let mut out = Vec::new();
        loop {
            tokio::select! {
                biased;
                Some(e) = cap_rx.system.recv() => out.push(e.frame),
                Some(e) = cap_rx.normal.recv() => out.push(e.frame),
                _ = tokio::time::sleep(Duration::from_millis(20)) => break,
            }
        }
        out
    }
}
