// SPDX-License-Identifier: Apache-2.0
//
//! `PipelineTask` — one running pipeline's lifecycle (PROCESSOR-DESIGN §4.1).
//!
//! Owns the wrapped pipeline (Source + user + Sink), the push queue, the clock,
//! the observer fan-out, idle detection, heartbeat/watchdog, and start/end
//! signalling. Mirrors pipecat `PipelineTask` (`task.py:142`).
//!
//! **Lifecycle:** `run()` builds channels + spawns every processor task, injects
//! `Start` and blocks until it reaches the Sink (every processor ran `start()`),
//! pumps queued frames into the head, and exits when a terminal frame
//! (`End`/`Stop`/`Cancel`) reaches the Sink.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use crate::error::Result;
use crate::observer::{FrameObserver, IdleFrameObserver, Observer};
use crate::processor::frame::{Direction, Frame, FrameKind, StartParams};
use crate::processor::runtime::ProcessorTx;
use crate::processor::{into_topology, Clock, Envelope, ProcessorSetup, StopReason, Topology};

use super::{link_topology, Sink, Source};

/// Parameters for a [`PipelineTask`]. Mirrors pipecat (`task.py:59-62`).
///
/// **flowcat-core's default keeps heartbeats OFF** (pipecat parity). An embedder
/// can opt in later (PROCESSOR-DESIGN §10 Q3) — do not flip the core
/// default here.
#[derive(Debug, Clone)]
pub struct PipelineTaskParams {
    pub audio_in_sample_rate: u32,
    pub audio_out_sample_rate: u32,
    pub enable_metrics: bool,
    pub enable_usage_metrics: bool,
    pub enable_tracing: bool,
    pub enable_heartbeats: bool,
    pub heartbeat_period: Duration,
    pub heartbeat_monitor: Duration,
    pub idle_timeout: Option<Duration>,
    pub cancel_on_idle: bool,
    pub idle_timeout_frames: Vec<FrameKind>,
}

impl Default for PipelineTaskParams {
    fn default() -> Self {
        Self {
            audio_in_sample_rate: 16_000,
            audio_out_sample_rate: 24_000,
            enable_metrics: false,
            enable_usage_metrics: false,
            enable_tracing: false,
            // pipecat parity: OFF by default. (Do not flip — §10 Q3.)
            enable_heartbeats: false,
            heartbeat_period: Duration::from_secs(1),
            heartbeat_monitor: Duration::from_secs(10),
            idle_timeout: Some(Duration::from_secs(300)),
            cancel_on_idle: true,
            idle_timeout_frames: vec![FrameKind::BotSpeaking, FrameKind::UserSpeaking],
        }
    }
}

impl PipelineTaskParams {
    fn start_params(&self) -> StartParams {
        StartParams {
            audio_in_sample_rate: self.audio_in_sample_rate,
            audio_out_sample_rate: self.audio_out_sample_rate,
            enable_metrics: self.enable_metrics,
            enable_usage_metrics: self.enable_usage_metrics,
            enable_tracing: self.enable_tracing,
            report_only_initial_ttfb: true,
        }
    }
}

type CallbackUnit = Box<dyn Fn() + Send + Sync>;
type CallbackReason = Box<dyn Fn(StopReason) + Send + Sync>;
type CallbackError = Box<dyn Fn(&str, bool) + Send + Sync>;

/// One running pipeline's lifecycle handle.
pub struct PipelineTask {
    topo: Option<Topology>,
    params: PipelineTaskParams,
    observers: Vec<Arc<dyn FrameObserver>>,
    cancel: CancellationToken,
    finished: Arc<AtomicBool>,
    // Queue of frames to inject into the head once ready.
    queue_tx: mpsc::UnboundedSender<Frame>,
    queue_rx: Option<mpsc::UnboundedReceiver<Frame>>,
    // Event hooks.
    on_started: Vec<CallbackUnit>,
    on_finished: Vec<CallbackReason>,
    on_error: Vec<CallbackError>,
    on_idle_timeout: Vec<CallbackUnit>,
    // The idle observer is auto-registered so the watcher can read activity.
    idle_observer: Arc<IdleFrameObserver>,
}

impl PipelineTask {
    /// Build a task over `pipeline` with `params` and external `observers`.
    pub fn new(
        pipeline: super::Pipeline,
        params: PipelineTaskParams,
        observers: Vec<Arc<dyn FrameObserver>>,
    ) -> Self {
        let (queue_tx, queue_rx) = mpsc::unbounded_channel();
        let idle_observer = Arc::new(IdleFrameObserver::new(params.idle_timeout_frames.clone()));
        Self {
            topo: Some(into_topology(Box::new(pipeline))),
            params,
            observers,
            cancel: CancellationToken::new(),
            finished: Arc::new(AtomicBool::new(false)),
            queue_tx,
            queue_rx: Some(queue_rx),
            on_started: Vec::new(),
            on_finished: Vec::new(),
            on_error: Vec::new(),
            on_idle_timeout: Vec::new(),
            idle_observer,
        }
    }

    /// Queue a downstream frame into the head of the pipeline.
    pub async fn queue_frame(&self, frame: Frame) {
        let _ = self.queue_tx.send(frame);
    }

    /// Queue several downstream frames.
    pub async fn queue_frames(&self, frames: impl IntoIterator<Item = Frame>) {
        for f in frames {
            let _ = self.queue_tx.send(f);
        }
    }

    /// A clonable handle to the head-injection queue, for an external producer
    /// (e.g. the S2S transport pump) that feeds frames into the pipeline head from
    /// its own task while [`run`](Self::run) drives the pipeline. Sending after the
    /// task finished is a harmless no-op (the receiver is drained on shutdown).
    pub fn queue_sender(&self) -> mpsc::UnboundedSender<Frame> {
        self.queue_tx.clone()
    }

    /// Graceful: queue an `End` so the pipeline drains then shuts down.
    pub async fn stop_when_done(&self) {
        let _ = self.queue_tx.send(Frame::End { reason: None });
    }

    /// Immediate: queue a `Cancel`.
    pub async fn cancel(&self, reason: Option<String>) {
        let _ = self.queue_tx.send(Frame::Cancel { reason });
    }

    /// Whether the task has finished.
    pub fn has_finished(&self) -> bool {
        self.finished.load(Ordering::Relaxed)
    }

    /// The shared cancellation token (a [`PipelineRunner`](super::PipelineRunner)
    /// holds it to force-abort on signal).
    pub fn cancel_token(&self) -> CancellationToken {
        self.cancel.clone()
    }

    // ---- event hooks ----

    pub fn on_started(&mut self, f: impl Fn() + Send + Sync + 'static) {
        self.on_started.push(Box::new(f));
    }
    pub fn on_finished(&mut self, f: impl Fn(StopReason) + Send + Sync + 'static) {
        self.on_finished.push(Box::new(f));
    }
    pub fn on_error(&mut self, f: impl Fn(&str, bool) + Send + Sync + 'static) {
        self.on_error.push(Box::new(f));
    }
    pub fn on_idle_timeout(&mut self, f: impl Fn() + Send + Sync + 'static) {
        self.on_idle_timeout.push(Box::new(f));
    }

    /// Run to completion (PROCESSOR-DESIGN §4.1): spawn all processor tasks, inject
    /// `Start`, wait until it reaches the Sink (ready), pump queued frames, and
    /// exit when a terminal frame reaches the Sink (or the cancel token fires).
    pub async fn run(mut self) -> Result<()> {
        // Infallible: `run` consumes `self` (`mut self`), so it can be called at
        // most once; `topo`/`queue_rx` are `Some` from `new()` until this take.
        let topo = self.topo.take().expect("run called once");
        let mut queue_rx = self.queue_rx.take().expect("run called once");

        // Build the observer fan-out: the auto idle-observer + external ones.
        let mut all_obs: Vec<Arc<dyn FrameObserver>> =
            vec![self.idle_observer.clone() as Arc<dyn FrameObserver>];
        all_obs.extend(self.observers.iter().cloned());
        let observer = Observer::new(all_obs);

        let clock = Clock::new();
        let setup = ProcessorSetup {
            clock: clock.clone(),
            observer: Some(observer.clone()),
            cancel: self.cancel.clone(),
            enable_metrics: self.params.enable_metrics,
            enable_usage_metrics: self.params.enable_usage_metrics,
        };

        // Sink tap → the task observes downstream frames here.
        let (sink_tap_tx, mut sink_tap_rx) = mpsc::unbounded_channel::<Frame>();

        // Wrap: [Source, <topo>, Sink].
        let wrapped = Topology::Chain(vec![
            Topology::Leaf(Box::new(Source)),
            topo,
            Topology::Leaf(Box::new(Sink { tap: sink_tap_tx })),
        ]);

        let mut tasks = JoinSet::new();
        let linked = link_topology(wrapped, None, None, &setup, &mut tasks);
        let head: ProcessorTx = linked.head;

        observer.on_pipeline_started().await;

        // 1) Inject Start and block until it reaches the Sink (ready).
        head.send(Envelope::new(
            Frame::Start(self.params.start_params()),
            Direction::Downstream,
        ))
        .await;

        let ready = wait_for(
            &mut sink_tap_rx,
            |f| matches!(f, Frame::Start(_)),
            &self.cancel,
        )
        .await;
        if ready {
            for cb in &self.on_started {
                cb();
            }
        }

        // Auxiliary (non-processor) watcher tasks — heartbeat + idle — run until
        // the pump loop exits. They listen on this child token (cancelled the moment
        // the pipeline finishes, step 5) so the grace-join below only ever waits on
        // the real processor tasks; otherwise an idle watcher sleeping on its (up to
        // 300 s) timer would pin every teardown to the full 5 s grace window,
        // delaying finalize/write-back on *every* call. As a child of `self.cancel`,
        // it is also cancelled by an external force-cancel.
        let aux_cancel = self.cancel.child_token();

        // 2) Heartbeat task (OFF by default — pipecat parity).
        let last_heartbeat = Arc::new(AtomicU64::new(0));
        if self.params.enable_heartbeats {
            let hb_head = head.clone();
            let period = self.params.heartbeat_period;
            let cancel = aux_cancel.clone();
            let clk = clock.clone();
            tasks.spawn(async move {
                loop {
                    tokio::select! {
                        _ = cancel.cancelled() => break,
                        _ = tokio::time::sleep(period) => {
                            hb_head
                                .send(Envelope::new(
                                    Frame::Heartbeat { timestamp_ns: clk.now_ns() },
                                    Direction::Downstream,
                                ))
                                .await;
                        }
                    }
                }
            });
        }

        // 3) Idle watcher.
        let idle_fired = Arc::new(AtomicBool::new(false));
        if let Some(timeout) = self.params.idle_timeout {
            let idle_obs = self.idle_observer.clone();
            let cancel = aux_cancel.clone();
            let cancel_on_idle = self.params.cancel_on_idle;
            let fired = idle_fired.clone();
            let qtx = self.queue_tx.clone();
            tasks.spawn(async move {
                let mut last = idle_obs.activity_count();
                loop {
                    tokio::select! {
                        _ = cancel.cancelled() => break,
                        _ = tokio::time::sleep(timeout) => {
                            let now = idle_obs.activity_count();
                            if now == last {
                                fired.store(true, Ordering::Relaxed);
                                if cancel_on_idle {
                                    let _ = qtx.send(Frame::Cancel {
                                        reason: Some("idle timeout".into()),
                                    });
                                }
                                break;
                            }
                            last = now;
                        }
                    }
                }
            });
        }

        // 4) Pump queued frames into the head; watch the Sink for terminal/heartbeat.
        let mut stop_reason: Option<StopReason> = None;
        loop {
            tokio::select! {
                biased;
                _ = self.cancel.cancelled() => {
                    stop_reason = Some(StopReason::Cancelled);
                    // Drive a Cancel through so processors flush-stop.
                    head.send(Envelope::new(Frame::Cancel { reason: None }, Direction::Downstream)).await;
                    break;
                }
                maybe = queue_rx.recv() => {
                    match maybe {
                        Some(frame) => {
                            head.send(Envelope::new(frame, Direction::Downstream)).await;
                        }
                        None => {
                            // queue closed (all senders dropped) — keep draining sink.
                        }
                    }
                }
                maybe = sink_tap_rx.recv() => {
                    match maybe {
                        Some(Frame::Heartbeat { timestamp_ns }) => {
                            last_heartbeat.store(timestamp_ns as u64, Ordering::Relaxed);
                        }
                        Some(Frame::Error { fatal, message, .. }) => {
                            for cb in &self.on_error {
                                cb(&message, fatal);
                            }
                            if fatal {
                                head.send(Envelope::new(Frame::Cancel { reason: None }, Direction::Downstream)).await;
                            }
                        }
                        Some(f) if f.is_terminal() => {
                            stop_reason = Some(terminal_reason(&f));
                            break;
                        }
                        Some(_) => {}
                        None => break,
                    }
                }
            }
        }

        // 5) Stop the auxiliary watcher tasks immediately (the pipeline is done), so
        // the grace-join only waits on the real processor tasks draining.
        aux_cancel.cancel();
        // Drain remaining processor tasks (bounded by a grace window).
        let _ = tokio::time::timeout(Duration::from_secs(5), async {
            while tasks.join_next().await.is_some() {}
        })
        .await;
        // Anything still wedged gets force-cancelled.
        self.cancel.cancel();
        tasks.shutdown().await;

        if idle_fired.load(Ordering::Relaxed) {
            for cb in &self.on_idle_timeout {
                cb();
            }
        }

        let reason = stop_reason.unwrap_or(StopReason::EndOfTask);
        for cb in &self.on_finished {
            cb(reason);
        }
        self.finished.store(true, Ordering::Relaxed);
        Ok(())
    }
}

fn terminal_reason(f: &Frame) -> StopReason {
    match f {
        Frame::End { .. } => StopReason::EndOfTask,
        Frame::Stop => StopReason::Stopped,
        Frame::Cancel { .. } => StopReason::Cancelled,
        _ => StopReason::EndOfTask,
    }
}

/// Block until a frame matching `pred` is seen on the sink tap, or the task is
/// cancelled. Returns whether the predicate matched (false = cancelled/closed).
async fn wait_for(
    rx: &mut mpsc::UnboundedReceiver<Frame>,
    pred: impl Fn(&Frame) -> bool,
    cancel: &CancellationToken,
) -> bool {
    loop {
        tokio::select! {
            _ = cancel.cancelled() => return false,
            maybe = rx.recv() => match maybe {
                Some(f) if pred(&f) => return true,
                Some(_) => continue,
                None => return false,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result;
    use crate::processor::{FrameProcessor, Link};
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    fn fast_idle(params: PipelineTaskParams) -> PipelineTaskParams {
        params
    }

    // Records the order it saw Start vs data, and forwards.
    struct Probe {
        start_seen: Arc<AtomicBool>,
        data_before_start: Arc<AtomicUsize>,
    }
    #[async_trait]
    impl FrameProcessor for Probe {
        fn name(&self) -> &str {
            "Probe"
        }
        async fn start(&mut self, _s: &ProcessorSetup, _p: &StartParams) -> Result<()> {
            self.start_seen.store(true, Ordering::Relaxed);
            Ok(())
        }
        async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
            if matches!(env.frame, Frame::Text(_)) && !self.start_seen.load(Ordering::Relaxed) {
                self.data_before_start.fetch_add(1, Ordering::Relaxed);
            }
            link.push(env.meta, env.frame, env.direction).await;
            Ok(())
        }
    }

    // Emits an upstream End request when it sees a specific Text ("bye").
    struct EndRequester;
    #[async_trait]
    impl FrameProcessor for EndRequester {
        fn name(&self) -> &str {
            "EndRequester"
        }
        async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
            if matches!(&env.frame, Frame::Text(t) if t == "bye") {
                link.push_up(Frame::End {
                    reason: Some("requested".into()),
                })
                .await;
                return Ok(());
            }
            link.push(env.meta, env.frame, env.direction).await;
            Ok(())
        }
    }

    #[tokio::test]
    async fn start_reaches_sink_before_any_data_frame() {
        let start_seen = Arc::new(AtomicBool::new(false));
        let bad = Arc::new(AtomicUsize::new(0));
        let pipeline = super::super::Pipeline::new(vec![Box::new(Probe {
            start_seen: start_seen.clone(),
            data_before_start: bad.clone(),
        })]);
        let started = Arc::new(AtomicBool::new(false));
        let started2 = started.clone();
        let mut task = PipelineTask::new(pipeline, PipelineTaskParams::default(), vec![]);
        task.on_started(move || started2.store(true, Ordering::Relaxed));
        task.queue_frame(Frame::Text("data".into())).await;
        task.stop_when_done().await;
        task.run().await.unwrap();
        assert!(started.load(Ordering::Relaxed), "on_started must fire");
        assert_eq!(
            bad.load(Ordering::Relaxed),
            0,
            "data must not precede start"
        );
    }

    #[tokio::test]
    async fn stop_when_done_drains_then_ends() {
        let count = Arc::new(AtomicUsize::new(0));
        let count2 = count.clone();
        let pipeline = super::super::Pipeline::new(vec![Box::new(CountText {
            count: count.clone(),
        })]);
        let mut task = PipelineTask::new(pipeline, PipelineTaskParams::default(), vec![]);
        let finished_reason = Arc::new(Mutex::new(None));
        let fr = finished_reason.clone();
        task.on_finished(move |r| *fr.lock().unwrap() = Some(r));
        task.queue_frames(vec![
            Frame::Text("a".into()),
            Frame::Text("b".into()),
            Frame::Text("c".into()),
        ])
        .await;
        task.stop_when_done().await;
        task.run().await.unwrap();
        assert_eq!(
            count2.load(Ordering::Relaxed),
            3,
            "all data drained before End"
        );
        assert_eq!(
            *finished_reason.lock().unwrap(),
            Some(StopReason::EndOfTask)
        );
        let _ = count;
    }

    #[tokio::test]
    async fn cancel_ends_without_draining_everything() {
        let pipeline = super::super::Pipeline::new(vec![Box::new(CountText {
            count: Arc::new(AtomicUsize::new(0)),
        })]);
        let mut task = PipelineTask::new(pipeline, PipelineTaskParams::default(), vec![]);
        let reason = Arc::new(Mutex::new(None));
        let r2 = reason.clone();
        task.on_finished(move |r| *r2.lock().unwrap() = Some(r));
        task.cancel(None).await;
        task.run().await.unwrap();
        assert_eq!(*reason.lock().unwrap(), Some(StopReason::Cancelled));
    }

    #[tokio::test]
    async fn upstream_end_request_converts_to_downstream_end() {
        let pipeline = super::super::Pipeline::new(vec![Box::new(EndRequester)]);
        let mut task = PipelineTask::new(pipeline, PipelineTaskParams::default(), vec![]);
        let reason = Arc::new(Mutex::new(None));
        let r2 = reason.clone();
        task.on_finished(move |r| *r2.lock().unwrap() = Some(r));
        task.queue_frame(Frame::Text("bye".into())).await;
        // No explicit stop — the upstream End request must end the task.
        task.run().await.unwrap();
        assert_eq!(*reason.lock().unwrap(), Some(StopReason::EndOfTask));
    }

    #[tokio::test]
    async fn idle_timeout_fires_and_cancels() {
        let pipeline = super::super::Pipeline::new(vec![Box::new(CountText {
            count: Arc::new(AtomicUsize::new(0)),
        })]);
        let params = fast_idle(PipelineTaskParams {
            idle_timeout: Some(Duration::from_millis(80)),
            cancel_on_idle: true,
            ..Default::default()
        });
        let mut task = PipelineTask::new(pipeline, params, vec![]);
        let idled = Arc::new(AtomicBool::new(false));
        let i2 = idled.clone();
        task.on_idle_timeout(move || i2.store(true, Ordering::Relaxed));
        let reason = Arc::new(Mutex::new(None));
        let r2 = reason.clone();
        task.on_finished(move |r| *r2.lock().unwrap() = Some(r));
        // Never queue activity → goes idle.
        task.run().await.unwrap();
        assert!(idled.load(Ordering::Relaxed), "idle hook must fire");
        assert_eq!(*reason.lock().unwrap(), Some(StopReason::Cancelled));
    }

    struct CountText {
        count: Arc<AtomicUsize>,
    }
    #[async_trait]
    impl FrameProcessor for CountText {
        fn name(&self) -> &str {
            "CountText"
        }
        async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
            if matches!(env.frame, Frame::Text(_)) {
                self.count.fetch_add(1, Ordering::Relaxed);
            }
            link.push(env.meta, env.frame, env.direction).await;
            Ok(())
        }
    }
}
