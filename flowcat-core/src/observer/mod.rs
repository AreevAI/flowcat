// SPDX-License-Identifier: Apache-2.0
//
//! Non-intrusive pipeline monitoring (PROCESSOR-DESIGN §5.1).
//!
//! A [`FrameObserver`] sees every processed/pushed frame without sitting in the
//! chain — the seam OpenTelemetry/Sentry/Langfuse/RTVI observers plug into. The
//! [`Observer`] fan-out is cheaply clonable; hooks are invoked synchronously on
//! the hot path **only when at least one observer is registered** (the run loop
//! skips the call entirely otherwise — zero-cost-when-off).
//!
//! Mirrors pipecat `BaseObserver` (`observers/base_observer.py:70`).
//!
//! The [`rtvi`] sibling is a pure frame→event RTVI-protocol observer; the
//! **network exporters** (OpenTelemetry/Sentry/Langfuse) live in
//! `flowcat-services` so core stays dependency-light.

pub mod rtvi;

pub use rtvi::{
    FunctionCallReportLevel, RtviMessage, RtviObserver, RtviObserverParams, RtviSink, VecSink,
    MESSAGE_LABEL as RTVI_MESSAGE_LABEL, PROTOCOL_VERSION as RTVI_PROTOCOL_VERSION,
};

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::processor::frame::{Direction, Frame, FrameKind, FrameMeta};
use crate::processor::metrics::MetricsData;

/// A processor is about to handle a frame.
pub struct FrameEvent<'a> {
    pub processor: &'a str,
    pub frame: &'a Frame,
    pub meta: &'a FrameMeta,
    pub direction: Direction,
    /// Pipeline-clock timestamp (ns).
    pub timestamp_ns: i64,
}

/// A frame was pushed source→destination.
pub struct FramePushEvent<'a> {
    pub source: &'a str,
    pub destination: &'a str,
    pub frame: &'a Frame,
    pub meta: &'a FrameMeta,
    pub direction: Direction,
    pub timestamp_ns: i64,
}

/// Non-intrusive observer. Mirrors pipecat `BaseObserver` (`base_observer.py`).
#[async_trait]
pub trait FrameObserver: Send + Sync {
    /// A processor is about to handle a frame (`base_observer.py:79`).
    async fn on_process(&self, _e: &FrameEvent<'_>) {}
    /// A frame was pushed source→destination (`base_observer.py:91`).
    async fn on_push(&self, _e: &FramePushEvent<'_>) {}
    /// The pipeline finished starting (`base_observer.py:103`).
    async fn on_pipeline_started(&self) {}
}

/// Cheap clonable fan-out over many observers (pipecat `TaskObserver` proxy,
/// `task.py:401`). When empty, the run loop never calls into it.
#[derive(Clone, Default)]
pub struct Observer(Arc<[Arc<dyn FrameObserver>]>);

impl Observer {
    /// Build a fan-out over `observers`.
    pub fn new(observers: Vec<Arc<dyn FrameObserver>>) -> Self {
        Observer(observers.into())
    }

    /// Whether any observer is registered (the run loop checks this to stay
    /// zero-cost when off — though it also `Option`-gates the whole `Observer`).
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Fan `on_process` out to every observer.
    pub async fn on_process(&self, e: &FrameEvent<'_>) {
        for o in self.0.iter() {
            o.on_process(e).await;
        }
    }

    /// Fan `on_push` out to every observer.
    pub async fn on_push(&self, e: &FramePushEvent<'_>) {
        for o in self.0.iter() {
            o.on_push(e).await;
        }
    }

    /// Fan `on_pipeline_started` out to every observer.
    pub async fn on_pipeline_started(&self) {
        for o in self.0.iter() {
            o.on_pipeline_started().await;
        }
    }
}

// ---------------------------------------------------------------------------
// Built-in observers (PROCESSOR-DESIGN §5.1). Network exporters live in flowcat-services.
// ---------------------------------------------------------------------------

/// A turn boundary: who spoke and when (ns on the pipeline clock). Mirrors
/// pipecat's `TurnTrackingObserver` output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnEdge {
    UserStarted,
    UserStopped,
    BotStarted,
    BotStopped,
}

/// Tracks user/bot speaking edges → turn boundaries (pipecat
/// `TurnTrackingObserver`). Records each boundary with its timestamp so tests
/// and exporters can read the turn sequence.
#[derive(Default)]
pub struct TurnTrackingObserver {
    edges: Mutex<Vec<(TurnEdge, i64)>>,
}

impl TurnTrackingObserver {
    pub fn new() -> Self {
        Self::default()
    }

    /// The recorded turn boundaries (edge, timestamp_ns), in observation order.
    pub fn edges(&self) -> Vec<(TurnEdge, i64)> {
        self.edges.lock().map(|g| g.clone()).unwrap_or_default()
    }
}

#[async_trait]
impl FrameObserver for TurnTrackingObserver {
    async fn on_process(&self, e: &FrameEvent<'_>) {
        let edge = match e.frame.kind() {
            FrameKind::UserStartedSpeaking => Some(TurnEdge::UserStarted),
            FrameKind::UserStoppedSpeaking => Some(TurnEdge::UserStopped),
            FrameKind::BotStartedSpeaking => Some(TurnEdge::BotStarted),
            FrameKind::BotStoppedSpeaking => Some(TurnEdge::BotStopped),
            _ => None,
        };
        if let Some(edge) = edge {
            if let Ok(mut g) = self.edges.lock() {
                g.push((edge, e.timestamp_ns));
            }
        }
    }
}

/// Measures the user-stop → first-bot-audio latency (TTFB of the bot's reply).
/// Mirrors pipecat `UserBotLatencyObserver`.
#[derive(Default)]
pub struct UserBotLatencyObserver {
    /// ns at which the user last stopped speaking; -1 = none pending.
    user_stop_ns: AtomicI64,
    /// last measured latency in ns; -1 = none yet.
    last_latency_ns: AtomicI64,
}

impl UserBotLatencyObserver {
    pub fn new() -> Self {
        Self {
            user_stop_ns: AtomicI64::new(-1),
            last_latency_ns: AtomicI64::new(-1),
        }
    }

    /// The most recently measured user-stop → bot-audio latency in ns, if any.
    pub fn last_latency_ns(&self) -> Option<i64> {
        let v = self.last_latency_ns.load(Ordering::Relaxed);
        (v >= 0).then_some(v)
    }
}

#[async_trait]
impl FrameObserver for UserBotLatencyObserver {
    async fn on_process(&self, e: &FrameEvent<'_>) {
        match e.frame.kind() {
            FrameKind::UserStoppedSpeaking => {
                self.user_stop_ns.store(e.timestamp_ns, Ordering::Relaxed);
            }
            FrameKind::BotStartedSpeaking | FrameKind::TtsAudio | FrameKind::OutputAudio => {
                let start = self.user_stop_ns.swap(-1, Ordering::Relaxed);
                if start >= 0 {
                    let lat = (e.timestamp_ns - start).max(0);
                    self.last_latency_ns.store(lat, Ordering::Relaxed);
                }
            }
            _ => {}
        }
    }
}

/// Drives idle detection (PROCESSOR-DESIGN §4.1 / pipecat `task.py:70`): bumps a
/// monotonic counter every time one of the configured "activity" frame kinds is
/// observed. The [`PipelineTask`](crate::pipeline::PipelineTask) idle watcher
/// reads the counter and fires a timeout when it stops advancing.
pub struct IdleFrameObserver {
    activity: Vec<FrameKind>,
    counter: AtomicU64,
}

impl IdleFrameObserver {
    /// Observe `activity` frame kinds as "not idle" signals.
    pub fn new(activity: Vec<FrameKind>) -> Self {
        Self {
            activity,
            counter: AtomicU64::new(0),
        }
    }

    /// The current activity counter (advances on each observed activity frame).
    pub fn activity_count(&self) -> u64 {
        self.counter.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl FrameObserver for IdleFrameObserver {
    async fn on_process(&self, e: &FrameEvent<'_>) {
        if self.activity.contains(&e.frame.kind()) {
            self.counter.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// One observed metric, tagged with the pipeline-clock timestamp it was seen at.
/// The collected record the [`MetricsLogObserver`] keeps for inspection/export.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservedMetric {
    /// The metric payload (TTFB / processing / usage / turn).
    pub data: MetricsData,
    /// Pipeline-clock timestamp (ns) the [`Frame::Metrics`] was observed at.
    pub timestamp_ns: i64,
}

/// Collects (and `tracing`-logs) every [`MetricsData`] flowing on
/// [`Frame::Metrics`] — a pure, network-free metrics observer. Mirrors pipecat's
/// `MetricsLogObserver`: it watches the push hook, de-dupes by frame id, and
/// records each metric with its timestamp so tests (and metric exporters) can
/// read TTFB/RTF/processing/usage semantics without a live sink.
///
/// "RTF" (real-time factor) is derived, not a first-class metric: it is
/// `processing_seconds / audio_seconds` for a stage that reports both — see
/// [`MetricsLogObserver::real_time_factor`].
#[derive(Default)]
pub struct MetricsLogObserver {
    seen: Mutex<MetricsLogState>,
}

#[derive(Default)]
struct MetricsLogState {
    frames_seen: Vec<u64>,
    metrics: Vec<ObservedMetric>,
}

impl MetricsLogObserver {
    /// A fresh collector.
    pub fn new() -> Self {
        Self::default()
    }

    /// Every metric observed so far, in observation order.
    pub fn metrics(&self) -> Vec<ObservedMetric> {
        self.seen
            .lock()
            .map(|g| g.metrics.clone())
            .unwrap_or_default()
    }

    /// The most-recent TTFB (seconds) reported by `processor`, if any. TTFB is
    /// the user-stop → first-output latency a service measures via
    /// [`Link::start_ttfb`](crate::processor::Link::start_ttfb) /
    /// [`Link::stop_ttfb`](crate::processor::Link::stop_ttfb).
    pub fn ttfb_seconds(&self, processor: &str) -> Option<f64> {
        self.last_seconds(processor, |d| match d {
            MetricsData::Ttfb {
                processor: p,
                seconds,
                ..
            } if p == processor => Some(*seconds),
            _ => None,
        })
    }

    /// The most-recent processing time (seconds) reported by `processor`.
    pub fn processing_seconds(&self, processor: &str) -> Option<f64> {
        self.last_seconds(processor, |d| match d {
            MetricsData::Processing {
                processor: p,
                seconds,
                ..
            } if p == processor => Some(*seconds),
            _ => None,
        })
    }

    /// Real-time factor for `processor`: its last processing-time divided by
    /// `audio_seconds` of audio it processed. RTF < 1 ⇒ faster than real time.
    /// Returns `None` if no processing metric was seen or `audio_seconds == 0`.
    pub fn real_time_factor(&self, processor: &str, audio_seconds: f64) -> Option<f64> {
        if audio_seconds <= 0.0 {
            return None;
        }
        self.processing_seconds(processor)
            .map(|p| p / audio_seconds)
    }

    /// Total LLM tokens summed across every `LlmUsage` metric seen.
    pub fn total_llm_tokens(&self) -> u64 {
        self.metrics()
            .iter()
            .filter_map(|m| match &m.data {
                MetricsData::LlmUsage { tokens, .. } => Some(tokens.total_tokens),
                _ => None,
            })
            .sum()
    }

    fn last_seconds(
        &self,
        _processor: &str,
        pick: impl Fn(&MetricsData) -> Option<f64>,
    ) -> Option<f64> {
        self.metrics().iter().rev().find_map(|m| pick(&m.data))
    }
}

#[async_trait]
impl FrameObserver for MetricsLogObserver {
    async fn on_push(&self, e: &FramePushEvent<'_>) {
        let Frame::Metrics(batch) = e.frame else {
            return;
        };
        // Broadcast frames: only the downstream copy (avoid double counting).
        if e.meta.broadcast_sibling_id.is_some() && e.direction != Direction::Downstream {
            return;
        }
        let Ok(mut st) = self.seen.lock() else {
            return;
        };
        if st.frames_seen.contains(&e.meta.id) {
            return;
        }
        st.frames_seen.push(e.meta.id);
        for d in batch {
            tracing::debug!(processor = ?metric_processor(d), ?d, "metric");
            st.metrics.push(ObservedMetric {
                data: d.clone(),
                timestamp_ns: e.timestamp_ns,
            });
        }
    }
}

/// The processor name a metric is attributed to (for logging/filtering).
fn metric_processor(d: &MetricsData) -> &str {
    match d {
        MetricsData::Ttfb { processor, .. }
        | MetricsData::Processing { processor, .. }
        | MetricsData::LlmUsage { processor, .. }
        | MetricsData::TtsUsage { processor, .. }
        | MetricsData::TurnPrediction { processor, .. } => processor,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processor::frame::FrameMeta;

    fn meta_for(f: &Frame) -> FrameMeta {
        FrameMeta::new(f)
    }

    #[tokio::test]
    async fn turn_tracking_emits_boundaries_off_scripted_edges() {
        let obs = TurnTrackingObserver::new();
        let script = [
            (Frame::UserStartedSpeaking, 100),
            (Frame::UserStoppedSpeaking, 200),
            (Frame::BotStartedSpeaking, 300),
            (Frame::BotStoppedSpeaking, 400),
            (Frame::Text("noise".into()), 450), // ignored
        ];
        for (frame, ts) in script {
            let m = meta_for(&frame);
            let ev = FrameEvent {
                processor: "p",
                frame: &frame,
                meta: &m,
                direction: Direction::Downstream,
                timestamp_ns: ts,
            };
            obs.on_process(&ev).await;
        }
        assert_eq!(
            obs.edges(),
            vec![
                (TurnEdge::UserStarted, 100),
                (TurnEdge::UserStopped, 200),
                (TurnEdge::BotStarted, 300),
                (TurnEdge::BotStopped, 400),
            ]
        );
    }

    #[tokio::test]
    async fn user_bot_latency_measures_stop_to_bot_audio() {
        let obs = UserBotLatencyObserver::new();
        let stop = Frame::UserStoppedSpeaking;
        let bot = Frame::BotStartedSpeaking;
        let m1 = meta_for(&stop);
        obs.on_process(&FrameEvent {
            processor: "p",
            frame: &stop,
            meta: &m1,
            direction: Direction::Downstream,
            timestamp_ns: 1_000,
        })
        .await;
        let m2 = meta_for(&bot);
        obs.on_process(&FrameEvent {
            processor: "p",
            frame: &bot,
            meta: &m2,
            direction: Direction::Downstream,
            timestamp_ns: 1_500,
        })
        .await;
        assert_eq!(obs.last_latency_ns(), Some(500));
    }

    #[tokio::test]
    async fn idle_observer_counts_only_activity_frames() {
        let obs = IdleFrameObserver::new(vec![FrameKind::BotSpeaking, FrameKind::UserSpeaking]);
        let frames = [
            Frame::BotSpeaking,
            Frame::Text("x".into()), // not activity
            Frame::UserSpeaking,
        ];
        for f in frames {
            let m = meta_for(&f);
            obs.on_process(&FrameEvent {
                processor: "p",
                frame: &f,
                meta: &m,
                direction: Direction::Downstream,
                timestamp_ns: 0,
            })
            .await;
        }
        assert_eq!(obs.activity_count(), 2);
    }

    #[tokio::test]
    async fn empty_observer_fanout_is_a_noop() {
        let obs = Observer::default();
        assert!(obs.is_empty());
        // No panic, nothing to assert — just that calling through is harmless.
        let f = Frame::Text("x".into());
        let m = meta_for(&f);
        obs.on_process(&FrameEvent {
            processor: "p",
            frame: &f,
            meta: &m,
            direction: Direction::Downstream,
            timestamp_ns: 0,
        })
        .await;
        obs.on_pipeline_started().await;
    }

    // --- MetricsLogObserver -------------------------------------------------

    use crate::processor::metrics::LlmTokenUsage;

    async fn push_metrics(obs: &MetricsLogObserver, batch: Vec<MetricsData>, ts: i64) {
        let f = Frame::Metrics(batch);
        let m = meta_for(&f);
        obs.on_push(&FramePushEvent {
            source: "src",
            destination: "dst",
            frame: &f,
            meta: &m,
            direction: Direction::Downstream,
            timestamp_ns: ts,
        })
        .await;
    }

    #[tokio::test]
    async fn metrics_observer_records_ttfb_processing_and_rtf() {
        let obs = MetricsLogObserver::new();
        push_metrics(
            &obs,
            vec![
                MetricsData::Ttfb {
                    processor: "tts".into(),
                    model: None,
                    seconds: 0.25,
                },
                MetricsData::Processing {
                    processor: "tts".into(),
                    model: None,
                    seconds: 0.5,
                },
            ],
            1_000,
        )
        .await;
        assert_eq!(obs.ttfb_seconds("tts"), Some(0.25));
        assert_eq!(obs.processing_seconds("tts"), Some(0.5));
        // RTF = processing / audio = 0.5 / 2.0 = 0.25 (faster than real time).
        assert_eq!(obs.real_time_factor("tts", 2.0), Some(0.25));
        // No metric for an unknown processor / zero audio.
        assert_eq!(obs.ttfb_seconds("stt"), None);
        assert_eq!(obs.real_time_factor("tts", 0.0), None);
        // Both metrics recorded with their timestamp.
        let recorded = obs.metrics();
        assert_eq!(recorded.len(), 2);
        assert!(recorded.iter().all(|m| m.timestamp_ns == 1_000));
    }

    #[tokio::test]
    async fn metrics_observer_sums_llm_tokens_and_dedups() {
        let obs = MetricsLogObserver::new();
        let f = Frame::Metrics(vec![MetricsData::LlmUsage {
            processor: "llm".into(),
            model: Some("gpt".into()),
            tokens: LlmTokenUsage {
                total_tokens: 30,
                ..Default::default()
            },
        }]);
        let m = meta_for(&f);
        // Same frame id seen twice (two links) → counted once.
        for dst in ["a", "b"] {
            obs.on_push(&FramePushEvent {
                source: "llm",
                destination: dst,
                frame: &f,
                meta: &m,
                direction: Direction::Downstream,
                timestamp_ns: 0,
            })
            .await;
        }
        assert_eq!(obs.total_llm_tokens(), 30);
        assert_eq!(obs.metrics().len(), 1);
    }

    #[tokio::test]
    async fn metrics_observer_takes_last_ttfb_for_a_processor() {
        let obs = MetricsLogObserver::new();
        push_metrics(
            &obs,
            vec![MetricsData::Ttfb {
                processor: "llm".into(),
                model: None,
                seconds: 0.4,
            }],
            0,
        )
        .await;
        push_metrics(
            &obs,
            vec![MetricsData::Ttfb {
                processor: "llm".into(),
                model: None,
                seconds: 0.1,
            }],
            10,
        )
        .await;
        // Most-recent wins (report_only_initial_ttfb is the producer's concern).
        assert_eq!(obs.ttfb_seconds("llm"), Some(0.1));
    }
}
