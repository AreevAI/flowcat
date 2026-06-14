// SPDX-License-Identifier: Apache-2.0
//
//! Sentry exporter (behind `obs-sentry`).
//!
//! A [`FrameObserver`](flowcat_core::observer::FrameObserver) that reports
//! pipeline **errors** (and TTFB/processing **transactions**, mirroring pipecat's
//! `processors/metrics/sentry.py` `SentryMetrics`) to Sentry.
//!
//! ## Testability (no network/keys)
//!
//! The transport is the injectable [`SentrySink`] trait. The frame→event mapping
//! is unit-tested against an in-memory [`RecordingSentrySink`] — no Sentry DSN,
//! no HTTP. The live sink ([`HttpSentrySink`]) POSTs the same [`SentryEvent`] JSON
//! to Sentry's store endpoint with `reqwest` (rustls) and is infra-gated.

use std::sync::Mutex;

use async_trait::async_trait;
use serde::Serialize;
use serde_json::{Map, Value};

use flowcat_core::observer::{FrameObserver, FramePushEvent};
use flowcat_core::processor::frame::Direction;
use flowcat_core::processor::metrics::MetricsData;
use flowcat_core::Frame;

/// Sentry event severity (`level` field).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SentryLevel {
    Info,
    Warning,
    Error,
    Fatal,
}

/// One Sentry event ready to capture. A trimmed Sentry "event" payload: a level,
/// a message, an optional `transaction` (op/name, like pipecat's metric
/// transactions), plus `tags`/`extra` maps. Serializes to the JSON Sentry's store
/// endpoint accepts.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SentryEvent {
    pub level: SentryLevel,
    pub message: String,
    /// A perf transaction op (`"ttfb"`/`"processing"`) when this event is a
    /// metric transaction rather than an error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transaction: Option<String>,
    #[serde(skip_serializing_if = "Map::is_empty")]
    pub tags: Map<String, Value>,
    #[serde(skip_serializing_if = "Map::is_empty")]
    pub extra: Map<String, Value>,
}

impl SentryEvent {
    fn error(message: impl Into<String>, level: SentryLevel) -> Self {
        Self {
            level,
            message: message.into(),
            transaction: None,
            tags: Map::new(),
            extra: Map::new(),
        }
    }

    fn transaction(op: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            level: SentryLevel::Info,
            message: message.into(),
            transaction: Some(op.into()),
            tags: Map::new(),
            extra: Map::new(),
        }
    }

    fn tag(mut self, key: &str, value: impl Into<Value>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }

    fn with_extra(mut self, key: &str, value: impl Into<Value>) -> Self {
        self.extra.insert(key.into(), value.into());
        self
    }
}

/// Where the [`SentryExporter`] hands captured events. `capture` is `async` so
/// the live [`HttpSentrySink`] can `.await` its `reqwest` POST directly from the
/// pipeline's `on_push` hook — no `tokio::spawn`, no extra runtime dep. Tests use
/// [`RecordingSentrySink`] (a trivial in-memory impl).
#[async_trait]
pub trait SentrySink: Send + Sync {
    /// Capture (report) one event.
    async fn capture(&self, event: SentryEvent);
}

#[async_trait]
impl<S: SentrySink + ?Sized> SentrySink for std::sync::Arc<S> {
    async fn capture(&self, event: SentryEvent) {
        (**self).capture(event).await
    }
}

/// In-memory [`SentrySink`] for tests — records every captured event.
#[derive(Debug, Default)]
pub struct RecordingSentrySink {
    events: Mutex<Vec<SentryEvent>>,
}

impl RecordingSentrySink {
    pub fn new() -> Self {
        Self::default()
    }

    /// Captured events, in order.
    pub fn events(&self) -> Vec<SentryEvent> {
        self.events.lock().map(|g| g.clone()).unwrap_or_default()
    }
}

#[async_trait]
impl SentrySink for RecordingSentrySink {
    async fn capture(&self, event: SentryEvent) {
        if let Ok(mut g) = self.events.lock() {
            g.push(event);
        }
    }
}

/// Live sink: POSTs each [`SentryEvent`] to Sentry's store endpoint. Holds a
/// `reqwest` client + the store URL and `X-Sentry-Auth` header derived from a
/// DSN. The send is `.await`ed (best-effort — a failed POST is logged, never
/// propagated, so observability can't break the call). Infra-gated (needs a real
/// DSN); the *mapping* it serializes is the tested surface.
pub struct HttpSentrySink {
    client: reqwest::Client,
    store_url: String,
    auth_header: String,
}

impl HttpSentrySink {
    /// Build from a pre-parsed store URL + `X-Sentry-Auth` header value. (DSN
    /// parsing is the host's job; this keeps the sink free of DSN formats.)
    pub fn new(
        client: reqwest::Client,
        store_url: impl Into<String>,
        auth_header: impl Into<String>,
    ) -> Self {
        Self {
            client,
            store_url: store_url.into(),
            auth_header: auth_header.into(),
        }
    }
}

#[async_trait]
impl SentrySink for HttpSentrySink {
    async fn capture(&self, event: SentryEvent) {
        if let Err(e) = self
            .client
            .post(&self.store_url)
            .header("X-Sentry-Auth", &self.auth_header)
            .json(&event)
            .send()
            .await
        {
            tracing::warn!(error = %e, "sentry capture failed");
        }
    }
}

/// Reports pipeline errors + metric transactions to Sentry via an injected
/// [`SentrySink`]. Implements [`FrameObserver`].
pub struct SentryExporter<S: SentrySink> {
    sink: S,
    seen: Mutex<Vec<u64>>,
}

impl<S: SentrySink> SentryExporter<S> {
    /// Build an exporter that captures to `sink`.
    pub fn new(sink: S) -> Self {
        Self {
            sink,
            seen: Mutex::new(Vec::new()),
        }
    }

    /// Map one metrics batch into Sentry transactions (pure; for direct testing).
    /// Only TTFB/processing become transactions (pipecat's `SentryMetrics`).
    pub async fn export_metrics(&self, batch: &[MetricsData]) {
        for d in batch {
            match d {
                MetricsData::Ttfb {
                    processor, seconds, ..
                } => {
                    self.sink
                        .capture(
                            SentryEvent::transaction("ttfb", format!("TTFB for {processor}"))
                                .tag("processor", processor.clone())
                                .with_extra("seconds", *seconds),
                        )
                        .await
                }
                MetricsData::Processing {
                    processor, seconds, ..
                } => {
                    self.sink
                        .capture(
                            SentryEvent::transaction(
                                "processing",
                                format!("Processing for {processor}"),
                            )
                            .tag("processor", processor.clone())
                            .with_extra("seconds", *seconds),
                        )
                        .await
                }
                // Usage/turn metrics are not Sentry transactions in pipecat.
                _ => {}
            }
        }
    }
}

#[async_trait]
impl<S: SentrySink> FrameObserver for SentryExporter<S> {
    async fn on_push(&self, e: &FramePushEvent<'_>) {
        if e.meta.broadcast_sibling_id.is_some() && e.direction != Direction::Downstream {
            return;
        }
        {
            let Ok(mut seen) = self.seen.lock() else {
                return;
            };
            if seen.contains(&e.meta.id) {
                return;
            }
            seen.push(e.meta.id);
        }
        match e.frame {
            Frame::Error {
                message,
                fatal,
                processor,
            } => {
                let level = if *fatal {
                    SentryLevel::Fatal
                } else {
                    SentryLevel::Error
                };
                let mut ev = SentryEvent::error(message.clone(), level);
                if let Some(p) = processor {
                    ev = ev.tag("processor", p.to_string());
                }
                self.sink.capture(ev).await;
            }
            Frame::Metrics(batch) => self.export_metrics(batch).await,
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flowcat_core::processor::frame::FrameMeta;
    use std::sync::Arc;

    async fn push(exp: &SentryExporter<Arc<RecordingSentrySink>>, frame: Frame) {
        let meta = FrameMeta::new(&frame);
        exp.on_push(&FramePushEvent {
            source: "src",
            destination: "dst",
            frame: &frame,
            meta: &meta,
            direction: Direction::Downstream,
            timestamp_ns: 0,
        })
        .await;
    }

    #[tokio::test]
    async fn fatal_error_maps_to_fatal_event() {
        let sink = Arc::new(RecordingSentrySink::new());
        let exp = SentryExporter::new(sink.clone());
        push(
            &exp,
            Frame::Error {
                message: "kaboom".into(),
                fatal: true,
                processor: Some(Arc::from("tts")),
            },
        )
        .await;
        let ev = &sink.events()[0];
        assert_eq!(ev.level, SentryLevel::Fatal);
        assert_eq!(ev.message, "kaboom");
        assert_eq!(ev.tags.get("processor").unwrap(), "tts");
        assert!(ev.transaction.is_none());
    }

    #[tokio::test]
    async fn non_fatal_error_maps_to_error_level() {
        let sink = Arc::new(RecordingSentrySink::new());
        let exp = SentryExporter::new(sink.clone());
        push(
            &exp,
            Frame::Error {
                message: "transient".into(),
                fatal: false,
                processor: None,
            },
        )
        .await;
        assert_eq!(sink.events()[0].level, SentryLevel::Error);
    }

    #[tokio::test]
    async fn ttfb_and_processing_metrics_map_to_transactions() {
        let sink = Arc::new(RecordingSentrySink::new());
        let exp = SentryExporter::new(sink.clone());
        push(
            &exp,
            Frame::Metrics(vec![
                MetricsData::Ttfb {
                    processor: "llm".into(),
                    model: None,
                    seconds: 0.3,
                },
                MetricsData::Processing {
                    processor: "tts".into(),
                    model: None,
                    seconds: 0.6,
                },
                // Usage is not a Sentry transaction → dropped.
                MetricsData::TtsUsage {
                    processor: "tts".into(),
                    characters: 5,
                },
            ]),
        )
        .await;
        let evs = sink.events();
        assert_eq!(evs.len(), 2);
        assert_eq!(evs[0].transaction.as_deref(), Some("ttfb"));
        assert_eq!(evs[0].message, "TTFB for llm");
        assert_eq!(evs[0].extra.get("seconds").unwrap(), 0.3);
        assert_eq!(evs[1].transaction.as_deref(), Some("processing"));
        assert_eq!(evs[1].tags.get("processor").unwrap(), "tts");
    }

    #[tokio::test]
    async fn dedups_and_serializes_to_sentry_json() {
        let sink = Arc::new(RecordingSentrySink::new());
        let exp = SentryExporter::new(sink.clone());
        let frame = Frame::Error {
            message: "once".into(),
            fatal: false,
            processor: None,
        };
        let meta = FrameMeta::new(&frame);
        for dst in ["a", "b"] {
            exp.on_push(&FramePushEvent {
                source: "src",
                destination: dst,
                frame: &frame,
                meta: &meta,
                direction: Direction::Downstream,
                timestamp_ns: 0,
            })
            .await;
        }
        assert_eq!(sink.events().len(), 1);
        // Serialized event uses lowercase level + omits empty maps/transaction.
        let v = serde_json::to_value(&sink.events()[0]).unwrap();
        assert_eq!(v["level"], "error");
        assert_eq!(v["message"], "once");
        assert!(v.get("transaction").is_none());
        assert!(v.get("tags").is_none());
    }
}
