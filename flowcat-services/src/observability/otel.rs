// SPDX-License-Identifier: Apache-2.0
//
//! OpenTelemetry exporter (behind `obs-otel`).
//!
//! A [`FrameObserver`](flowcat_core::observer::FrameObserver) that turns the
//! pipeline's [`Frame::Metrics`](flowcat_core::Frame::Metrics) (and error) stream
//! into OpenTelemetry spans, following pipecat's tracing semantic conventions
//! (`pipecat/utils/tracing/service_attributes.py`): `gen_ai.provider.name`,
//! `gen_ai.request.model`, `metrics.ttfb`, `metrics.character_count`, plus
//! token-usage counters.
//!
//! ## Testability (no network/keys)
//!
//! The export **sink is injectable** via the [`OtelSink`] trait, so the
//! frame→span mapping is unit-tested against an in-memory [`RecordingOtelSink`]
//! with no OTLP endpoint or `global` tracer. A live exporter that pushes into a
//! real OTel `Tracer` is the infra-gated production sink (sketched in
//! [`TracerOtelSink`]); the mapping it feeds is identical and fully covered.

use std::sync::Mutex;

use async_trait::async_trait;
use opentelemetry::KeyValue;

use flowcat_core::observer::{FrameObserver, FramePushEvent};
use flowcat_core::processor::frame::Direction;
use flowcat_core::processor::metrics::MetricsData;
use flowcat_core::Frame;

/// One span ready to export: a name + OTel-typed attributes. Mirrors the shape a
/// pipecat tracing decorator builds before `span.set_attribute(...)`.
#[derive(Debug, Clone)]
pub struct OtelSpan {
    /// Span name, e.g. `"metrics.ttfb"`, `"gen_ai.usage"`, `"pipeline.error"`.
    pub name: String,
    /// OTel attributes (`gen_ai.*`, `metrics.*`, …).
    pub attributes: Vec<KeyValue>,
}

impl OtelSpan {
    fn new(name: impl Into<String>, attributes: Vec<KeyValue>) -> Self {
        Self {
            name: name.into(),
            attributes,
        }
    }

    /// Look up an attribute value by key as a string (test helper).
    pub fn attr(&self, key: &str) -> Option<String> {
        self.attributes
            .iter()
            .find(|kv| kv.key.as_str() == key)
            .map(|kv| kv.value.to_string())
    }
}

/// Where the [`OtelExporter`] hands finished spans. The production impl pushes
/// into a real OTel tracer; tests use [`RecordingOtelSink`]. Keeping this a trait
/// is what makes the exporter unit-testable without an OTLP endpoint.
pub trait OtelSink: Send + Sync {
    /// Export one span.
    fn export(&self, span: OtelSpan);
}

/// `Arc<S>` is a sink if `S` is — lets the exporter own an `Arc` while a test (or
/// the host) keeps a shared handle to inspect/flush the same sink.
impl<S: OtelSink + ?Sized> OtelSink for std::sync::Arc<S> {
    fn export(&self, span: OtelSpan) {
        (**self).export(span)
    }
}

/// In-memory [`OtelSink`] that records every span — the test sink.
#[derive(Debug, Default)]
pub struct RecordingOtelSink {
    spans: Mutex<Vec<OtelSpan>>,
}

impl RecordingOtelSink {
    /// A fresh recording sink.
    pub fn new() -> Self {
        Self::default()
    }

    /// Spans recorded so far, in export order.
    pub fn spans(&self) -> Vec<OtelSpan> {
        self.spans.lock().map(|g| g.clone()).unwrap_or_default()
    }

    /// Recorded span names, in order.
    pub fn names(&self) -> Vec<String> {
        self.spans
            .lock()
            .map(|g| g.iter().map(|s| s.name.clone()).collect())
            .unwrap_or_default()
    }
}

impl OtelSink for RecordingOtelSink {
    fn export(&self, span: OtelSpan) {
        if let Ok(mut g) = self.spans.lock() {
            g.push(span);
        }
    }
}

/// Live sink: emits each [`OtelSpan`] as a short-lived span on a real
/// OpenTelemetry [`Tracer`](opentelemetry::trace::Tracer). Constructed from a
/// configured global tracer in production (infra-gated; needs an OTLP exporter
/// wired by the host). Kept thin so the *mapping* — not the SDK plumbing — is
/// what carries the tested logic.
pub struct TracerOtelSink<T> {
    tracer: T,
}

impl<T> TracerOtelSink<T> {
    /// Wrap a configured OTel tracer.
    pub fn new(tracer: T) -> Self {
        Self { tracer }
    }
}

impl<T> OtelSink for TracerOtelSink<T>
where
    T: opentelemetry::trace::Tracer + Send + Sync,
{
    fn export(&self, span: OtelSpan) {
        let mut builder = self.tracer.span_builder(span.name);
        builder.attributes = Some(span.attributes);
        let _span = builder.start(&self.tracer);
        // `_span` ends (and is exported) on drop — a point-in-time metric span.
    }
}

/// Exports the pipeline's metric/error frames to OpenTelemetry via an injected
/// [`OtelSink`]. Implements [`FrameObserver`]; on each [`Frame::Metrics`] it emits
/// one span per metric record, and on a [`Frame::Error`] an error span.
pub struct OtelExporter<S: OtelSink> {
    sink: S,
    seen: Mutex<Vec<u64>>,
}

impl<S: OtelSink> OtelExporter<S> {
    /// Build an exporter that writes spans to `sink`.
    pub fn new(sink: S) -> Self {
        Self {
            sink,
            seen: Mutex::new(Vec::new()),
        }
    }

    /// Map one metrics batch into spans (pure; exposed for direct testing).
    pub fn export_metrics(&self, batch: &[MetricsData]) {
        for d in batch {
            self.sink.export(metric_span(d));
        }
    }
}

/// Map one [`MetricsData`] to an [`OtelSpan`] with pipecat's attribute keys.
fn metric_span(d: &MetricsData) -> OtelSpan {
    match d {
        MetricsData::Ttfb {
            processor,
            model,
            seconds,
        } => {
            let mut attrs = vec![
                KeyValue::new("processor", processor.clone()),
                KeyValue::new("metrics.ttfb", *seconds),
            ];
            if let Some(m) = model {
                attrs.push(KeyValue::new("gen_ai.request.model", m.clone()));
            }
            OtelSpan::new("metrics.ttfb", attrs)
        }
        MetricsData::Processing {
            processor,
            model,
            seconds,
        } => {
            let mut attrs = vec![
                KeyValue::new("processor", processor.clone()),
                KeyValue::new("metrics.processing", *seconds),
            ];
            if let Some(m) = model {
                attrs.push(KeyValue::new("gen_ai.request.model", m.clone()));
            }
            OtelSpan::new("metrics.processing", attrs)
        }
        MetricsData::LlmUsage {
            processor,
            model,
            tokens,
        } => {
            let mut attrs = vec![
                KeyValue::new("processor", processor.clone()),
                KeyValue::new("gen_ai.operation.name", "chat"),
                KeyValue::new("gen_ai.usage.input_tokens", tokens.prompt_tokens as i64),
                KeyValue::new(
                    "gen_ai.usage.output_tokens",
                    tokens.completion_tokens as i64,
                ),
                KeyValue::new("gen_ai.usage.total_tokens", tokens.total_tokens as i64),
            ];
            if let Some(m) = model {
                attrs.push(KeyValue::new("gen_ai.request.model", m.clone()));
            }
            if let Some(c) = tokens.cache_read_input_tokens {
                attrs.push(KeyValue::new(
                    "gen_ai.usage.cache_read_input_tokens",
                    c as i64,
                ));
            }
            if let Some(r) = tokens.reasoning_tokens {
                attrs.push(KeyValue::new("gen_ai.usage.reasoning_tokens", r as i64));
            }
            OtelSpan::new("gen_ai.usage", attrs)
        }
        MetricsData::TtsUsage {
            processor,
            characters,
        } => OtelSpan::new(
            "metrics.tts_usage",
            vec![
                KeyValue::new("processor", processor.clone()),
                KeyValue::new("metrics.character_count", *characters as i64),
            ],
        ),
        MetricsData::TurnPrediction {
            processor,
            is_complete,
            probability,
            e2e_processing_ms,
        } => OtelSpan::new(
            "metrics.turn",
            vec![
                KeyValue::new("processor", processor.clone()),
                KeyValue::new("turn.is_complete", *is_complete),
                KeyValue::new("turn.probability", *probability as f64),
                KeyValue::new("turn.e2e_processing_ms", *e2e_processing_ms),
            ],
        ),
    }
}

#[async_trait]
impl<S: OtelSink> FrameObserver for OtelExporter<S> {
    async fn on_push(&self, e: &FramePushEvent<'_>) {
        // Only the downstream copy of broadcast frames (avoid double export).
        if e.meta.broadcast_sibling_id.is_some() && e.direction != Direction::Downstream {
            return;
        }
        // Dedup by frame id.
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
            Frame::Metrics(batch) => self.export_metrics(batch),
            Frame::Error {
                message,
                fatal,
                processor,
            } => {
                let mut attrs = vec![
                    KeyValue::new("error.message", message.clone()),
                    KeyValue::new("error.fatal", *fatal),
                ];
                if let Some(p) = processor {
                    attrs.push(KeyValue::new("processor", p.to_string()));
                }
                self.sink.export(OtelSpan::new("pipeline.error", attrs));
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flowcat_core::processor::frame::FrameMeta;
    use flowcat_core::processor::metrics::LlmTokenUsage;
    use std::sync::Arc;

    async fn push(exp: &OtelExporter<Arc<RecordingOtelSink>>, frame: Frame) {
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
    async fn ttfb_metric_maps_to_span_with_pipecat_attrs() {
        let sink = Arc::new(RecordingOtelSink::new());
        let exp = OtelExporter::new(sink.clone());
        push(
            &exp,
            Frame::Metrics(vec![MetricsData::Ttfb {
                processor: "deepgram".into(),
                model: Some("nova-2".into()),
                seconds: 0.123,
            }]),
        )
        .await;
        let spans = sink.spans();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].name, "metrics.ttfb");
        assert_eq!(spans[0].attr("processor").as_deref(), Some("deepgram"));
        assert_eq!(
            spans[0].attr("gen_ai.request.model").as_deref(),
            Some("nova-2")
        );
        assert_eq!(spans[0].attr("metrics.ttfb").as_deref(), Some("0.123"));
    }

    #[tokio::test]
    async fn llm_usage_maps_to_gen_ai_usage_span() {
        let sink = Arc::new(RecordingOtelSink::new());
        let exp = OtelExporter::new(sink.clone());
        push(
            &exp,
            Frame::Metrics(vec![MetricsData::LlmUsage {
                processor: "openai".into(),
                model: Some("gpt-4o".into()),
                tokens: LlmTokenUsage {
                    prompt_tokens: 100,
                    completion_tokens: 50,
                    total_tokens: 150,
                    cache_read_input_tokens: Some(20),
                    reasoning_tokens: Some(10),
                    ..Default::default()
                },
            }]),
        )
        .await;
        let s = &sink.spans()[0];
        assert_eq!(s.name, "gen_ai.usage");
        assert_eq!(s.attr("gen_ai.usage.input_tokens").as_deref(), Some("100"));
        assert_eq!(s.attr("gen_ai.usage.output_tokens").as_deref(), Some("50"));
        assert_eq!(s.attr("gen_ai.usage.total_tokens").as_deref(), Some("150"));
        assert_eq!(
            s.attr("gen_ai.usage.cache_read_input_tokens").as_deref(),
            Some("20")
        );
        assert_eq!(
            s.attr("gen_ai.usage.reasoning_tokens").as_deref(),
            Some("10")
        );
    }

    #[tokio::test]
    async fn tts_usage_and_turn_map() {
        let sink = Arc::new(RecordingOtelSink::new());
        let exp = OtelExporter::new(sink.clone());
        push(
            &exp,
            Frame::Metrics(vec![
                MetricsData::TtsUsage {
                    processor: "cartesia".into(),
                    characters: 42,
                },
                MetricsData::TurnPrediction {
                    processor: "smart-turn".into(),
                    is_complete: true,
                    probability: 0.95,
                    e2e_processing_ms: 12.5,
                },
            ]),
        )
        .await;
        assert_eq!(sink.names(), vec!["metrics.tts_usage", "metrics.turn"]);
        let tts = &sink.spans()[0];
        assert_eq!(tts.attr("metrics.character_count").as_deref(), Some("42"));
        let turn = &sink.spans()[1];
        assert_eq!(turn.attr("turn.is_complete").as_deref(), Some("true"));
    }

    #[tokio::test]
    async fn error_frame_maps_to_error_span() {
        let sink = Arc::new(RecordingOtelSink::new());
        let exp = OtelExporter::new(sink.clone());
        push(
            &exp,
            Frame::Error {
                message: "boom".into(),
                fatal: true,
                processor: Some(Arc::from("llm")),
            },
        )
        .await;
        let s = &sink.spans()[0];
        assert_eq!(s.name, "pipeline.error");
        assert_eq!(s.attr("error.message").as_deref(), Some("boom"));
        assert_eq!(s.attr("error.fatal").as_deref(), Some("true"));
        assert_eq!(s.attr("processor").as_deref(), Some("llm"));
    }

    #[tokio::test]
    async fn dedups_by_frame_id_and_ignores_non_metric_frames() {
        let sink = Arc::new(RecordingOtelSink::new());
        let exp = OtelExporter::new(sink.clone());
        let frame = Frame::Metrics(vec![MetricsData::Processing {
            processor: "p".into(),
            model: None,
            seconds: 1.0,
        }]);
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
        // A non-metric frame is ignored.
        push(&exp, Frame::Text("ignored".into())).await;
        assert_eq!(sink.spans().len(), 1);
        assert_eq!(sink.spans()[0].name, "metrics.processing");
    }
}
