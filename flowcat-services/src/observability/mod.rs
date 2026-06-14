// SPDX-License-Identifier: Apache-2.0
//
//! Networked observability exporters.
//!
//! These [`FrameObserver`](flowcat_core::observer::FrameObserver) impls wire the
//! `Frame::Metrics` stream out to external sinks — they pull network deps
//! (`reqwest`/OTel SDK) so they live here, NOT in flowcat-core (the trait +
//! metrics frames stay in core). Each exporter is behind its own feature.
//!
//! - [`otel`] — OpenTelemetry (behind `obs-otel`).
//! - [`sentry`] — Sentry (behind `obs-sentry`).
//! - [`langfuse`] — Langfuse (behind `obs-langfuse`).

#[cfg(feature = "obs-otel")]
pub mod otel;
#[cfg(feature = "obs-otel")]
pub use otel::{OtelExporter, OtelSink, OtelSpan, RecordingOtelSink, TracerOtelSink};

#[cfg(feature = "obs-sentry")]
pub mod sentry;
#[cfg(feature = "obs-sentry")]
pub use sentry::{
    HttpSentrySink, RecordingSentrySink, SentryEvent, SentryExporter, SentryLevel, SentrySink,
};

#[cfg(feature = "obs-langfuse")]
pub mod langfuse;
#[cfg(feature = "obs-langfuse")]
pub use langfuse::{
    HttpLangfuseSink, LangfuseExporter, LangfuseGeneration, LangfuseSink, LangfuseUsage,
    RecordingLangfuseSink,
};
