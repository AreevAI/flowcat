// SPDX-License-Identifier: Apache-2.0
//
//! Langfuse exporter (behind `obs-langfuse`).
//!
//! A [`FrameObserver`](flowcat_core::observer::FrameObserver) that assembles
//! per-turn LLM **generations** from the pipeline frame stream and ships them to
//! Langfuse. It folds:
//!
//! - [`Frame::LlmContext`](flowcat_core::Frame::LlmContext) — the input messages
//!   (mirrors pipecat's `standardize_messages_to_chatml` shaping at the boundary);
//! - [`Frame::LlmText`](flowcat_core::Frame::LlmText) — streamed output chunks,
//!   concatenated into the generation `output`;
//! - [`Frame::Metrics`](flowcat_core::Frame::Metrics) `LlmUsage` — token usage;
//!
//! and flushes one [`LangfuseGeneration`] on
//! [`Frame::LlmResponseEnd`](flowcat_core::Frame::LlmResponseEnd).
//!
//! ## Testability (no network/keys)
//!
//! The transport is the injectable async [`LangfuseSink`]. The frame→generation
//! assembly is unit-tested against [`RecordingLangfuseSink`] — no Langfuse host,
//! no HTTP. The live [`HttpLangfuseSink`] POSTs the same [`LangfuseGeneration`]
//! JSON to the Langfuse ingestion API (`reqwest`/rustls) and is infra-gated.

use std::sync::Mutex;

use async_trait::async_trait;
use serde::Serialize;
use serde_json::Value;

use flowcat_core::observer::{FrameObserver, FramePushEvent};
use flowcat_core::processor::frame::Direction;
use flowcat_core::processor::metrics::MetricsData;
use flowcat_core::Frame;

/// Token usage on a Langfuse generation (Langfuse `usage` shape).
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct LangfuseUsage {
    #[serde(rename = "promptTokens")]
    pub prompt_tokens: u64,
    #[serde(rename = "completionTokens")]
    pub completion_tokens: u64,
    #[serde(rename = "totalTokens")]
    pub total_tokens: u64,
}

/// One LLM **generation** to export — the unit Langfuse traces. Carries the
/// input messages, concatenated output text, model, and token usage.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LangfuseGeneration {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The LLM input messages (last context), shaped as-is from `LlmContext`.
    pub input: Vec<Value>,
    /// The concatenated streamed LLM output text.
    pub output: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<LangfuseUsage>,
}

/// Where the [`LangfuseExporter`] hands finished generations. Async so the live
/// [`HttpLangfuseSink`] can `.await` its ingestion POST directly. Tests use
/// [`RecordingLangfuseSink`].
#[async_trait]
pub trait LangfuseSink: Send + Sync {
    /// Export one generation.
    async fn export(&self, generation: LangfuseGeneration);
}

#[async_trait]
impl<S: LangfuseSink + ?Sized> LangfuseSink for std::sync::Arc<S> {
    async fn export(&self, generation: LangfuseGeneration) {
        (**self).export(generation).await
    }
}

/// In-memory [`LangfuseSink`] for tests — records every generation.
#[derive(Debug, Default)]
pub struct RecordingLangfuseSink {
    generations: Mutex<Vec<LangfuseGeneration>>,
}

impl RecordingLangfuseSink {
    pub fn new() -> Self {
        Self::default()
    }

    /// Exported generations, in order.
    pub fn generations(&self) -> Vec<LangfuseGeneration> {
        self.generations
            .lock()
            .map(|g| g.clone())
            .unwrap_or_default()
    }
}

#[async_trait]
impl LangfuseSink for RecordingLangfuseSink {
    async fn export(&self, generation: LangfuseGeneration) {
        if let Ok(mut g) = self.generations.lock() {
            g.push(generation);
        }
    }
}

/// Live sink: POSTs each [`LangfuseGeneration`] to the Langfuse ingestion API.
/// Holds a `reqwest` client + the ingestion URL and a basic-auth header
/// (public/secret key). The send is `.await`ed best-effort — a failure is logged,
/// never propagated. Infra-gated; the mapping it serializes is the tested surface.
pub struct HttpLangfuseSink {
    client: reqwest::Client,
    ingestion_url: String,
    auth_header: String,
}

impl HttpLangfuseSink {
    /// Build from the ingestion URL + a pre-built `Authorization` header value
    /// (e.g. `Basic <base64(public:secret)>`). Key handling is the host's job.
    pub fn new(
        client: reqwest::Client,
        ingestion_url: impl Into<String>,
        auth_header: impl Into<String>,
    ) -> Self {
        Self {
            client,
            ingestion_url: ingestion_url.into(),
            auth_header: auth_header.into(),
        }
    }
}

#[async_trait]
impl LangfuseSink for HttpLangfuseSink {
    async fn export(&self, generation: LangfuseGeneration) {
        if let Err(e) = self
            .client
            .post(&self.ingestion_url)
            .header("Authorization", &self.auth_header)
            .json(&generation)
            .send()
            .await
        {
            tracing::warn!(error = %e, "langfuse export failed");
        }
    }
}

/// Mutable per-turn assembly state.
#[derive(Default)]
struct LangfuseState {
    frames_seen: Vec<u64>,
    /// Input messages from the most recent `LlmContext`.
    input: Vec<Value>,
    /// Concatenated streamed output text.
    output: String,
    /// Model + token usage folded in from the `LlmUsage` metric.
    model: Option<String>,
    usage: Option<LangfuseUsage>,
    /// Whether a generation is in flight (between response start/end).
    active: bool,
}

/// Assembles LLM generations from the frame stream and exports them to Langfuse
/// via an injected [`LangfuseSink`]. Implements [`FrameObserver`].
pub struct LangfuseExporter<S: LangfuseSink> {
    sink: S,
    state: Mutex<LangfuseState>,
}

impl<S: LangfuseSink> LangfuseExporter<S> {
    /// Build an exporter that writes generations to `sink`.
    pub fn new(sink: S) -> Self {
        Self {
            sink,
            state: Mutex::new(LangfuseState::default()),
        }
    }

    /// Take + reset the assembled generation, if a turn was in flight. Returns
    /// `None` when there is nothing to flush.
    fn take_generation(&self) -> Option<LangfuseGeneration> {
        let mut st = self.state.lock().ok()?;
        if !st.active {
            return None;
        }
        let gen = LangfuseGeneration {
            name: "llm-generation".into(),
            model: st.model.take(),
            input: std::mem::take(&mut st.input),
            output: std::mem::take(&mut st.output),
            usage: st.usage.take(),
        };
        st.active = false;
        Some(gen)
    }
}

#[async_trait]
impl<S: LangfuseSink> FrameObserver for LangfuseExporter<S> {
    async fn on_push(&self, e: &FramePushEvent<'_>) {
        if e.meta.broadcast_sibling_id.is_some() && e.direction != Direction::Downstream {
            return;
        }
        {
            let Ok(mut st) = self.state.lock() else {
                return;
            };
            if st.frames_seen.contains(&e.meta.id) {
                return;
            }
            st.frames_seen.push(e.meta.id);

            match e.frame {
                Frame::LlmResponseStart => {
                    st.active = true;
                    st.output.clear();
                }
                Frame::LlmContext(ctx) => {
                    st.input = ctx.messages.clone();
                }
                Frame::LlmText(chunk) => {
                    st.active = true;
                    st.output.push_str(chunk);
                }
                Frame::Metrics(batch) => {
                    for d in batch {
                        if let MetricsData::LlmUsage { model, tokens, .. } = d {
                            st.model = model.clone();
                            st.usage = Some(LangfuseUsage {
                                prompt_tokens: tokens.prompt_tokens,
                                completion_tokens: tokens.completion_tokens,
                                total_tokens: tokens.total_tokens,
                            });
                        }
                    }
                }
                // Flush happens below (outside the lock) on response end.
                Frame::LlmResponseEnd => {}
                _ => return,
            }
        }

        // Flush a complete generation when the response ends.
        if matches!(e.frame, Frame::LlmResponseEnd) {
            if let Some(gen) = self.take_generation() {
                self.sink.export(gen).await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flowcat_core::processor::frame::{FrameMeta, LlmContext};
    use flowcat_core::processor::metrics::LlmTokenUsage;
    use serde_json::json;
    use std::sync::Arc;

    async fn push(exp: &LangfuseExporter<Arc<RecordingLangfuseSink>>, frame: Frame) {
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
    async fn assembles_a_generation_from_a_turn() {
        let sink = Arc::new(RecordingLangfuseSink::new());
        let exp = LangfuseExporter::new(sink.clone());

        let ctx = LlmContext {
            messages: vec![json!({"role": "user", "content": "hi"})],
            tools: vec![],
        };
        push(&exp, Frame::LlmContext(Arc::new(ctx))).await;
        push(&exp, Frame::LlmResponseStart).await;
        push(&exp, Frame::LlmText("Hello".into())).await;
        push(&exp, Frame::LlmText(" there".into())).await;
        push(
            &exp,
            Frame::Metrics(vec![MetricsData::LlmUsage {
                processor: "openai".into(),
                model: Some("gpt-4o".into()),
                tokens: LlmTokenUsage {
                    prompt_tokens: 5,
                    completion_tokens: 2,
                    total_tokens: 7,
                    ..Default::default()
                },
            }]),
        )
        .await;
        // No flush until the response ends.
        assert!(sink.generations().is_empty());
        push(&exp, Frame::LlmResponseEnd).await;

        let gens = sink.generations();
        assert_eq!(gens.len(), 1);
        let g = &gens[0];
        assert_eq!(g.name, "llm-generation");
        assert_eq!(g.model.as_deref(), Some("gpt-4o"));
        assert_eq!(g.output, "Hello there");
        assert_eq!(g.input[0]["content"], "hi");
        let u = g.usage.as_ref().unwrap();
        assert_eq!(u.prompt_tokens, 5);
        assert_eq!(u.completion_tokens, 2);
        assert_eq!(u.total_tokens, 7);
    }

    #[tokio::test]
    async fn response_end_without_a_turn_flushes_nothing() {
        let sink = Arc::new(RecordingLangfuseSink::new());
        let exp = LangfuseExporter::new(sink.clone());
        push(&exp, Frame::LlmResponseEnd).await;
        assert!(sink.generations().is_empty());
    }

    #[tokio::test]
    async fn output_resets_between_turns() {
        let sink = Arc::new(RecordingLangfuseSink::new());
        let exp = LangfuseExporter::new(sink.clone());
        // Turn 1.
        push(&exp, Frame::LlmResponseStart).await;
        push(&exp, Frame::LlmText("one".into())).await;
        push(&exp, Frame::LlmResponseEnd).await;
        // Turn 2.
        push(&exp, Frame::LlmResponseStart).await;
        push(&exp, Frame::LlmText("two".into())).await;
        push(&exp, Frame::LlmResponseEnd).await;
        let gens = sink.generations();
        assert_eq!(gens.len(), 2);
        assert_eq!(gens[0].output, "one");
        assert_eq!(gens[1].output, "two");
    }

    #[tokio::test]
    async fn generation_serializes_with_langfuse_field_names() {
        let sink = Arc::new(RecordingLangfuseSink::new());
        let exp = LangfuseExporter::new(sink.clone());
        push(&exp, Frame::LlmResponseStart).await;
        push(&exp, Frame::LlmText("x".into())).await;
        push(
            &exp,
            Frame::Metrics(vec![MetricsData::LlmUsage {
                processor: "p".into(),
                model: None,
                tokens: LlmTokenUsage {
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    total_tokens: 2,
                    ..Default::default()
                },
            }]),
        )
        .await;
        push(&exp, Frame::LlmResponseEnd).await;
        let v = serde_json::to_value(&sink.generations()[0]).unwrap();
        assert_eq!(v["output"], "x");
        // Langfuse camelCase usage keys.
        assert_eq!(v["usage"]["promptTokens"], 1);
        assert_eq!(v["usage"]["totalTokens"], 2);
        // model omitted when None.
        assert!(v.get("model").is_none());
    }

    #[tokio::test]
    async fn dedups_by_frame_id() {
        let sink = Arc::new(RecordingLangfuseSink::new());
        let exp = LangfuseExporter::new(sink.clone());
        push(&exp, Frame::LlmResponseStart).await;
        // Same LlmText frame id seen twice → output counted once.
        let frame = Frame::LlmText("dup".into());
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
        push(&exp, Frame::LlmResponseEnd).await;
        assert_eq!(sink.generations()[0].output, "dup");
    }
}
