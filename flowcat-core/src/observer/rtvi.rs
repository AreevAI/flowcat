// SPDX-License-Identifier: Apache-2.0
//
//! RTVI-protocol observer (pure frame→message mapping, no network).
//!
//! Translates pipeline [`Frame`]s into outgoing **RTVI client-protocol**
//! messages (user/bot transcription, speaking edges, LLM/TTS lifecycle,
//! function-call signalling, metrics). Mirrors pipecat's
//! `processors/frameworks/rtvi/observer.py` (`RTVIObserver`) and the v1 message
//! shapes in `.../rtvi/models.py`.
//!
//! This module is **pure**: it maps a frame to an [`RtviMessage`] and hands it to
//! an injectable [`RtviSink`]. The actual transport (WebSocket/Daily/etc.) lives
//! in the embedder; tests use the in-memory [`VecSink`]. No `reqwest`/network dep
//! is pulled into flowcat-core (the networked OTel/Sentry/Langfuse exporters live
//! in `flowcat-services` instead).
//!
//! ## What is mapped (parity with pipecat `RTVIObserver.on_push_frame`)
//!
//! | Frame | RTVI message `type` |
//! |---|---|
//! | [`Frame::UserStartedSpeaking`] | `user-started-speaking` |
//! | [`Frame::UserStoppedSpeaking`] | `user-stopped-speaking` |
//! | [`Frame::BotStartedSpeaking`] | `bot-started-speaking` |
//! | [`Frame::BotStoppedSpeaking`] | `bot-stopped-speaking` |
//! | [`Frame::SttMute`] | `user-mute-started` / `user-mute-stopped` |
//! | [`Frame::Transcription`] (`user_id != "bot"`) | `user-transcription` (`final: true`) |
//! | [`Frame::Transcription`] (`user_id == "bot"`, realtime s2s) | `bot-transcription` |
//! | [`Frame::InterimTranscription`] (`user_id != "bot"`) | `user-transcription` (`final: false`) |
//! | [`Frame::LlmResponseStart`] | `bot-llm-started` |
//! | [`Frame::LlmResponseEnd`] | `bot-llm-stopped` |
//! | [`Frame::LlmText`] | `bot-llm-text` (+ `bot-transcription` on sentence end) |
//! | [`Frame::TtsStarted`] | `bot-tts-started` |
//! | [`Frame::TtsStopped`] | `bot-tts-stopped` |
//! | [`Frame::TtsText`] | `bot-tts-text` + `bot-output` (`spoken: true`) |
//! | [`Frame::FunctionCallsStarted`] | `llm-function-call-started` (per call) |
//! | [`Frame::FunctionCallInProgress`] | `llm-function-call-in-progress` |
//! | [`Frame::FunctionCallResult`] | `llm-function-call-stopped` (`cancelled: false`) |
//! | [`Frame::FunctionCallCancel`] | `llm-function-call-stopped` (`cancelled: true`) |
//! | [`Frame::Metrics`] | `metrics` (bucketed ttfb/processing/tokens/characters) |

use std::sync::Mutex;

use async_trait::async_trait;
use serde::Serialize;
use serde_json::{json, Map, Value};

use crate::observer::{FrameObserver, FramePushEvent};
use crate::processor::frame::{Direction, Frame, FunctionCall};
use crate::processor::metrics::MetricsData;

/// RTVI protocol version this observer emits (mirrors pipecat
/// `models.PROTOCOL_VERSION`).
pub const PROTOCOL_VERSION: &str = "1.3.0";

/// The constant `label` on every RTVI message (pipecat `MESSAGE_LABEL`).
pub const MESSAGE_LABEL: &str = "rtvi-ai";

/// One outgoing RTVI server→client message. Serializes to the exact pipecat v1
/// wire shape: `{ "label": "rtvi-ai", "type": "<type>", "data"?: { … } }`.
///
/// `data` is omitted entirely (not `null`) for payload-free messages, matching
/// pipecat's `Message` model where `data` defaults to `None` and the observer
/// dumps with `exclude_none=True`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RtviMessage {
    /// Always [`MESSAGE_LABEL`].
    pub label: &'static str,
    /// The RTVI message type literal, e.g. `"user-transcription"`.
    #[serde(rename = "type")]
    pub kind: &'static str,
    /// Message payload; omitted when absent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl RtviMessage {
    fn bare(kind: &'static str) -> Self {
        Self {
            label: MESSAGE_LABEL,
            kind,
            data: None,
        }
    }

    fn with_data(kind: &'static str, data: Value) -> Self {
        Self {
            label: MESSAGE_LABEL,
            kind,
            data: Some(data),
        }
    }

    /// The RTVI message `type` (e.g. `"metrics"`), for tests/inspection.
    pub fn kind(&self) -> &'static str {
        self.kind
    }
}

/// Where the [`RtviObserver`] hands finished messages. Decouples the pure
/// frame→message mapping from any transport — the live transport (WS/Daily) in
/// the embedder implements this; tests use [`VecSink`].
pub trait RtviSink: Send + Sync {
    /// Deliver one mapped RTVI message toward the client.
    fn send(&self, message: RtviMessage);
}

/// An in-memory [`RtviSink`] that records every message — the test/inspection
/// sink. Cheap to clone-share via `Arc`.
#[derive(Debug, Default)]
pub struct VecSink {
    messages: Mutex<Vec<RtviMessage>>,
}

impl VecSink {
    /// A fresh, empty recording sink.
    pub fn new() -> Self {
        Self::default()
    }

    /// All messages recorded so far, in send order.
    pub fn messages(&self) -> Vec<RtviMessage> {
        self.messages.lock().map(|g| g.clone()).unwrap_or_default()
    }

    /// The recorded message `type`s, in order (handy for assertions).
    pub fn kinds(&self) -> Vec<&'static str> {
        self.messages
            .lock()
            .map(|g| g.iter().map(|m| m.kind).collect())
            .unwrap_or_default()
    }

    /// Drop all recorded messages.
    pub fn clear(&self) {
        if let Ok(mut g) = self.messages.lock() {
            g.clear();
        }
    }
}

impl RtviSink for VecSink {
    fn send(&self, message: RtviMessage) {
        if let Ok(mut g) = self.messages.lock() {
            g.push(message);
        }
    }
}

/// How much function-call detail an RTVI message exposes (pipecat
/// `RTVIFunctionCallReportLevel`). Defaults to [`None`](Self::None) — the most
/// secure level that still emits the event (tool-call-id only, no name/args).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FunctionCallReportLevel {
    /// Emit no event for this function call.
    Disabled,
    /// Event with `tool_call_id` only — no function name, args, or result.
    #[default]
    None,
    /// Add the function name (still no args/result).
    Name,
    /// Add function name, arguments, and result.
    Full,
}

impl FunctionCallReportLevel {
    fn shows_name(self) -> bool {
        matches!(self, Self::Name | Self::Full)
    }
}

/// Toggles for which RTVI message families the observer emits. Mirrors pipecat
/// `RTVIObserverParams`. All default to **on** (the conventional client UX),
/// except audio levels (off — those need the audio payload + a rate limiter).
#[derive(Debug, Clone, Copy)]
pub struct RtviObserverParams {
    pub bot_output_enabled: bool,
    pub bot_llm_enabled: bool,
    pub bot_tts_enabled: bool,
    pub bot_speaking_enabled: bool,
    pub user_llm_enabled: bool,
    pub user_speaking_enabled: bool,
    pub user_mute_enabled: bool,
    pub user_transcription_enabled: bool,
    pub metrics_enabled: bool,
    /// Detail level for function-call events (security gate).
    pub function_call_report_level: FunctionCallReportLevel,
}

impl Default for RtviObserverParams {
    fn default() -> Self {
        Self {
            bot_output_enabled: true,
            bot_llm_enabled: true,
            bot_tts_enabled: true,
            bot_speaking_enabled: true,
            user_llm_enabled: true,
            user_speaking_enabled: true,
            user_mute_enabled: true,
            user_transcription_enabled: true,
            metrics_enabled: true,
            function_call_report_level: FunctionCallReportLevel::Full,
        }
    }
}

/// Mutable per-call aggregation state, behind one `Mutex` so the observer stays
/// `Sync` while the `on_push` hook takes `&self`.
#[derive(Default)]
struct RtviState {
    /// Frame ids already mapped (dedup — pipecat `_frames_seen`).
    frames_seen: Vec<u64>,
    /// Accumulating bot LLM text for the legacy `bot-transcription` message.
    bot_transcription: String,
}

/// Pure frame→RTVI-message observer (PROCESSOR-DESIGN §5.1; pipecat
/// `RTVIObserver`). Plugs into the pipeline [`Observer`](crate::observer::Observer)
/// fan-out via [`FrameObserver`]; for each pushed frame it emits zero or more
/// [`RtviMessage`]s to the injected [`RtviSink`].
///
/// Like pipecat, it observes on the **push** hook (`on_push_frame`), processes
/// only the **downstream** copy of broadcast frames, and de-duplicates by frame
/// id so a frame seen at multiple links is mapped once.
pub struct RtviObserver {
    sink: std::sync::Arc<dyn RtviSink>,
    params: RtviObserverParams,
    state: Mutex<RtviState>,
}

impl RtviObserver {
    /// Build an observer that maps frames to `sink` with default params.
    pub fn new(sink: std::sync::Arc<dyn RtviSink>) -> Self {
        Self::with_params(sink, RtviObserverParams::default())
    }

    /// Build an observer with explicit [`RtviObserverParams`].
    pub fn with_params(sink: std::sync::Arc<dyn RtviSink>, params: RtviObserverParams) -> Self {
        Self {
            sink,
            params,
            state: Mutex::new(RtviState::default()),
        }
    }

    fn emit(&self, message: RtviMessage) {
        self.sink.send(message);
    }

    /// Map one pushed frame to RTVI message(s). Pure (besides the sink + the
    /// aggregation state). Exposed for direct unit testing without a pipeline.
    pub fn map_frame(&self, frame: &Frame) {
        match frame {
            Frame::UserStartedSpeaking if self.params.user_speaking_enabled => {
                self.emit(RtviMessage::bare("user-started-speaking"));
            }
            Frame::UserStoppedSpeaking if self.params.user_speaking_enabled => {
                self.emit(RtviMessage::bare("user-stopped-speaking"));
            }
            Frame::SttMute(muted) if self.params.user_mute_enabled => {
                self.emit(RtviMessage::bare(if *muted {
                    "user-mute-started"
                } else {
                    "user-mute-stopped"
                }));
            }
            Frame::BotStartedSpeaking if self.params.bot_speaking_enabled => {
                self.emit(RtviMessage::bare("bot-started-speaking"));
            }
            Frame::BotStoppedSpeaking if self.params.bot_speaking_enabled => {
                self.emit(RtviMessage::bare("bot-stopped-speaking"));
            }
            Frame::Transcription {
                text,
                user_id,
                final_,
                ..
            } => {
                // The realtime (s2s) path emits BOTH sides as `Transcription`,
                // distinguished only by `user_id` ("user" vs "bot"). Map bot text to a
                // bot message, not a user bubble. Cascaded never sends `user_id=="bot"`
                // here (its bot text flows via LlmText/TtsText), so this is safe.
                if user_id.as_ref() == "bot" {
                    if self.params.bot_output_enabled {
                        self.emit(RtviMessage::with_data(
                            "bot-transcription",
                            json!({ "text": text }),
                        ));
                    }
                } else if self.params.user_transcription_enabled {
                    self.emit(RtviMessage::with_data(
                        "user-transcription",
                        json!({
                            "text": text,
                            "user_id": user_id.as_ref(),
                            "timestamp": "",
                            "final": final_,
                        }),
                    ));
                }
            }
            Frame::InterimTranscription { text, user_id, .. } => {
                // Bot interims are dropped (only finalized bot text becomes a bubble);
                // a bot interim must never render as a user bubble.
                if user_id.as_ref() != "bot" && self.params.user_transcription_enabled {
                    self.emit(RtviMessage::with_data(
                        "user-transcription",
                        json!({
                            "text": text,
                            "user_id": user_id.as_ref(),
                            "timestamp": "",
                            "final": false,
                        }),
                    ));
                }
            }
            Frame::LlmResponseStart if self.params.bot_llm_enabled => {
                self.emit(RtviMessage::bare("bot-llm-started"));
            }
            Frame::LlmResponseEnd if self.params.bot_llm_enabled => {
                self.emit(RtviMessage::bare("bot-llm-stopped"));
            }
            Frame::LlmText(text) if self.params.bot_llm_enabled => {
                self.emit(RtviMessage::with_data(
                    "bot-llm-text",
                    json!({ "text": text }),
                ));
                self.flush_bot_transcription_on_sentence(text);
            }
            Frame::TtsStarted { .. } if self.params.bot_tts_enabled => {
                self.emit(RtviMessage::bare("bot-tts-started"));
            }
            Frame::TtsStopped { .. } if self.params.bot_tts_enabled => {
                self.emit(RtviMessage::bare("bot-tts-stopped"));
            }
            Frame::TtsText { text, .. } => {
                // The bot's spoken text — drives both `bot-output` (spoken) and
                // `bot-tts-text`, gated independently (pipecat
                // `_send_aggregated_llm_text`).
                if self.params.bot_output_enabled {
                    self.emit(RtviMessage::with_data(
                        "bot-output",
                        json!({
                            "text": text,
                            "spoken": true,
                            "aggregated_by": "word",
                        }),
                    ));
                }
                if self.params.bot_tts_enabled {
                    self.emit(RtviMessage::with_data(
                        "bot-tts-text",
                        json!({ "text": text }),
                    ));
                }
            }
            Frame::FunctionCallsStarted(calls) => {
                for call in calls {
                    self.emit_function_call_started(call);
                }
            }
            Frame::FunctionCallInProgress { call, .. } => {
                self.emit_function_call_in_progress(call);
            }
            Frame::FunctionCallResult(result) => {
                self.emit_function_call_stopped(
                    &result.function_name,
                    &result.tool_call_id,
                    false,
                    Some(&result.result),
                );
            }
            Frame::FunctionCallCancel {
                function_name,
                tool_call_id,
            } => {
                self.emit_function_call_stopped(function_name, tool_call_id, true, None);
            }
            Frame::Metrics(data) if self.params.metrics_enabled => {
                if let Some(msg) = map_metrics(data) {
                    self.emit(msg);
                }
            }
            _ => {}
        }
    }

    /// Legacy `bot-transcription`: accumulate bot LLM text and flush a full
    /// sentence when one completes (pipecat `_handle_llm_text_frame`).
    fn flush_bot_transcription_on_sentence(&self, text: &str) {
        let flush = {
            let Ok(mut st) = self.state.lock() else {
                return;
            };
            st.bot_transcription.push_str(text);
            if ends_sentence(&st.bot_transcription) && !st.bot_transcription.is_empty() {
                Some(std::mem::take(&mut st.bot_transcription))
            } else {
                None
            }
        };
        if let Some(sentence) = flush {
            self.emit(RtviMessage::with_data(
                "bot-transcription",
                json!({ "text": sentence }),
            ));
        }
    }

    fn emit_function_call_started(&self, call: &FunctionCall) {
        let level = self.params.function_call_report_level;
        if level == FunctionCallReportLevel::Disabled {
            return;
        }
        let mut data = Map::new();
        if level.shows_name() {
            data.insert("function_name".into(), json!(call.function_name));
        }
        self.emit(RtviMessage::with_data(
            "llm-function-call-started",
            Value::Object(data),
        ));
    }

    fn emit_function_call_in_progress(&self, call: &FunctionCall) {
        let level = self.params.function_call_report_level;
        if level == FunctionCallReportLevel::Disabled {
            return;
        }
        let mut data = Map::new();
        data.insert("tool_call_id".into(), json!(call.tool_call_id));
        if level.shows_name() {
            data.insert("function_name".into(), json!(call.function_name));
        }
        if level == FunctionCallReportLevel::Full {
            data.insert("arguments".into(), call.arguments.clone());
        }
        self.emit(RtviMessage::with_data(
            "llm-function-call-in-progress",
            Value::Object(data),
        ));
    }

    fn emit_function_call_stopped(
        &self,
        function_name: &str,
        tool_call_id: &str,
        cancelled: bool,
        result: Option<&Value>,
    ) {
        let level = self.params.function_call_report_level;
        if level == FunctionCallReportLevel::Disabled {
            return;
        }
        let mut data = Map::new();
        data.insert("tool_call_id".into(), json!(tool_call_id));
        data.insert("cancelled".into(), json!(cancelled));
        if level.shows_name() {
            data.insert("function_name".into(), json!(function_name));
        }
        if level == FunctionCallReportLevel::Full {
            if let Some(r) = result {
                data.insert("result".into(), r.clone());
            }
        }
        self.emit(RtviMessage::with_data(
            "llm-function-call-stopped",
            Value::Object(data),
        ));
    }
}

/// Whether `text` ends a sentence (a coarse port of pipecat
/// `match_endofsentence`): the last non-space char is terminal punctuation.
fn ends_sentence(text: &str) -> bool {
    text.trim_end()
        .chars()
        .next_back()
        .is_some_and(|c| matches!(c, '.' | '!' | '?' | '。' | '！' | '？'))
}

/// Bucket a `Frame::Metrics` payload into the RTVI `metrics` message shape
/// (pipecat `_handle_metrics`): `{ ttfb: [...], processing: [...], tokens: [...],
/// characters: [...] }`, each value mirroring pipecat's `model_dump`. Returns
/// `None` for an all-empty/unmapped batch (turn-prediction has no RTVI bucket).
fn map_metrics(data: &[MetricsData]) -> Option<RtviMessage> {
    let mut ttfb: Vec<Value> = Vec::new();
    let mut processing: Vec<Value> = Vec::new();
    let mut tokens: Vec<Value> = Vec::new();
    let mut characters: Vec<Value> = Vec::new();

    for d in data {
        match d {
            MetricsData::Ttfb {
                processor,
                model,
                seconds,
            } => ttfb.push(metric_value(processor, model.as_deref(), *seconds)),
            MetricsData::Processing {
                processor,
                model,
                seconds,
            } => processing.push(metric_value(processor, model.as_deref(), *seconds)),
            MetricsData::LlmUsage { tokens: usage, .. } => {
                tokens.push(serde_json::to_value(usage).unwrap_or(Value::Null));
            }
            MetricsData::TtsUsage {
                processor,
                characters: chars,
            } => characters.push(json!({ "processor": processor, "value": chars })),
            // Turn-prediction has no RTVI metrics bucket in pipecat — skip.
            MetricsData::TurnPrediction { .. } => {}
        }
    }

    if ttfb.is_empty() && processing.is_empty() && tokens.is_empty() && characters.is_empty() {
        return None;
    }

    let mut map = Map::new();
    if !ttfb.is_empty() {
        map.insert("ttfb".into(), Value::Array(ttfb));
    }
    if !processing.is_empty() {
        map.insert("processing".into(), Value::Array(processing));
    }
    if !tokens.is_empty() {
        map.insert("tokens".into(), Value::Array(tokens));
    }
    if !characters.is_empty() {
        map.insert("characters".into(), Value::Array(characters));
    }
    Some(RtviMessage::with_data("metrics", Value::Object(map)))
}

/// `{ processor, model?, value }` — pipecat's TTFB/Processing `model_dump`
/// (`value` is the seconds figure; `model` omitted when absent).
fn metric_value(processor: &str, model: Option<&str>, value: f64) -> Value {
    let mut m = Map::new();
    m.insert("processor".into(), json!(processor));
    if let Some(model) = model {
        m.insert("model".into(), json!(model));
    }
    m.insert("value".into(), json!(value));
    Value::Object(m)
}

#[async_trait]
impl FrameObserver for RtviObserver {
    async fn on_push(&self, e: &FramePushEvent<'_>) {
        // For broadcast frames, only the downstream copy is mapped (avoid dupes).
        if e.meta.broadcast_sibling_id.is_some() && e.direction != Direction::Downstream {
            return;
        }
        // Dedup by frame id (a frame seen at multiple links maps once).
        {
            let Ok(mut st) = self.state.lock() else {
                return;
            };
            if st.frames_seen.contains(&e.meta.id) {
                return;
            }
            st.frames_seen.push(e.meta.id);
        }
        self.map_frame(e.frame);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processor::frame::{FrameMeta, FunctionCallResult};
    use crate::processor::metrics::LlmTokenUsage;
    use std::sync::Arc;

    fn obs_with(params: RtviObserverParams) -> (RtviObserver, Arc<VecSink>) {
        let sink = Arc::new(VecSink::new());
        let obs = RtviObserver::with_params(sink.clone(), params);
        (obs, sink)
    }

    fn obs() -> (RtviObserver, Arc<VecSink>) {
        obs_with(RtviObserverParams::default())
    }

    async fn push(obs: &RtviObserver, frame: Frame) {
        let meta = FrameMeta::new(&frame);
        let e = FramePushEvent {
            source: "src",
            destination: "dst",
            frame: &frame,
            meta: &meta,
            direction: Direction::Downstream,
            timestamp_ns: 0,
        };
        obs.on_push(&e).await;
    }

    #[tokio::test]
    async fn transcription_splits_user_vs_bot_by_user_id() {
        // Realtime s2s emits both sides as Frame::Transcription differing only by
        // user_id. user → user-transcription (user bubble); bot → bot-transcription
        // (bot bubble), NOT another user bubble.
        let (obs, sink) = obs();
        push(
            &obs,
            Frame::Transcription {
                text: "hello there".into(),
                user_id: Arc::from("user"),
                language: None,
                final_: true,
            },
        )
        .await;
        push(
            &obs,
            Frame::Transcription {
                text: "hi, how can I help?".into(),
                user_id: Arc::from("bot"),
                language: None,
                final_: true,
            },
        )
        .await;
        assert_eq!(
            sink.kinds(),
            vec!["user-transcription", "bot-transcription"]
        );
    }

    #[tokio::test]
    async fn speaking_edges_map_to_rtvi() {
        let (obs, sink) = obs();
        push(&obs, Frame::UserStartedSpeaking).await;
        push(&obs, Frame::UserStoppedSpeaking).await;
        push(&obs, Frame::BotStartedSpeaking).await;
        push(&obs, Frame::BotStoppedSpeaking).await;
        assert_eq!(
            sink.kinds(),
            vec![
                "user-started-speaking",
                "user-stopped-speaking",
                "bot-started-speaking",
                "bot-stopped-speaking",
            ]
        );
    }

    #[tokio::test]
    async fn user_transcription_final_and_interim() {
        let (obs, sink) = obs();
        push(
            &obs,
            Frame::Transcription {
                text: "hello world".into(),
                user_id: Arc::from("u1"),
                language: None,
                final_: true,
            },
        )
        .await;
        push(
            &obs,
            Frame::InterimTranscription {
                text: "hel".into(),
                user_id: Arc::from("u1"),
                language: None,
            },
        )
        .await;
        let msgs = sink.messages();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].kind, "user-transcription");
        let d0 = msgs[0].data.as_ref().unwrap();
        assert_eq!(d0["text"], "hello world");
        assert_eq!(d0["user_id"], "u1");
        assert_eq!(d0["final"], true);
        let d1 = msgs[1].data.as_ref().unwrap();
        assert_eq!(d1["final"], false);
        assert_eq!(d1["text"], "hel");
    }

    #[tokio::test]
    async fn stt_mute_maps_to_user_mute() {
        let (obs, sink) = obs();
        push(&obs, Frame::SttMute(true)).await;
        push(&obs, Frame::SttMute(false)).await;
        assert_eq!(sink.kinds(), vec!["user-mute-started", "user-mute-stopped"]);
    }

    #[tokio::test]
    async fn llm_lifecycle_and_bot_transcription_on_sentence() {
        let (obs, sink) = obs();
        push(&obs, Frame::LlmResponseStart).await;
        push(&obs, Frame::LlmText("Hello".into())).await;
        push(&obs, Frame::LlmText(" there.".into())).await;
        push(&obs, Frame::LlmResponseEnd).await;
        let kinds = sink.kinds();
        // start, llm-text, llm-text + bot-transcription (sentence flush), stop
        assert_eq!(
            kinds,
            vec![
                "bot-llm-started",
                "bot-llm-text",
                "bot-llm-text",
                "bot-transcription",
                "bot-llm-stopped",
            ]
        );
        // The accumulated bot-transcription text is the full sentence.
        let bt = sink
            .messages()
            .into_iter()
            .find(|m| m.kind == "bot-transcription")
            .unwrap();
        assert_eq!(bt.data.unwrap()["text"], "Hello there.");
    }

    #[tokio::test]
    async fn tts_text_emits_bot_output_and_tts_text() {
        let (obs, sink) = obs();
        push(
            &obs,
            Frame::TtsText {
                text: "spoken".into(),
                context_id: None,
            },
        )
        .await;
        let msgs = sink.messages();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].kind, "bot-output");
        assert_eq!(msgs[0].data.as_ref().unwrap()["spoken"], true);
        assert_eq!(msgs[0].data.as_ref().unwrap()["text"], "spoken");
        assert_eq!(msgs[1].kind, "bot-tts-text");
    }

    #[tokio::test]
    async fn tts_lifecycle_maps() {
        let (obs, sink) = obs();
        push(&obs, Frame::TtsStarted { context_id: None }).await;
        push(&obs, Frame::TtsStopped { context_id: None }).await;
        assert_eq!(sink.kinds(), vec!["bot-tts-started", "bot-tts-stopped"]);
    }

    #[tokio::test]
    async fn function_call_full_level_exposes_name_args_result() {
        let (obs, sink) = obs(); // default = Full
        push(
            &obs,
            Frame::FunctionCallsStarted(vec![FunctionCall {
                function_name: "get_weather".into(),
                tool_call_id: "call-1".into(),
                arguments: json!({"city": "SG"}),
            }]),
        )
        .await;
        push(
            &obs,
            Frame::FunctionCallInProgress {
                call: FunctionCall {
                    function_name: "get_weather".into(),
                    tool_call_id: "call-1".into(),
                    arguments: json!({"city": "SG"}),
                },
                cancel_on_interruption: false,
            },
        )
        .await;
        push(
            &obs,
            Frame::FunctionCallResult(FunctionCallResult {
                function_name: "get_weather".into(),
                tool_call_id: "call-1".into(),
                result: json!({"temp": 31}),
            }),
        )
        .await;
        let msgs = sink.messages();
        assert_eq!(
            sink.kinds(),
            vec![
                "llm-function-call-started",
                "llm-function-call-in-progress",
                "llm-function-call-stopped",
            ]
        );
        // started carries the name at Full.
        assert_eq!(
            msgs[0].data.as_ref().unwrap()["function_name"],
            "get_weather"
        );
        // in-progress carries tool_call_id + name + arguments.
        let ip = msgs[1].data.as_ref().unwrap();
        assert_eq!(ip["tool_call_id"], "call-1");
        assert_eq!(ip["arguments"]["city"], "SG");
        // stopped (not cancelled) carries result.
        let st = msgs[2].data.as_ref().unwrap();
        assert_eq!(st["cancelled"], false);
        assert_eq!(st["result"]["temp"], 31);
    }

    #[tokio::test]
    async fn function_call_cancel_maps_to_stopped_cancelled() {
        let (obs, sink) = obs();
        push(
            &obs,
            Frame::FunctionCallCancel {
                function_name: "f".into(),
                tool_call_id: "c1".into(),
            },
        )
        .await;
        let m = &sink.messages()[0];
        assert_eq!(m.kind, "llm-function-call-stopped");
        assert_eq!(m.data.as_ref().unwrap()["cancelled"], true);
        assert!(m.data.as_ref().unwrap().get("result").is_none());
    }

    #[tokio::test]
    async fn function_call_none_level_hides_name_and_args() {
        let (obs, sink) = obs_with(RtviObserverParams {
            function_call_report_level: FunctionCallReportLevel::None,
            ..Default::default()
        });
        push(
            &obs,
            Frame::FunctionCallInProgress {
                call: FunctionCall {
                    function_name: "secret".into(),
                    tool_call_id: "c1".into(),
                    arguments: json!({"k": "v"}),
                },
                cancel_on_interruption: false,
            },
        )
        .await;
        let d = sink.messages()[0].data.clone().unwrap();
        assert_eq!(d["tool_call_id"], "c1");
        assert!(d.get("function_name").is_none());
        assert!(d.get("arguments").is_none());
    }

    #[tokio::test]
    async fn function_call_disabled_emits_nothing() {
        let (obs, sink) = obs_with(RtviObserverParams {
            function_call_report_level: FunctionCallReportLevel::Disabled,
            ..Default::default()
        });
        push(
            &obs,
            Frame::FunctionCallsStarted(vec![FunctionCall {
                function_name: "f".into(),
                tool_call_id: "c1".into(),
                arguments: json!({}),
            }]),
        )
        .await;
        assert!(sink.messages().is_empty());
    }

    #[tokio::test]
    async fn metrics_bucket_into_rtvi_metrics_message() {
        let (obs, sink) = obs();
        push(
            &obs,
            Frame::Metrics(vec![
                MetricsData::Ttfb {
                    processor: "stt".into(),
                    model: Some("nova-2".into()),
                    seconds: 0.12,
                },
                MetricsData::Processing {
                    processor: "llm".into(),
                    model: None,
                    seconds: 0.5,
                },
                MetricsData::LlmUsage {
                    processor: "llm".into(),
                    model: Some("gpt-4o".into()),
                    tokens: LlmTokenUsage {
                        prompt_tokens: 10,
                        completion_tokens: 20,
                        total_tokens: 30,
                        ..Default::default()
                    },
                },
                MetricsData::TtsUsage {
                    processor: "tts".into(),
                    characters: 42,
                },
                // Turn-prediction has no RTVI bucket and must be dropped.
                MetricsData::TurnPrediction {
                    processor: "turn".into(),
                    is_complete: true,
                    probability: 0.9,
                    e2e_processing_ms: 5.0,
                },
            ]),
        )
        .await;
        let msgs = sink.messages();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].kind, "metrics");
        let d = msgs[0].data.as_ref().unwrap();
        assert_eq!(d["ttfb"][0]["processor"], "stt");
        assert_eq!(d["ttfb"][0]["model"], "nova-2");
        assert_eq!(d["ttfb"][0]["value"], 0.12);
        // processing with no model omits the model key.
        assert!(d["processing"][0].get("model").is_none());
        assert_eq!(d["processing"][0]["value"], 0.5);
        assert_eq!(d["tokens"][0]["total_tokens"], 30);
        assert_eq!(d["characters"][0]["value"], 42);
        // No turn bucket.
        assert!(d.get("turn").is_none());
    }

    #[tokio::test]
    async fn metrics_with_only_turn_prediction_emits_nothing() {
        let (obs, sink) = obs();
        push(
            &obs,
            Frame::Metrics(vec![MetricsData::TurnPrediction {
                processor: "turn".into(),
                is_complete: false,
                probability: 0.1,
                e2e_processing_ms: 1.0,
            }]),
        )
        .await;
        assert!(sink.messages().is_empty());
    }

    #[tokio::test]
    async fn params_disable_suppresses_family() {
        let (obs, sink) = obs_with(RtviObserverParams {
            user_transcription_enabled: false,
            metrics_enabled: false,
            ..Default::default()
        });
        push(
            &obs,
            Frame::Transcription {
                text: "x".into(),
                user_id: Arc::from("u"),
                language: None,
                final_: true,
            },
        )
        .await;
        push(
            &obs,
            Frame::Metrics(vec![MetricsData::Ttfb {
                processor: "p".into(),
                model: None,
                seconds: 0.1,
            }]),
        )
        .await;
        assert!(sink.messages().is_empty());
    }

    #[tokio::test]
    async fn dedup_by_frame_id_maps_once() {
        let (obs, sink) = obs();
        let frame = Frame::UserStartedSpeaking;
        let meta = FrameMeta::new(&frame);
        // Same frame id pushed at two links → mapped once.
        for dst in ["a", "b"] {
            let e = FramePushEvent {
                source: "src",
                destination: dst,
                frame: &frame,
                meta: &meta,
                direction: Direction::Downstream,
                timestamp_ns: 0,
            };
            obs.on_push(&e).await;
        }
        assert_eq!(sink.kinds(), vec!["user-started-speaking"]);
    }

    #[tokio::test]
    async fn broadcast_upstream_copy_is_skipped() {
        let (obs, sink) = obs();
        let frame = Frame::UserStartedSpeaking;
        let mut meta = FrameMeta::new(&frame);
        meta.broadcast_sibling_id = Some(999);
        let e = FramePushEvent {
            source: "src",
            destination: "dst",
            frame: &frame,
            meta: &meta,
            direction: Direction::Upstream,
            timestamp_ns: 0,
        };
        obs.on_push(&e).await;
        assert!(sink.messages().is_empty());
    }

    #[test]
    fn rtvi_message_serializes_to_pipecat_wire_shape() {
        // Bare message: data omitted entirely.
        let bare = RtviMessage::bare("user-started-speaking");
        let v = serde_json::to_value(&bare).unwrap();
        assert_eq!(v["label"], "rtvi-ai");
        assert_eq!(v["type"], "user-started-speaking");
        assert!(v.get("data").is_none());

        // Data message keeps the payload under `data`.
        let with = RtviMessage::with_data("bot-llm-text", json!({"text": "hi"}));
        let v = serde_json::to_value(&with).unwrap();
        assert_eq!(v["type"], "bot-llm-text");
        assert_eq!(v["data"]["text"], "hi");
    }

    #[test]
    fn ends_sentence_detects_terminal_punctuation() {
        assert!(ends_sentence("Done."));
        assert!(ends_sentence("Really?  "));
        assert!(ends_sentence("Wow!"));
        assert!(!ends_sentence("Not yet"));
        assert!(!ends_sentence(""));
    }
}
