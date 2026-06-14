// SPDX-License-Identifier: Apache-2.0
//
//! The pipeline `Frame` taxonomy (PROCESSOR-DESIGN §1).
//!
//! A closed [`Frame`] enum mirroring the pipecat frame tree
//! (`pipecat/src/pipecat/frames/frames.py`), plus a single [`Frame::Custom`]
//! variant carrying [`Arc<dyn CustomFrame>`] as the OSS extension hatch. Core
//! processors `match` on named variants (alloc-free, branch-predicted,
//! exhaustiveness-checked); OSS extensions ride in `Custom` and are downcast
//! only by the processors that care.
//!
//! **NOTE:** this is distinct from [`crate::types`] (today's data-shape module,
//! renamed from `frame.rs` in step M0). `Frame` here is the *pipeline* frame
//! that flows between [`FrameProcessor`](crate::processor::FrameProcessor)s.

use std::any::Any;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::processor::metrics::MetricsData;

/// Direction of frame flow. Mirrors pipecat `FrameDirection`
/// (`frame_processor.py:56`). `Downstream` = source→sink; `Upstream` = sink→source
/// (errors, end-of-task requests, RTVI acks).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Downstream,
    Upstream,
}

impl Direction {
    /// The opposite direction (used when broadcasting both ways).
    pub fn flip(self) -> Direction {
        match self {
            Direction::Downstream => Direction::Upstream,
            Direction::Upstream => Direction::Downstream,
        }
    }
}

/// Frame scheduling class — mirrors pipecat's three base classes
/// (`frames.py:95/106/118`). Drives queue priority and interruptibility:
/// `System` jumps the queue and survives interruption; `Data` is dropped on
/// interruption; `Control` is ordered like Data but also survives interruption
/// when `uninterruptible()` is set (e.g. `End`, `Stop`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameClass {
    System,
    Data,
    Control,
}

/// Process-unique monotonic frame id source (mirrors pipecat's per-object id).
static FRAME_ID: AtomicU64 = AtomicU64::new(1);

/// Allocate the next process-unique monotonic frame id.
pub fn next_frame_id() -> u64 {
    FRAME_ID.fetch_add(1, Ordering::Relaxed)
}

/// Per-frame metadata, carried **out of band** in the channel [`Envelope`] so the
/// hot enum variant (`OutputAudio(Arc<AudioFrame>)`) stays a thin pointer move
/// (PROCESSOR-DESIGN §1.1). Mirrors the pipecat `Frame` base fields
/// (`frames.py:73-79`). Cheap to clone (the extra map is `Arc`ed).
///
/// [`Envelope`]: crate::processor::Envelope
#[derive(Debug, Clone)]
pub struct FrameMeta {
    /// Process-unique monotonic id (see [`next_frame_id`]).
    pub id: u64,
    /// Human name for tracing/observers, e.g. "OutputAudio". Built lazily.
    pub name: &'static str,
    /// Presentation timestamp in **nanoseconds** on the pipeline clock, if set.
    pub pts: Option<i64>,
    /// Paired id when a frame was broadcast both directions (`frames.py:76`).
    pub broadcast_sibling_id: Option<u64>,
    /// Arbitrary sideband metadata (`Arc` so cloning a frame is cheap).
    pub extra: Option<Arc<serde_json::Map<String, serde_json::Value>>>,
    /// Transport source/destination track names (`frames.py:78-79`).
    pub transport_source: Option<Arc<str>>,
    pub transport_destination: Option<Arc<str>>,
}

impl FrameMeta {
    /// A fresh meta for `frame` with a new monotonic id and the frame's static
    /// `name`. The common path for producing a frame inside a processor.
    pub fn new(frame: &Frame) -> Self {
        Self {
            id: next_frame_id(),
            name: frame.name(),
            pts: None,
            broadcast_sibling_id: None,
            extra: None,
            transport_source: None,
            transport_destination: None,
        }
    }
}

/// OSS extension point: a frame type defined outside flowcat-core, carried in
/// [`Frame::Custom`]. Processors that understand it `downcast_ref` via
/// [`CustomFrame::as_any`]; everyone else forwards it unchanged.
pub trait CustomFrame: Any + Send + Sync + std::fmt::Debug {
    /// Scheduling class (System/Data/Control) for this custom frame.
    fn frame_class(&self) -> FrameClass;
    /// True if this frame must survive interruption (pipecat `UninterruptibleFrame`).
    fn uninterruptible(&self) -> bool {
        false
    }
    /// A stable human name for tracing/observers.
    fn name(&self) -> &'static str {
        "Custom"
    }
    /// Downcast hook so a processor can recover the concrete type.
    fn as_any(&self) -> &dyn Any;
}

/// Mono 16-bit LE PCM with an explicit sample rate and channel count.
/// Mirrors today's [`crate::types::AudioChunk`] (PROCESSOR-DESIGN §1.1) but adds
/// `num_channels`; Arc-wrapped in the [`Frame`] enum so the hot path never copies
/// PCM. Convertible from/to `AudioChunk` (mono) via [`From`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioFrame {
    pub pcm: Vec<i16>,
    pub sample_rate: u32,
    pub num_channels: u16,
}

impl AudioFrame {
    /// Mono PCM at `sample_rate`.
    pub fn mono(pcm: Vec<i16>, sample_rate: u32) -> Self {
        Self {
            pcm,
            sample_rate,
            num_channels: 1,
        }
    }

    /// Number of samples (per channel is not tracked separately — `pcm.len()`).
    pub fn len(&self) -> usize {
        self.pcm.len()
    }

    /// Whether this frame carries no samples.
    pub fn is_empty(&self) -> bool {
        self.pcm.is_empty()
    }
}

impl From<crate::types::AudioChunk> for AudioFrame {
    fn from(c: crate::types::AudioChunk) -> Self {
        AudioFrame::mono(c.pcm, c.sample_rate)
    }
}

impl From<&crate::types::AudioChunk> for AudioFrame {
    fn from(c: &crate::types::AudioChunk) -> Self {
        AudioFrame::mono(c.pcm.clone(), c.sample_rate)
    }
}

/// Lossy conversion back to the mono [`crate::types::AudioChunk`] data shape
/// (drops `num_channels`; the live path is always mono).
impl From<&AudioFrame> for crate::types::AudioChunk {
    fn from(f: &AudioFrame) -> Self {
        crate::types::AudioChunk::new(f.pcm.clone(), f.sample_rate)
    }
}

/// Pipeline init params. Mirrors pipecat `StartFrame` (`frames.py:847`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StartParams {
    pub audio_in_sample_rate: u32,
    pub audio_out_sample_rate: u32,
    pub enable_metrics: bool,
    pub enable_usage_metrics: bool,
    pub enable_tracing: bool,
    /// Report only the first TTFB metric per turn (`frame_processor.py` TTFB gate).
    pub report_only_initial_ttfb: bool,
}

impl Default for StartParams {
    fn default() -> Self {
        Self {
            audio_in_sample_rate: 16_000,
            audio_out_sample_rate: 24_000,
            enable_metrics: false,
            enable_usage_metrics: false,
            enable_tracing: false,
            report_only_initial_ttfb: true,
        }
    }
}

/// A telephony keypad entry (DTMF). Mirrors pipecat `KeypadEntry`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KeypadEntry {
    Zero,
    One,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Star,
    Pound,
}

/// A spoken-language tag (BCP-47-ish). Mirrors pipecat `Language` (string-backed
/// here so the long enum doesn't bloat core; promote to a real enum if
/// exhaustive matching is needed).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Language(pub String);

/// Which service a [`Frame::UpdateSettings`] / control frame targets. Mirrors
/// pipecat's settings-frame routing (`frames.py:1878`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServiceKind {
    Stt,
    Tts,
    Llm,
    Filter,
    Mixer,
    All,
}

/// VAD parameter bundle broadcast on [`Frame::SpeechControlParams`]. Mirrors
/// pipecat `VADParams`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VadParams {
    pub confidence: f32,
    pub start_secs: f32,
    pub stop_secs: f32,
    pub min_volume: f32,
}

/// Turn-analyzer parameter bundle. Mirrors pipecat turn params.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TurnParams {
    pub stop_secs: f32,
    pub pre_speech_ms: f32,
    pub max_duration_secs: Option<f32>,
}

/// A function/tool call request. Mirrors pipecat `FunctionCallFromLLM`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    pub function_name: String,
    pub tool_call_id: String,
    pub arguments: serde_json::Value,
}

/// A function/tool call result, fed back to the LLM. Mirrors pipecat
/// `FunctionCallResultFrame` (`frames.py:719`). Uninterruptible.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCallResult {
    pub function_name: String,
    pub tool_call_id: String,
    pub result: serde_json::Value,
}

/// The universal LLM context to run. Mirrors pipecat `LLMContext`
/// (`frames.py:502`). Opaque message list — providers interpret it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LlmContext {
    pub messages: Vec<serde_json::Value>,
    pub tools: Vec<serde_json::Value>,
}

/// The frame that flows through every processor. One closed enum mirroring the
/// pipecat frame tree (`frames.py`), plus [`Frame::Custom`] for OSS extensions.
///
/// The hot audio variants box their payload in `Arc<AudioFrame>` so cloning a
/// frame for the broadcast/observer paths never copies PCM.
#[derive(Debug, Clone)]
pub enum Frame {
    // ---- System frames (priority; survive interruption) — frames.py:846+ ----
    /// Pipeline init: carries sample rates + metric/trace toggles (`StartFrame`).
    Start(StartParams),
    /// Immediate stop; flush nothing (`CancelFrame`).
    Cancel {
        reason: Option<String>,
    },
    /// Error notification pushed upstream (`ErrorFrame`). `fatal` ⇒ task cancels.
    Error {
        message: String,
        fatal: bool,
        processor: Option<Arc<str>>,
    },
    /// Barge-in (`InterruptionFrame`). Broadcast both directions.
    Interruption,
    /// Raw caller audio from a transport input (`InputAudioRawFrame`).
    InputAudio(Arc<AudioFrame>),
    /// User-associated audio (`UserAudioRawFrame`) — carries `user_id`.
    UserAudio {
        audio: Arc<AudioFrame>,
        user_id: Arc<str>,
    },
    /// Raw text input from a transport (`InputTextRawFrame`) — text-chat path.
    InputText(String),
    /// Inbound DTMF keypress (`InputDTMFFrame`).
    InputDtmf(KeypadEntry),
    /// VAD/turn lifecycle (`frames.py:971-1104`). One variant per pipecat frame.
    UserStartedSpeaking,
    UserStoppedSpeaking,
    UserSpeaking,
    BotStartedSpeaking,
    BotStoppedSpeaking,
    BotSpeaking,
    /// Definitive VAD edges with the deciding secs (`VAD*SpeakingFrame`).
    VadUserStartedSpeaking {
        start_secs: f32,
    },
    VadUserStoppedSpeaking {
        stop_secs: f32,
    },
    /// Mute/unmute the STT service (`STTMuteFrame`).
    SttMute(bool),
    /// Performance metrics (`MetricsFrame`) — TTFB/processing/usage/turn.
    Metrics(Vec<MetricsData>),
    /// Transport-level message in/out urgent (`Input/OutputTransportMessage*`).
    TransportMessage {
        payload: serde_json::Value,
        urgent: bool,
    },
    /// SFU/transport lifecycle (`ClientConnected`).
    ClientConnected,
    /// SFU/transport lifecycle (`BotConnected`).
    BotConnected,
    /// Function-call signalling (`FunctionCallsStarted`).
    FunctionCallsStarted(Vec<FunctionCall>),
    /// Function-call signalling (`FunctionCallInProgress`).
    FunctionCallInProgress {
        call: FunctionCall,
        cancel_on_interruption: bool,
    },
    /// Function-call signalling (`FunctionCallCancel`).
    FunctionCallCancel {
        function_name: String,
        tool_call_id: String,
    },

    // ---- Data frames (ordered; dropped on interruption) — frames.py:190+ ----
    /// Output audio to a transport (`OutputAudioRawFrame`).
    OutputAudio(Arc<AudioFrame>),
    /// TTS-generated audio, tagged with its context id (`TTSAudioRawFrame`).
    TtsAudio {
        audio: Arc<AudioFrame>,
        context_id: Option<Arc<str>>,
    },
    /// Generic text (`TextFrame`) — flows LLM→aggregator→TTS.
    Text(String),
    /// LLM-generated text chunk (`LLMTextFrame`).
    LlmText(String),
    /// Final transcription (`TranscriptionFrame`).
    Transcription {
        text: String,
        user_id: Arc<str>,
        language: Option<Language>,
        final_: bool,
    },
    /// Interim/partial transcription (`InterimTranscriptionFrame`).
    InterimTranscription {
        text: String,
        user_id: Arc<str>,
        language: Option<Language>,
    },
    /// Text the TTS should speak (`TTSSpeakFrame`).
    TtsSpeak {
        text: String,
        append_to_context: Option<bool>,
    },
    /// Word/segment text emitted by TTS with its context (`TTSTextFrame`).
    TtsText {
        text: String,
        context_id: Option<Arc<str>>,
    },
    /// Function-call result, fed back to the LLM (`FunctionCallResultFrame`).
    /// Uninterruptible — once produced, context must always be updated.
    FunctionCallResult(FunctionCallResult),
    /// Trigger an LLM run over the current context (`LLMRunFrame`).
    LlmRun,
    /// The universal LLM context to run (`LLMContextFrame`).
    LlmContext(Arc<LlmContext>),
    /// Outbound DTMF (`OutputDTMFFrame`).
    OutputDtmf(Vec<KeypadEntry>),

    // ---- Control frames (ordered; End/Stop survive interruption) — :1580+ ----
    /// Graceful shutdown after flush (`EndFrame`). Uninterruptible.
    End {
        reason: Option<String>,
    },
    /// Stop but keep processors connected (`StopFrame`). Uninterruptible.
    Stop,
    /// LLM response framing (`LLMFullResponseStart`).
    LlmResponseStart,
    /// LLM response framing (`LLMFullResponseEnd`).
    LlmResponseEnd,
    /// TTS response framing (`TTSStartedFrame`).
    TtsStarted {
        context_id: Option<Arc<str>>,
    },
    /// TTS response framing (`TTSStoppedFrame`).
    TtsStopped {
        context_id: Option<Arc<str>>,
    },
    /// Update a service's settings live (`ServiceUpdateSettingsFrame`).
    /// Uninterruptible. `target` = STT/TTS/LLM/Filter/Mixer/All.
    UpdateSettings {
        target: ServiceKind,
        settings: serde_json::Value,
    },
    /// Speech-control params broadcast (`SpeechControlParamsFrame`) + VAD param
    /// updates (`VADParamsUpdateFrame`).
    SpeechControlParams {
        vad: Option<VadParams>,
        turn: Option<TurnParams>,
    },
    /// Liveness probe (`HeartbeatFrame`).
    Heartbeat {
        timestamp_ns: i64,
    },
    /// Output transport ready (`OutputTransportReadyFrame`).
    OutputTransportReady,
    /// Enable/disable an input-leg audio filter at runtime (`FilterEnableFrame`,
    /// `frames.py:1971`). Promoted out of `Custom` (PROCESSOR-DESIGN §1.2) so the
    /// [`AudioFilterProcessor`](crate::audio::filter::AudioFilterProcessor) can be
    /// toggled hot. Control class (ordered, interruptible).
    FilterEnable(bool),
    /// Enable/disable an output-leg audio mixer at runtime (`MixerEnableFrame`,
    /// `frames.py:2000`). Promoted out of `Custom`. Toggles the
    /// [`MixerProcessor`](crate::audio::mixer::MixerProcessor). Control class.
    MixerEnable(bool),

    // ---- OSS extension point ----
    Custom(Arc<dyn CustomFrame>),
}

impl Frame {
    /// Scheduling class — drives queue priority + interruptibility
    /// (PROCESSOR-DESIGN §2.3). `Custom` delegates to
    /// [`CustomFrame::frame_class`].
    pub fn class(&self) -> FrameClass {
        use Frame::*;
        match self {
            // System
            Start(_)
            | Cancel { .. }
            | Error { .. }
            | Interruption
            | InputAudio(_)
            | UserAudio { .. }
            | InputText(_)
            | InputDtmf(_)
            | UserStartedSpeaking
            | UserStoppedSpeaking
            | UserSpeaking
            | BotStartedSpeaking
            | BotStoppedSpeaking
            | BotSpeaking
            | VadUserStartedSpeaking { .. }
            | VadUserStoppedSpeaking { .. }
            | SttMute(_)
            | Metrics(_)
            | TransportMessage { .. }
            | ClientConnected
            | BotConnected
            | FunctionCallsStarted(_)
            | FunctionCallInProgress { .. }
            | FunctionCallCancel { .. } => FrameClass::System,

            // Control
            End { .. }
            | Stop
            | LlmResponseStart
            | LlmResponseEnd
            | TtsStarted { .. }
            | TtsStopped { .. }
            | UpdateSettings { .. }
            | SpeechControlParams { .. }
            | Heartbeat { .. }
            | OutputTransportReady
            | FilterEnable(_)
            | MixerEnable(_) => FrameClass::Control,

            // Data
            OutputAudio(_)
            | TtsAudio { .. }
            | Text(_)
            | LlmText(_)
            | Transcription { .. }
            | InterimTranscription { .. }
            | TtsSpeak { .. }
            | TtsText { .. }
            | FunctionCallResult(_)
            | LlmRun
            | LlmContext(_)
            | OutputDtmf(_) => FrameClass::Data,

            Custom(c) => c.frame_class(),
        }
    }

    /// True ⇒ kept in the queue and not cancelled on interruption (pipecat
    /// `UninterruptibleFrame`: End, Stop, FunctionCallResult, UpdateSettings).
    /// `Custom` delegates to [`CustomFrame::uninterruptible`].
    pub fn uninterruptible(&self) -> bool {
        use Frame::*;
        match self {
            End { .. } | Stop | FunctionCallResult(_) | UpdateSettings { .. } => true,
            Custom(c) => c.uninterruptible(),
            _ => false,
        }
    }

    /// A stable, human-readable name for tracing/observers/meta.
    pub fn name(&self) -> &'static str {
        use Frame::*;
        match self {
            Start(_) => "Start",
            Cancel { .. } => "Cancel",
            Error { .. } => "Error",
            Interruption => "Interruption",
            InputAudio(_) => "InputAudio",
            UserAudio { .. } => "UserAudio",
            InputText(_) => "InputText",
            InputDtmf(_) => "InputDtmf",
            UserStartedSpeaking => "UserStartedSpeaking",
            UserStoppedSpeaking => "UserStoppedSpeaking",
            UserSpeaking => "UserSpeaking",
            BotStartedSpeaking => "BotStartedSpeaking",
            BotStoppedSpeaking => "BotStoppedSpeaking",
            BotSpeaking => "BotSpeaking",
            VadUserStartedSpeaking { .. } => "VadUserStartedSpeaking",
            VadUserStoppedSpeaking { .. } => "VadUserStoppedSpeaking",
            SttMute(_) => "SttMute",
            Metrics(_) => "Metrics",
            TransportMessage { .. } => "TransportMessage",
            ClientConnected => "ClientConnected",
            BotConnected => "BotConnected",
            FunctionCallsStarted(_) => "FunctionCallsStarted",
            FunctionCallInProgress { .. } => "FunctionCallInProgress",
            FunctionCallCancel { .. } => "FunctionCallCancel",
            OutputAudio(_) => "OutputAudio",
            TtsAudio { .. } => "TtsAudio",
            Text(_) => "Text",
            LlmText(_) => "LlmText",
            Transcription { .. } => "Transcription",
            InterimTranscription { .. } => "InterimTranscription",
            TtsSpeak { .. } => "TtsSpeak",
            TtsText { .. } => "TtsText",
            FunctionCallResult(_) => "FunctionCallResult",
            LlmRun => "LlmRun",
            LlmContext(_) => "LlmContext",
            OutputDtmf(_) => "OutputDtmf",
            End { .. } => "End",
            Stop => "Stop",
            LlmResponseStart => "LlmResponseStart",
            LlmResponseEnd => "LlmResponseEnd",
            TtsStarted { .. } => "TtsStarted",
            TtsStopped { .. } => "TtsStopped",
            UpdateSettings { .. } => "UpdateSettings",
            SpeechControlParams { .. } => "SpeechControlParams",
            Heartbeat { .. } => "Heartbeat",
            OutputTransportReady => "OutputTransportReady",
            FilterEnable(_) => "FilterEnable",
            MixerEnable(_) => "MixerEnable",
            Custom(c) => c.name(),
        }
    }

    /// Whether this frame is a terminal lifecycle frame (`End`/`Stop`/`Cancel`)
    /// that ends a [`PipelineTask`](crate::pipeline::PipelineTask) when it reaches
    /// the sink.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Frame::End { .. } | Frame::Stop | Frame::Cancel { .. })
    }

    /// The discriminant kind, for event-hook / idle-frame matching without
    /// carrying the payload. See [`FrameKind`].
    pub fn kind(&self) -> FrameKind {
        use Frame::*;
        match self {
            Start(_) => FrameKind::Start,
            Cancel { .. } => FrameKind::Cancel,
            Error { .. } => FrameKind::Error,
            Interruption => FrameKind::Interruption,
            InputAudio(_) => FrameKind::InputAudio,
            UserAudio { .. } => FrameKind::UserAudio,
            InputText(_) => FrameKind::InputText,
            InputDtmf(_) => FrameKind::InputDtmf,
            UserStartedSpeaking => FrameKind::UserStartedSpeaking,
            UserStoppedSpeaking => FrameKind::UserStoppedSpeaking,
            UserSpeaking => FrameKind::UserSpeaking,
            BotStartedSpeaking => FrameKind::BotStartedSpeaking,
            BotStoppedSpeaking => FrameKind::BotStoppedSpeaking,
            BotSpeaking => FrameKind::BotSpeaking,
            VadUserStartedSpeaking { .. } => FrameKind::VadUserStartedSpeaking,
            VadUserStoppedSpeaking { .. } => FrameKind::VadUserStoppedSpeaking,
            SttMute(_) => FrameKind::SttMute,
            Metrics(_) => FrameKind::Metrics,
            TransportMessage { .. } => FrameKind::TransportMessage,
            ClientConnected => FrameKind::ClientConnected,
            BotConnected => FrameKind::BotConnected,
            FunctionCallsStarted(_) => FrameKind::FunctionCallsStarted,
            FunctionCallInProgress { .. } => FrameKind::FunctionCallInProgress,
            FunctionCallCancel { .. } => FrameKind::FunctionCallCancel,
            OutputAudio(_) => FrameKind::OutputAudio,
            TtsAudio { .. } => FrameKind::TtsAudio,
            Text(_) => FrameKind::Text,
            LlmText(_) => FrameKind::LlmText,
            Transcription { .. } => FrameKind::Transcription,
            InterimTranscription { .. } => FrameKind::InterimTranscription,
            TtsSpeak { .. } => FrameKind::TtsSpeak,
            TtsText { .. } => FrameKind::TtsText,
            FunctionCallResult(_) => FrameKind::FunctionCallResult,
            LlmRun => FrameKind::LlmRun,
            LlmContext(_) => FrameKind::LlmContext,
            OutputDtmf(_) => FrameKind::OutputDtmf,
            End { .. } => FrameKind::End,
            Stop => FrameKind::Stop,
            LlmResponseStart => FrameKind::LlmResponseStart,
            LlmResponseEnd => FrameKind::LlmResponseEnd,
            TtsStarted { .. } => FrameKind::TtsStarted,
            TtsStopped { .. } => FrameKind::TtsStopped,
            UpdateSettings { .. } => FrameKind::UpdateSettings,
            SpeechControlParams { .. } => FrameKind::SpeechControlParams,
            Heartbeat { .. } => FrameKind::Heartbeat,
            OutputTransportReady => FrameKind::OutputTransportReady,
            FilterEnable(_) => FrameKind::FilterEnable,
            MixerEnable(_) => FrameKind::MixerEnable,
            Custom(_) => FrameKind::Custom,
        }
    }
}

/// A payload-free discriminant for [`Frame`], used by event hooks
/// (`on_frame_reached_*`), idle-frame detection, and observers that filter by
/// type. Mirrors using `isinstance` on a pipecat frame class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrameKind {
    Start,
    Cancel,
    Error,
    Interruption,
    InputAudio,
    UserAudio,
    InputText,
    InputDtmf,
    UserStartedSpeaking,
    UserStoppedSpeaking,
    UserSpeaking,
    BotStartedSpeaking,
    BotStoppedSpeaking,
    BotSpeaking,
    VadUserStartedSpeaking,
    VadUserStoppedSpeaking,
    SttMute,
    Metrics,
    TransportMessage,
    ClientConnected,
    BotConnected,
    FunctionCallsStarted,
    FunctionCallInProgress,
    FunctionCallCancel,
    OutputAudio,
    TtsAudio,
    Text,
    LlmText,
    Transcription,
    InterimTranscription,
    TtsSpeak,
    TtsText,
    FunctionCallResult,
    LlmRun,
    LlmContext,
    OutputDtmf,
    End,
    Stop,
    LlmResponseStart,
    LlmResponseEnd,
    TtsStarted,
    TtsStopped,
    UpdateSettings,
    SpeechControlParams,
    Heartbeat,
    OutputTransportReady,
    FilterEnable,
    MixerEnable,
    Custom,
}

#[cfg(test)]
mod tests {
    use super::*;

    // A custom frame for the extension-hatch tests.
    #[derive(Debug)]
    struct MyCustom {
        class: FrameClass,
        keep: bool,
        payload: u32,
    }
    impl CustomFrame for MyCustom {
        fn frame_class(&self) -> FrameClass {
            self.class
        }
        fn uninterruptible(&self) -> bool {
            self.keep
        }
        fn name(&self) -> &'static str {
            "MyCustom"
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// Table test: every named variant's `class()` matches the pipecat base class.
    #[test]
    fn class_table_covers_every_variant() {
        let a = Arc::new(AudioFrame::mono(vec![0], 16_000));
        let uid: Arc<str> = Arc::from("u");
        let cases: Vec<(Frame, FrameClass)> = vec![
            // System
            (Frame::Start(StartParams::default()), FrameClass::System),
            (Frame::Cancel { reason: None }, FrameClass::System),
            (
                Frame::Error {
                    message: "x".into(),
                    fatal: false,
                    processor: None,
                },
                FrameClass::System,
            ),
            (Frame::Interruption, FrameClass::System),
            (Frame::InputAudio(a.clone()), FrameClass::System),
            (
                Frame::UserAudio {
                    audio: a.clone(),
                    user_id: uid.clone(),
                },
                FrameClass::System,
            ),
            (Frame::InputText("hi".into()), FrameClass::System),
            (Frame::InputDtmf(KeypadEntry::One), FrameClass::System),
            (Frame::UserStartedSpeaking, FrameClass::System),
            (Frame::UserStoppedSpeaking, FrameClass::System),
            (Frame::UserSpeaking, FrameClass::System),
            (Frame::BotStartedSpeaking, FrameClass::System),
            (Frame::BotStoppedSpeaking, FrameClass::System),
            (Frame::BotSpeaking, FrameClass::System),
            (
                Frame::VadUserStartedSpeaking { start_secs: 0.2 },
                FrameClass::System,
            ),
            (
                Frame::VadUserStoppedSpeaking { stop_secs: 0.5 },
                FrameClass::System,
            ),
            (Frame::SttMute(true), FrameClass::System),
            (Frame::Metrics(vec![]), FrameClass::System),
            (
                Frame::TransportMessage {
                    payload: serde_json::json!({}),
                    urgent: true,
                },
                FrameClass::System,
            ),
            (Frame::ClientConnected, FrameClass::System),
            (Frame::BotConnected, FrameClass::System),
            (Frame::FunctionCallsStarted(vec![]), FrameClass::System),
            (
                Frame::FunctionCallInProgress {
                    call: FunctionCall {
                        function_name: "f".into(),
                        tool_call_id: "1".into(),
                        arguments: serde_json::json!({}),
                    },
                    cancel_on_interruption: false,
                },
                FrameClass::System,
            ),
            (
                Frame::FunctionCallCancel {
                    function_name: "f".into(),
                    tool_call_id: "1".into(),
                },
                FrameClass::System,
            ),
            // Data
            (Frame::OutputAudio(a.clone()), FrameClass::Data),
            (
                Frame::TtsAudio {
                    audio: a.clone(),
                    context_id: None,
                },
                FrameClass::Data,
            ),
            (Frame::Text("t".into()), FrameClass::Data),
            (Frame::LlmText("t".into()), FrameClass::Data),
            (
                Frame::Transcription {
                    text: "t".into(),
                    user_id: uid.clone(),
                    language: None,
                    final_: true,
                },
                FrameClass::Data,
            ),
            (
                Frame::InterimTranscription {
                    text: "t".into(),
                    user_id: uid.clone(),
                    language: None,
                },
                FrameClass::Data,
            ),
            (
                Frame::TtsSpeak {
                    text: "t".into(),
                    append_to_context: None,
                },
                FrameClass::Data,
            ),
            (
                Frame::TtsText {
                    text: "t".into(),
                    context_id: None,
                },
                FrameClass::Data,
            ),
            (
                Frame::FunctionCallResult(FunctionCallResult {
                    function_name: "f".into(),
                    tool_call_id: "1".into(),
                    result: serde_json::json!({}),
                }),
                FrameClass::Data,
            ),
            (Frame::LlmRun, FrameClass::Data),
            (
                Frame::LlmContext(Arc::new(LlmContext::default())),
                FrameClass::Data,
            ),
            (
                Frame::OutputDtmf(vec![KeypadEntry::Pound]),
                FrameClass::Data,
            ),
            // Control
            (Frame::End { reason: None }, FrameClass::Control),
            (Frame::Stop, FrameClass::Control),
            (Frame::LlmResponseStart, FrameClass::Control),
            (Frame::LlmResponseEnd, FrameClass::Control),
            (Frame::TtsStarted { context_id: None }, FrameClass::Control),
            (Frame::TtsStopped { context_id: None }, FrameClass::Control),
            (
                Frame::UpdateSettings {
                    target: ServiceKind::Llm,
                    settings: serde_json::json!({}),
                },
                FrameClass::Control,
            ),
            (
                Frame::SpeechControlParams {
                    vad: None,
                    turn: None,
                },
                FrameClass::Control,
            ),
            (Frame::Heartbeat { timestamp_ns: 0 }, FrameClass::Control),
            (Frame::OutputTransportReady, FrameClass::Control),
            (Frame::FilterEnable(true), FrameClass::Control),
            (Frame::MixerEnable(false), FrameClass::Control),
        ];
        for (frame, expected) in cases {
            assert_eq!(
                frame.class(),
                expected,
                "class() mismatch for {}",
                frame.name()
            );
        }
    }

    #[test]
    fn uninterruptible_table() {
        // The four uninterruptible named frames.
        assert!(Frame::End { reason: None }.uninterruptible());
        assert!(Frame::Stop.uninterruptible());
        assert!(Frame::FunctionCallResult(FunctionCallResult {
            function_name: "f".into(),
            tool_call_id: "1".into(),
            result: serde_json::json!({}),
        })
        .uninterruptible());
        assert!(Frame::UpdateSettings {
            target: ServiceKind::All,
            settings: serde_json::json!({}),
        }
        .uninterruptible());
        // A sample of interruptible frames.
        assert!(!Frame::Interruption.uninterruptible());
        assert!(!Frame::Text("t".into()).uninterruptible());
        assert!(!Frame::OutputAudio(Arc::new(AudioFrame::mono(vec![0], 8000))).uninterruptible());
        assert!(!Frame::Start(StartParams::default()).uninterruptible());
    }

    #[test]
    fn custom_frame_delegates_class_and_uninterruptible() {
        let c = Frame::Custom(Arc::new(MyCustom {
            class: FrameClass::Control,
            keep: true,
            payload: 7,
        }));
        assert_eq!(c.class(), FrameClass::Control);
        assert!(c.uninterruptible());
        assert_eq!(c.name(), "MyCustom");
        assert_eq!(c.kind(), FrameKind::Custom);

        let d = Frame::Custom(Arc::new(MyCustom {
            class: FrameClass::Data,
            keep: false,
            payload: 9,
        }));
        assert_eq!(d.class(), FrameClass::Data);
        assert!(!d.uninterruptible());

        // Downcast recovers the concrete type.
        if let Frame::Custom(inner) = &d {
            let mc = inner.as_any().downcast_ref::<MyCustom>().expect("downcast");
            assert_eq!(mc.payload, 9);
        } else {
            panic!("expected Custom");
        }
    }

    #[test]
    fn frame_ids_are_monotonic_and_unique() {
        let a = next_frame_id();
        let b = next_frame_id();
        let c = next_frame_id();
        assert!(a < b && b < c);
    }

    #[test]
    fn audio_chunk_roundtrips_to_audio_frame() {
        let chunk = crate::types::AudioChunk::new(vec![1, 2, 3], 16_000);
        let frame: AudioFrame = (&chunk).into();
        assert_eq!(frame.num_channels, 1);
        assert_eq!(frame.pcm, vec![1, 2, 3]);
        let back: crate::types::AudioChunk = (&frame).into();
        assert_eq!(back, chunk);
    }
}
