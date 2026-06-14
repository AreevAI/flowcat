// SPDX-License-Identifier: Apache-2.0
//
//! # flowcat-core
//!
//! A native-Rust real-time voice-agent runtime. It carries a phone/WebRTC
//! call's audio, runs a speech-to-speech model (Gemini Live first), and lets a
//! pluggable "brain" drive the conversation — all in one `tokio` process, no
//! GIL, no FFI.
//!
//! `flowcat-core` knows **nothing** about any embedder, web routing, SQL, or any
//! wire contract — it is a self-contained Apache-2.0 runtime library. The
//! embedder-specific glue lives in the consumer crate.
//!
//! ## The seams (traits)
//!
//! - [`MediaTransport`] — the transport-agnostic *decoded-audio* seam the
//!   pipeline drives (carrier WS frames or RTP look identical to it).
//! - [`MediaSocket`] — a raw bidirectional WS text/binary frame pipe.
//! - [`MediaSerializer`] — pure per-carrier WS framing (Plivo).
//!   [`WsCarrierTransport`] composes a `MediaSocket` + `MediaSerializer` into a
//!   [`MediaTransport`] (the WS/Plivo path).
//! - [`RealtimeLlm`] — streaming speech-to-speech model ([`GeminiLive`]).
//! - [`AgentBrain`] — conversation decision-maker (the host wires the engine).
//! - [`SessionSource`] — call bootstrap + finalize (the host wires `/internal`).
//!
//! [`pipeline::build_s2s_task`] / [`pipeline::build_cascaded_task`] assemble the
//! processor pipeline that orchestrates them for one call.
//!
//! See `DESIGN.md` for the full design (crate layout, trait contracts, audio
//! path, Gemini Live protocol).

pub mod agent;
pub mod audio;
pub mod brain;
pub mod codec;
pub mod error;
pub mod observer;
pub mod pipeline;
pub mod processor;
pub mod realtime;
pub mod serializer;
pub mod service;
pub mod session;
pub mod sip;
pub mod transcript;
pub mod transport;
pub mod types;

/// Deprecated re-export shim: `crate::frame` was renamed to [`crate::types`]
/// (PROCESSOR-DESIGN §8.4, step M0). Kept for one release so existing call sites
/// (`crate::frame::AudioChunk`, …) keep compiling; new code should use
/// [`crate::types`]. Do **not** confuse with [`crate::processor::frame`] (the
/// pipeline `Frame` enum).
#[deprecated(note = "renamed to `crate::types`; use that instead (see §8.4 M0)")]
pub mod frame {
    pub use crate::types::*;
}

// ---- Public contract surface (flat re-exports for ergonomic use) ----

pub use error::{FlowcatError, Result};

pub use types::{
    AudioChunk, BrainAction, Finalize, RealtimeEvent, RealtimeSetup, ResolvedCall, SerIn, ToolDecl,
    UploadTarget, Usage, WsIn, WsOut,
};

// ---- The processor framework: the live call orchestration. ----

pub use observer::{FrameEvent, FrameObserver, FramePushEvent, Observer};
pub use pipeline::{
    ParallelPipeline, Pipeline, PipelineRunner, PipelineTask, PipelineTaskParams, SourceHandle,
    SourcePump,
};
pub use processor::frame::{
    AudioFrame, CustomFrame, Direction, Frame, FrameClass, FrameMeta, KeypadEntry, Language,
    LlmContext, StartParams, VadParams,
};
pub use processor::metrics::{LlmTokenUsage, MetricsData};
pub use processor::{Envelope, FrameProcessor, Link, ProcessorSetup, StopReason};

// The trait seams.
pub use brain::AgentBrain;
pub use realtime::{RealtimeBackend, RealtimeKickoff, RealtimeLlm};
pub use serializer::MediaSerializer;
pub use session::SessionSource;
pub use transport::{MediaSocket, MediaTransport};

// The transport-agnostic inbound event.
pub use transport::MediaIn;

// Concrete implementations.
pub use realtime::{GeminiLive, ServiceRealtimeAdapter};
pub use serializer::PlivoSerializer;
pub use transport::{WsCarrierTransport, WsMediaTransport};

// Native SIP/RTP (a `MediaTransport` over a real SIP trunk).
pub use sip::{
    InboundInvite, SipAgent, SipConfig, SipTransport, DEFAULT_RTP_PORT_BASE, DEFAULT_RTP_PORT_TRIES,
};
