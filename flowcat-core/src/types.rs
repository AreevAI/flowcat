// SPDX-License-Identifier: Apache-2.0
//
//! Shared data types that flow across the trait seams.
//!
//! Audio is **16-bit little-endian mono PCM** internally; the sample rate is
//! carried explicitly on every buffer (see DESIGN.md "Trait contracts" and
//! "Audio path"). These type shapes are CONTRACTS other agents build against —
//! their names and fields are fixed; only `todo!()` bodies elsewhere get filled.
//!
//! Module-rename housekeeping (PROCESSOR-DESIGN §8.4, step M0): this module was
//! `frame.rs`; it holds *data shapes* (`AudioChunk`, `RealtimeEvent`, …), **not**
//! pipeline frames. It was renamed `types.rs` to avoid colliding with the new
//! [`crate::processor::frame`] module (the `Frame` enum). `crate::frame` remains
//! a deprecated re-export shim for one release (see `lib.rs`).

use serde::{Deserialize, Serialize};

/// A buffer of mono PCM audio with an explicit sample rate.
///
/// `pcm` is 16-bit signed little-endian samples, one channel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioChunk {
    /// Mono 16-bit signed PCM samples.
    pub pcm: Vec<i16>,
    /// Sample rate of `pcm` in Hz (e.g. 8000 carrier, 16000 Gemini-in, 24000 Gemini-out).
    pub sample_rate: u32,
}

impl AudioChunk {
    /// Construct an `AudioChunk` from raw samples + rate.
    pub fn new(pcm: Vec<i16>, sample_rate: u32) -> Self {
        Self { pcm, sample_rate }
    }

    /// Number of samples (frames, since mono) in this chunk.
    pub fn len(&self) -> usize {
        self.pcm.len()
    }

    /// Whether this chunk carries no samples.
    pub fn is_empty(&self) -> bool {
        self.pcm.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Transport-level frames (what a MediaSocket yields / accepts).
// ---------------------------------------------------------------------------

/// An inbound WebSocket message as seen by a [`crate::transport::MediaSocket`].
///
/// Mirrors the three WS message kinds Flowcat cares about. `Binary` carries raw
/// bytes (e.g. L16/μ-law audio on some gateways); `Text` carries JSON control
/// + base64-audio frames (Plivo).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WsIn {
    /// A UTF-8 text frame (typically JSON).
    Text(String),
    /// A binary frame (raw bytes).
    Binary(Vec<u8>),
    /// The socket was closed by the peer.
    Close,
}

/// An outbound WebSocket message produced by a [`crate::serializer::MediaSerializer`]
/// and written via a [`crate::transport::MediaSocket`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WsOut {
    /// Send a UTF-8 text frame (typically JSON).
    Text(String),
    /// Send a binary frame (raw bytes).
    Binary(Vec<u8>),
}

/// The normalized result of feeding one [`WsIn`] through a per-carrier
/// [`crate::serializer::MediaSerializer`]. Carrier-specific framing is collapsed
/// into these cases.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SerIn {
    /// The media stream started; carries identifiers parsed from the carrier's
    /// start frame.
    StreamStart {
        /// Carrier-side call/stream identifier.
        call_id: String,
        /// Optional stream id (some carriers send a distinct stream sid).
        stream_id: Option<String>,
    },
    /// Decoded inbound audio from the caller.
    Audio(AudioChunk),
    /// The carrier signalled the stream should stop.
    Stop,
    /// A frame that carries no actionable content (keepalive, mark ack, …).
    Ignore,
}

// ---------------------------------------------------------------------------
// Realtime speech-to-speech model (Gemini Live) types.
// ---------------------------------------------------------------------------

/// Connection-time configuration for a [`crate::realtime::RealtimeLlm`].
///
/// Carries the initial system prompt + tool declarations and the audio I/O
/// rates the model is expected to use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeSetup {
    /// Model identifier (e.g. a Gemini Live model name).
    pub model: String,
    /// The initial system instruction / prompt.
    pub system_prompt: String,
    /// Tools (transitions + endCall, etc.) exposed to the model.
    pub tools: Vec<ToolDecl>,
    /// PCM sample rate the model expects on input (Gemini Live: 16000).
    pub input_sample_rate: u32,
    /// PCM sample rate the model emits on output (Gemini Live: 24000).
    pub output_sample_rate: u32,
}

/// An event emitted by a [`crate::realtime::RealtimeLlm`] (`next_event`).
///
/// Covers the server→client surface of the Gemini Live protocol that the
/// pipeline reacts to (see DESIGN.md "Gemini Live protocol").
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RealtimeEvent {
    /// Bot audio out (Gemini Live: 24 kHz PCM).
    AudioOut(AudioChunk),
    /// A finalized user utterance transcription (the provider's "completed"
    /// event). Rendered as a committed transcript line.
    UserText(String),
    /// An incremental/partial user-transcription delta (the provider's streaming
    /// "delta" events, one per word/segment). The pipeline accumulates these into
    /// a single growing interim line until the matching [`UserText`] finalizes it.
    UserInterimText(String),
    /// Incremental transcription of what the bot said.
    BotText(String),
    /// The model wants to invoke a tool/function.
    ToolCall {
        /// Tool/function name.
        name: String,
        /// Tool arguments as JSON.
        args: serde_json::Value,
        /// Provider-assigned call id, echoed back in the tool result.
        id: String,
    },
    /// Barge-in: the model was interrupted; the carrier's queued audio should
    /// be cleared.
    Interrupted,
    /// Token/usage accounting from the provider.
    Usage(Usage),
    /// The realtime session closed.
    Closed,
}

/// Token / usage accounting reported by the realtime model.
///
/// Fields are optional because providers report different subsets; the pipeline
/// folds these into the session [`Finalize::usage`] JSON.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    /// Input (prompt) tokens consumed.
    pub input_tokens: Option<u64>,
    /// Output (response) tokens produced.
    pub output_tokens: Option<u64>,
    /// Total tokens, if reported directly.
    pub total_tokens: Option<u64>,
    /// Provider-specific extra accounting passed through verbatim.
    pub extra: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Brain (conversation decision-maker) types.
// ---------------------------------------------------------------------------

/// A single tool/function declaration exposed to the realtime model.
///
/// `params` is a JSON-Schema object describing the function arguments
/// (transitions are typically no-arg, i.e. an empty object schema).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDecl {
    /// Function name (must be a valid identifier for the model's tool calling).
    pub name: String,
    /// Human/LLM-facing description of when to call it.
    pub description: String,
    /// JSON-Schema for the function parameters.
    pub params: serde_json::Value,
}

/// The decision a [`crate::brain::AgentBrain`] returns for a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BrainAction {
    /// Move to a new conversation state: re-prompt the model and swap its tools.
    Transition {
        /// New system prompt for the destination state.
        system_prompt: String,
        /// New tool set for the destination state.
        tools: Vec<ToolDecl>,
        /// Optional line for the bot to say on entering the state.
        say: Option<String>,
    },
    /// No state change; keep the current prompt/tools.
    Stay,
    /// End the call.
    End {
        /// Optional disposition/outcome label.
        disposition: Option<String>,
    },
}

// ---------------------------------------------------------------------------
// Session bootstrap / finalize types.
// ---------------------------------------------------------------------------

/// The resolved call context returned by [`crate::session::SessionSource::resolve`].
///
/// `brain_config` is opaque to flowcat-core (it is the embedder's graph/spec +
/// runtime options + seed vars); the host's brain implementation interprets it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedCall {
    /// Carrier/provider name (e.g. "plivo", a SIP trunk).
    pub provider: String,
    /// Opaque brain configuration (the embedder's graph/spec + runtime + seed vars).
    pub brain_config: serde_json::Value,
    /// Whether this run is already completed (idempotency / replay guard).
    pub is_completed: bool,
}

/// The finalize payload written back at the end of a call via
/// [`crate::session::SessionSource::complete`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Finalize {
    /// Usage/accounting JSON for the run.
    pub usage: serde_json::Value,
    /// Variables collected during the conversation.
    pub collected_vars: serde_json::Value,
    /// URL of the uploaded recording, if any.
    pub recording_url: Option<String>,
    /// URL of the uploaded transcript, if any.
    pub transcript_url: Option<String>,
}

/// A pre-signed upload destination returned by
/// [`crate::session::SessionSource::artifact_upload_url`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UploadTarget {
    /// The (pre-signed) URL to PUT bytes to.
    pub url: String,
    /// The **stored** object key/path for this artifact. This — not the (expiring,
    /// secret-bearing) presigned `url` — is what gets reported back as the
    /// recording/transcript reference in [`Finalize`]; the control plane resolves
    /// the key to a presigned GET on demand.
    pub key: String,
    /// The content type the upload must use.
    pub content_type: String,
}
