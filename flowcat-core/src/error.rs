// SPDX-License-Identifier: Apache-2.0
//
//! The crate-wide error type.
//!
//! Every fallible trait method in flowcat-core returns `Result<_, FlowcatError>`.

use thiserror::Error;

/// Errors surfaced by the Flowcat runtime.
///
/// Variants are intentionally broad; implementation agents may add fields or
/// new variants as needed, but should keep `FlowcatError` as the single error
/// type that flows through the trait contracts (see DESIGN.md "Trait contracts").
#[derive(Debug, Error)]
pub enum FlowcatError {
    /// The media transport / socket failed (connection closed, send error, …).
    #[error("transport error: {0}")]
    Transport(String),

    /// A per-carrier serializer could not parse or encode a frame.
    #[error("serializer error: {0}")]
    Serializer(String),

    /// The realtime speech-to-speech model (e.g. Gemini Live) errored.
    #[error("realtime LLM error: {0}")]
    Realtime(String),

    /// Audio codec / resampling failure (G.711, rubato, …).
    #[error("codec error: {0}")]
    Codec(String),

    /// Session bootstrap / finalize against the control plane failed.
    #[error("session error: {0}")]
    Session(String),

    /// A network/HTTP call failed.
    #[error("network error: {0}")]
    Network(String),

    /// (De)serialization of a protocol/JSON frame failed.
    #[error("protocol error: {0}")]
    Protocol(String),

    /// Underlying JSON (de)serialization error.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    /// Underlying I/O error.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// A catch-all for anything not yet categorized.
    #[error("{0}")]
    Other(String),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, FlowcatError>;
