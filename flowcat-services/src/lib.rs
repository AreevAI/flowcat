// SPDX-License-Identifier: Apache-2.0
//
//! # flowcat-services
//!
//! Speech/LLM **service providers** for the Flowcat runtime: realtime
//! speech-to-speech, streaming STT, streaming TTS, context-driven LLM, the
//! networked observability **exporters**, and MCP-as-processor.
//!
//! These impl the service **traits frozen in
//! [`flowcat_core::service`]** (`SttService`/`TtsService`/`LlmService`/
//! `RealtimeLlmService`) + the [`flowcat_core::observer::FrameObserver`] seam.
//! flowcat-core never depends back on this crate (the forbidden core→sibling
//! edge); the traits live in core, the network-pulling impls live here.
//!
//! ## Note
//!
//! Every provider is behind its own `dep:`-gated Cargo feature (see
//! `Cargo.toml`), so a **default build compiles nothing but these module stubs**
//! — no `reqwest`/`tonic`/`tokio-tungstenite`/`whisper-rs`/`opentelemetry` is
//! pulled until a feature is enabled. Adding a provider fills the body of its
//! already-declared module; it never adds a `mod` decl or a dep line.
//!
//! **The Gemini realtime client stays in `flowcat-core`** (re-exported as
//! [`flowcat_core::GeminiLive`], imported directly by the embedder); only the
//! NEW realtime providers live here. Moving Gemini into this crate is deferred
//! (it would break the embedder's workspace build).

pub mod realtime;

pub mod stt;

pub mod tts;

pub mod llm;

/// Provider **factory** — construct a boxed service from a provider *name* +
/// options (string → `Box<dyn SttService>` etc.), for config-driven hosts.
pub mod factory;
pub use factory::ProviderSpec;

pub mod observability;

/// MCP-as-processor (pulls an MCP/HTTP client → sibling, not core).
#[cfg(feature = "mcp")]
pub mod mcp;

/// Remote "brain" (HTTP webhook) adapter: a reference [`flowcat_core::AgentBrain`]
/// impl that drives the conversation policy from an out-of-process HTTP service.
#[cfg(feature = "brain-http")]
pub mod brain;
#[cfg(feature = "brain-http")]
pub use brain::remote::RemoteBrain;
