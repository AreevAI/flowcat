// SPDX-License-Identifier: Apache-2.0
//
//! Realtime / speech-to-speech providers.
//!
//! Home for the **NEW** realtime providers only. The
//! [`RealtimeLlmService`](flowcat_core::service::RealtimeLlmService) trait is
//! frozen in flowcat-core; each module here impls it behind its own feature.
//!
//! **Gemini is NOT here** — its client stays in `flowcat-core`
//! ([`flowcat_core::GeminiLive`], imported by the embedder); the
//! goAway/reconnect enhancement edits `flowcat-core/src/realtime/gemini_live.rs`
//! in place. Moving Gemini into this crate is deferred to a later release
//! (it would break the embedder's workspace build).

/// OpenAI Realtime (WebSocket). Behind the `realtime-openai` feature.
/// The priority full provider — full encode/decode + fixture tests.
#[cfg(feature = "realtime-openai")]
pub mod openai;

/// Azure OpenAI Realtime (WebSocket). Behind `realtime-azure`, which enables
/// `realtime-openai` (Azure reuses the OpenAI Realtime wire protocol over an
/// Azure endpoint with `api-key` auth).
#[cfg(feature = "realtime-azure")]
pub mod azure;

/// xAI Grok Realtime (WebSocket, OpenAI-Realtime-compatible). Behind
/// `realtime-grok`.
#[cfg(feature = "realtime-grok")]
pub mod grok;

/// Inworld Realtime (WebSocket, OpenAI-Realtime-style). Behind
/// `realtime-inworld`.
#[cfg(feature = "realtime-inworld")]
pub mod inworld;

/// Ultravox Realtime (join-URL WebSocket; binary PCM + JSON control). Behind
/// `realtime-ultravox`.
#[cfg(feature = "realtime-ultravox")]
pub mod ultravox;

/// Amazon Nova Sonic (AWS Bedrock bidirectional event envelopes). Behind
/// `realtime-novasonic`. The AWS SDK bidi *transport* is a live-only follow-up;
/// the fixture-skeleton here encodes/decodes the event JSON envelopes.
#[cfg(feature = "realtime-novasonic")]
pub mod nova_sonic;

// Convenience re-exports — each gated by its feature so a default build pulls
// nothing. One concrete client type per provider.
#[cfg(feature = "realtime-azure")]
pub use azure::AzureRealtime;
#[cfg(feature = "realtime-grok")]
pub use grok::GrokRealtime;
#[cfg(feature = "realtime-inworld")]
pub use inworld::InworldRealtime;
#[cfg(feature = "realtime-novasonic")]
pub use nova_sonic::NovaSonicRealtime;
#[cfg(feature = "realtime-openai")]
pub use openai::OpenAiRealtime;
#[cfg(feature = "realtime-ultravox")]
pub use ultravox::UltravoxRealtime;
