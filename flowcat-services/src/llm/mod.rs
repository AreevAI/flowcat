// SPDX-License-Identifier: Apache-2.0
//
//! Context-driven LLM providers.
//!
//! Impls of [`LlmService`](flowcat_core::service::LlmService). The reference impl
//! (**OpenAI** chat-completions streaming) sits behind `llm-openai`; it also speaks
//! to **OpenRouter** and any OpenAI-compatible endpoint via a `base_url` override.
//! Each provider is `dep:`-feature-gated so the default build pulls no LLM dep.
//!
//! ## Provider homes (PROVIDERS.md §1)
//!
//! The LLM list is overwhelmingly **OpenAI-compatible** — 18 of the providers are
//! thin **(W)rappers** over [`OpenAiLlm`] (a `base_url` + default-model change), and
//! only 4 are **(D)istinct** clients (anthropic, google/gemini,
//! openai_responses, aws_bedrock). Every provider has a stub module below so
//! adding a provider fills one body and never edits this `mod`/`use` list.

#[cfg(feature = "llm-openai")]
pub mod openai;
#[cfg(feature = "llm-openai")]
pub use openai::{OpenAiLlm, OpenAiLlmBuilder};

// --- (D)istinct LLM clients (PROVIDERS.md §1 — own wire protocols) ---
#[cfg(feature = "llm-anthropic")]
pub mod anthropic;
#[cfg(feature = "llm-anthropic")]
pub use anthropic::AnthropicLlm;
#[cfg(feature = "llm-openai-responses")]
pub mod openai_responses;
#[cfg(feature = "llm-openai-responses")]
pub use openai_responses::OpenAiResponsesLlm;
#[cfg(feature = "llm-google")]
pub mod google;
#[cfg(feature = "llm-google")]
pub use google::GoogleLlm;
// Vertex shares Gemini's wire format (reuses google's body builder + SSE decode) but
// on the Vertex surface (regional aiplatform host + OAuth2 Bearer); enables llm-google.
#[cfg(feature = "llm-google-vertex")]
pub mod google_vertex;
#[cfg(feature = "llm-google-vertex")]
pub use google_vertex::GoogleVertexLlm;
#[cfg(feature = "llm-aws-bedrock")]
pub mod aws_bedrock;
#[cfg(feature = "llm-aws-bedrock")]
pub use aws_bedrock::AwsBedrockLlm;

// --- (W)rapper LLM clients (OpenAI-compatible — base_url over OpenAiLlm) ---
#[cfg(feature = "llm-groq")]
pub mod groq;
#[cfg(feature = "llm-groq")]
pub use groq::GroqLlm;
#[cfg(feature = "llm-together")]
pub mod together;
#[cfg(feature = "llm-together")]
pub use together::TogetherLlm;
#[cfg(feature = "llm-fireworks")]
pub mod fireworks;
#[cfg(feature = "llm-fireworks")]
pub use fireworks::FireworksLlm;
#[cfg(feature = "llm-openrouter")]
pub mod openrouter;
#[cfg(feature = "llm-openrouter")]
pub use openrouter::OpenRouterLlm;
#[cfg(feature = "llm-perplexity")]
pub mod perplexity;
#[cfg(feature = "llm-perplexity")]
pub use perplexity::PerplexityLlm;
#[cfg(feature = "llm-deepseek")]
pub mod deepseek;
#[cfg(feature = "llm-deepseek")]
pub use deepseek::DeepSeekLlm;
#[cfg(feature = "llm-cerebras")]
pub mod cerebras;
#[cfg(feature = "llm-cerebras")]
pub use cerebras::CerebrasLlm;
#[cfg(feature = "llm-minimax")]
pub mod minimax;
#[cfg(feature = "llm-minimax")]
pub use minimax::MiniMaxLlm;
#[cfg(feature = "llm-sambanova")]
pub mod sambanova;
#[cfg(feature = "llm-sambanova")]
pub use sambanova::SambaNovaLlm;
#[cfg(feature = "llm-nebius")]
pub mod nebius;
#[cfg(feature = "llm-nebius")]
pub use nebius::NebiusLlm;
#[cfg(feature = "llm-novita")]
pub mod novita;
#[cfg(feature = "llm-novita")]
pub use novita::NovitaLlm;
#[cfg(feature = "llm-qwen")]
pub mod qwen;
#[cfg(feature = "llm-qwen")]
pub use qwen::QwenLlm;
#[cfg(feature = "llm-grok")]
pub mod grok;
#[cfg(feature = "llm-grok")]
pub use grok::GrokLlm;
#[cfg(feature = "llm-nvidia-nim")]
pub mod nvidia_nim;
#[cfg(feature = "llm-nvidia-nim")]
pub use nvidia_nim::NvidiaNimLlm;
#[cfg(feature = "llm-ollama")]
pub mod ollama;
#[cfg(feature = "llm-ollama")]
pub use ollama::OllamaLlm;
#[cfg(feature = "llm-sarvam")]
pub mod sarvam;
#[cfg(feature = "llm-sarvam")]
pub use sarvam::SarvamLlm;
#[cfg(feature = "llm-mistral")]
pub mod mistral;
#[cfg(feature = "llm-mistral")]
pub use mistral::MistralLlm;
#[cfg(feature = "llm-azure")]
pub mod azure;
#[cfg(feature = "llm-azure")]
pub use azure::AzureLlm;
#[cfg(feature = "llm-speaches")]
pub mod speaches;
#[cfg(feature = "llm-speaches")]
pub use speaches::SpeachesLlm;
