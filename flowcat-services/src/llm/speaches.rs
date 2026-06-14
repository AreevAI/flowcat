// SPDX-License-Identifier: Apache-2.0
//
//! **Speaches (self-hosted, OpenAI-compatible)** LLM — a thin **(W)rapper** over [`OpenAiLlm`](super::OpenAiLlm).
//!
//! Speaches (self-hosted, OpenAI-compatible) exposes an **OpenAI-compatible** chat-completions endpoint (verified
//! against pipecat `services/speaches/llm.py`, which is
//! `class SpeachesLLMService(OpenAILLMService)`), so this is the reference [`OpenAiLlm`]
//! pointed at [`SPEACHES_API_BASE`] with the [`SPEACHES_DEFAULT_MODEL`] default. Behind the
//! `llm-speaches` feature (which enables `llm-openai`).

use async_trait::async_trait;
use futures::stream::BoxStream;

use flowcat_core::error::Result;
use flowcat_core::processor::frame::{Frame, LlmContext, StartParams};
use flowcat_core::service::{LlmService, Tool};

use super::{OpenAiLlm, OpenAiLlmBuilder};

/// Speaches (self-hosted, OpenAI-compatible)'s OpenAI-compatible API base.
pub const SPEACHES_API_BASE: &str = "http://localhost:11434/v1";
/// Speaches (self-hosted, OpenAI-compatible)'s default model.
pub const SPEACHES_DEFAULT_MODEL: &str = "gpt-4o";

/// Speaches (self-hosted, OpenAI-compatible) LLM service — an [`OpenAiLlm`] pointed at the Speaches (self-hosted, OpenAI-compatible) base URL.
pub struct SpeachesLlm {
    inner: OpenAiLlm,
}

impl SpeachesLlm {
    /// Construct bound to `api_key`, using [`SPEACHES_DEFAULT_MODEL`].
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_model(api_key, SPEACHES_DEFAULT_MODEL)
    }

    /// Construct bound to `api_key` with an explicit `model`.
    pub fn with_model(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let inner = OpenAiLlmBuilder::new(api_key)
            .base_url(SPEACHES_API_BASE)
            .model(model)
            .build();
        Self { inner }
    }
}

#[async_trait]
impl LlmService for SpeachesLlm {
    fn name(&self) -> &str {
        "speaches"
    }

    async fn start(&mut self, params: &StartParams) -> Result<()> {
        self.inner.start(params).await
    }

    async fn run_llm<'a>(&'a mut self, ctx: &'a LlmContext) -> Result<BoxStream<'a, Frame>> {
        self.inner.run_llm(ctx).await
    }

    fn set_tools(&mut self, tools: Vec<Tool>) {
        self.inner.set_tools(tools)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructs_with_base_and_default_model() {
        // Construction must not panic; the name is the provider id.
        assert_eq!(SpeachesLlm::new("k").name(), "speaches");
        assert_eq!(SpeachesLlm::with_model("k", "custom").name(), "speaches");
    }
}
