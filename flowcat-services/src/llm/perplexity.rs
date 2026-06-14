// SPDX-License-Identifier: Apache-2.0
//
//! **Perplexity** LLM — a thin **(W)rapper** over [`OpenAiLlm`](super::OpenAiLlm).
//!
//! Perplexity exposes an **OpenAI-compatible** chat-completions endpoint (verified
//! against pipecat `services/perplexity/llm.py`, which is
//! `class PerplexityLLMService(OpenAILLMService)`), so this is the reference [`OpenAiLlm`]
//! pointed at [`PERPLEXITY_API_BASE`] with the [`PERPLEXITY_DEFAULT_MODEL`] default. Behind the
//! `llm-perplexity` feature (which enables `llm-openai`).

use async_trait::async_trait;
use futures::stream::BoxStream;

use flowcat_core::error::Result;
use flowcat_core::processor::frame::{Frame, LlmContext, StartParams};
use flowcat_core::service::{LlmService, Tool};

use super::{OpenAiLlm, OpenAiLlmBuilder};

/// Perplexity's OpenAI-compatible API base.
pub const PERPLEXITY_API_BASE: &str = "https://api.perplexity.ai";
/// Perplexity's default model.
pub const PERPLEXITY_DEFAULT_MODEL: &str = "sonar";

/// Perplexity LLM service — an [`OpenAiLlm`] pointed at the Perplexity base URL.
pub struct PerplexityLlm {
    inner: OpenAiLlm,
}

impl PerplexityLlm {
    /// Construct bound to `api_key`, using [`PERPLEXITY_DEFAULT_MODEL`].
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_model(api_key, PERPLEXITY_DEFAULT_MODEL)
    }

    /// Construct bound to `api_key` with an explicit `model`.
    pub fn with_model(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let inner = OpenAiLlmBuilder::new(api_key)
            .base_url(PERPLEXITY_API_BASE)
            .model(model)
            .build();
        Self { inner }
    }
}

#[async_trait]
impl LlmService for PerplexityLlm {
    fn name(&self) -> &str {
        "perplexity"
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
        assert_eq!(PerplexityLlm::new("k").name(), "perplexity");
        assert_eq!(
            PerplexityLlm::with_model("k", "custom").name(),
            "perplexity"
        );
    }
}
