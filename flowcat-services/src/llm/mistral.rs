// SPDX-License-Identifier: Apache-2.0
//
//! **Mistral** LLM — a thin **(W)rapper** over [`OpenAiLlm`](super::OpenAiLlm).
//!
//! Mistral exposes an **OpenAI-compatible** chat-completions endpoint (verified
//! against pipecat `services/mistral/llm.py`, which is
//! `class MistralLLMService(OpenAILLMService)`), so this is the reference [`OpenAiLlm`]
//! pointed at [`MISTRAL_API_BASE`] with the [`MISTRAL_DEFAULT_MODEL`] default. Behind the
//! `llm-mistral` feature (which enables `llm-openai`).

use async_trait::async_trait;
use futures::stream::BoxStream;

use flowcat_core::error::Result;
use flowcat_core::processor::frame::{Frame, LlmContext, StartParams};
use flowcat_core::service::{LlmService, Tool};

use super::{OpenAiLlm, OpenAiLlmBuilder};

/// Mistral's OpenAI-compatible API base.
pub const MISTRAL_API_BASE: &str = "https://api.mistral.ai/v1";
/// Mistral's default model.
pub const MISTRAL_DEFAULT_MODEL: &str = "mistral-small-latest";

/// Mistral LLM service — an [`OpenAiLlm`] pointed at the Mistral base URL.
pub struct MistralLlm {
    inner: OpenAiLlm,
}

impl MistralLlm {
    /// Construct bound to `api_key`, using [`MISTRAL_DEFAULT_MODEL`].
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_model(api_key, MISTRAL_DEFAULT_MODEL)
    }

    /// Construct bound to `api_key` with an explicit `model`.
    pub fn with_model(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let inner = OpenAiLlmBuilder::new(api_key)
            .base_url(MISTRAL_API_BASE)
            .model(model)
            .build();
        Self { inner }
    }
}

#[async_trait]
impl LlmService for MistralLlm {
    fn name(&self) -> &str {
        "mistral"
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
        assert_eq!(MistralLlm::new("k").name(), "mistral");
        assert_eq!(MistralLlm::with_model("k", "custom").name(), "mistral");
    }
}
