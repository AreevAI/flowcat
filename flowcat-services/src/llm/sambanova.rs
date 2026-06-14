// SPDX-License-Identifier: Apache-2.0
//
//! **SambaNova** LLM — a thin **(W)rapper** over [`OpenAiLlm`](super::OpenAiLlm).
//!
//! SambaNova exposes an **OpenAI-compatible** chat-completions endpoint (verified
//! against pipecat `services/sambanova/llm.py`, which is
//! `class SambaNovaLLMService(OpenAILLMService)`), so this is the reference [`OpenAiLlm`]
//! pointed at [`SAMBANOVA_API_BASE`] with the [`SAMBANOVA_DEFAULT_MODEL`] default. Behind the
//! `llm-sambanova` feature (which enables `llm-openai`).

use async_trait::async_trait;
use futures::stream::BoxStream;

use flowcat_core::error::Result;
use flowcat_core::processor::frame::{Frame, LlmContext, StartParams};
use flowcat_core::service::{LlmService, Tool};

use super::{OpenAiLlm, OpenAiLlmBuilder};

/// SambaNova's OpenAI-compatible API base.
pub const SAMBANOVA_API_BASE: &str = "https://api.sambanova.ai/v1";
/// SambaNova's default model.
pub const SAMBANOVA_DEFAULT_MODEL: &str = "Llama-4-Maverick-17B-128E-Instruct";

/// SambaNova LLM service — an [`OpenAiLlm`] pointed at the SambaNova base URL.
pub struct SambaNovaLlm {
    inner: OpenAiLlm,
}

impl SambaNovaLlm {
    /// Construct bound to `api_key`, using [`SAMBANOVA_DEFAULT_MODEL`].
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_model(api_key, SAMBANOVA_DEFAULT_MODEL)
    }

    /// Construct bound to `api_key` with an explicit `model`.
    pub fn with_model(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let inner = OpenAiLlmBuilder::new(api_key)
            .base_url(SAMBANOVA_API_BASE)
            .model(model)
            .build();
        Self { inner }
    }
}

#[async_trait]
impl LlmService for SambaNovaLlm {
    fn name(&self) -> &str {
        "sambanova"
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
        assert_eq!(SambaNovaLlm::new("k").name(), "sambanova");
        assert_eq!(SambaNovaLlm::with_model("k", "custom").name(), "sambanova");
    }
}
