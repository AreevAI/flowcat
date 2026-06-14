// SPDX-License-Identifier: Apache-2.0
//
//! **Nebius** LLM — a thin **(W)rapper** over [`OpenAiLlm`](super::OpenAiLlm).
//!
//! Nebius exposes an **OpenAI-compatible** chat-completions endpoint (verified
//! against pipecat `services/nebius/llm.py`, which is
//! `class NebiusLLMService(OpenAILLMService)`), so this is the reference [`OpenAiLlm`]
//! pointed at [`NEBIUS_API_BASE`] with the [`NEBIUS_DEFAULT_MODEL`] default. Behind the
//! `llm-nebius` feature (which enables `llm-openai`).

use async_trait::async_trait;
use futures::stream::BoxStream;

use flowcat_core::error::Result;
use flowcat_core::processor::frame::{Frame, LlmContext, StartParams};
use flowcat_core::service::{LlmService, Tool};

use super::{OpenAiLlm, OpenAiLlmBuilder};

/// Nebius's OpenAI-compatible API base.
pub const NEBIUS_API_BASE: &str = "https://api.tokenfactory.nebius.com/v1";
/// Nebius's default model.
pub const NEBIUS_DEFAULT_MODEL: &str = "openai/gpt-oss-120b";

/// Nebius LLM service — an [`OpenAiLlm`] pointed at the Nebius base URL.
pub struct NebiusLlm {
    inner: OpenAiLlm,
}

impl NebiusLlm {
    /// Construct bound to `api_key`, using [`NEBIUS_DEFAULT_MODEL`].
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_model(api_key, NEBIUS_DEFAULT_MODEL)
    }

    /// Construct bound to `api_key` with an explicit `model`.
    pub fn with_model(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let inner = OpenAiLlmBuilder::new(api_key)
            .base_url(NEBIUS_API_BASE)
            .model(model)
            .build();
        Self { inner }
    }
}

#[async_trait]
impl LlmService for NebiusLlm {
    fn name(&self) -> &str {
        "nebius"
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
        assert_eq!(NebiusLlm::new("k").name(), "nebius");
        assert_eq!(NebiusLlm::with_model("k", "custom").name(), "nebius");
    }
}
