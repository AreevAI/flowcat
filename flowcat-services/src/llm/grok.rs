// SPDX-License-Identifier: Apache-2.0
//
//! **Grok (xAI)** LLM — a thin **(W)rapper** over [`OpenAiLlm`](super::OpenAiLlm).
//!
//! Grok (xAI) exposes an **OpenAI-compatible** chat-completions endpoint (verified
//! against pipecat `services/grok/llm.py`, which is
//! `class GrokLLMService(OpenAILLMService)`), so this is the reference [`OpenAiLlm`]
//! pointed at [`GROK_API_BASE`] with the [`GROK_DEFAULT_MODEL`] default. Behind the
//! `llm-grok` feature (which enables `llm-openai`).

use async_trait::async_trait;
use futures::stream::BoxStream;

use flowcat_core::error::Result;
use flowcat_core::processor::frame::{Frame, LlmContext, StartParams};
use flowcat_core::service::{LlmService, Tool};

use super::{OpenAiLlm, OpenAiLlmBuilder};

/// Grok (xAI)'s OpenAI-compatible API base.
pub const GROK_API_BASE: &str = "https://api.x.ai/v1";
/// Grok (xAI)'s default model.
pub const GROK_DEFAULT_MODEL: &str = "grok-4.20-non-reasoning";

/// Grok (xAI) LLM service — an [`OpenAiLlm`] pointed at the Grok (xAI) base URL.
pub struct GrokLlm {
    inner: OpenAiLlm,
}

impl GrokLlm {
    /// Construct bound to `api_key`, using [`GROK_DEFAULT_MODEL`].
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_model(api_key, GROK_DEFAULT_MODEL)
    }

    /// Construct bound to `api_key` with an explicit `model`.
    pub fn with_model(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let inner = OpenAiLlmBuilder::new(api_key)
            .base_url(GROK_API_BASE)
            .model(model)
            .build();
        Self { inner }
    }
}

#[async_trait]
impl LlmService for GrokLlm {
    fn name(&self) -> &str {
        "grok"
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
        assert_eq!(GrokLlm::new("k").name(), "grok");
        assert_eq!(GrokLlm::with_model("k", "custom").name(), "grok");
    }
}
