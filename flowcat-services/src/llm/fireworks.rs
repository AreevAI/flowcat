// SPDX-License-Identifier: Apache-2.0
//
//! **Fireworks AI** LLM — a thin **(W)rapper** over [`OpenAiLlm`](super::OpenAiLlm).
//!
//! Fireworks AI exposes an **OpenAI-compatible** chat-completions endpoint (verified
//! against pipecat `services/fireworks/llm.py`, which is
//! `class FireworksLLMService(OpenAILLMService)`), so this is the reference [`OpenAiLlm`]
//! pointed at [`FIREWORKS_API_BASE`] with the [`FIREWORKS_DEFAULT_MODEL`] default. Behind the
//! `llm-fireworks` feature (which enables `llm-openai`).

use async_trait::async_trait;
use futures::stream::BoxStream;

use flowcat_core::error::Result;
use flowcat_core::processor::frame::{Frame, LlmContext, StartParams};
use flowcat_core::service::{LlmService, Tool};

use super::{OpenAiLlm, OpenAiLlmBuilder};

/// Fireworks AI's OpenAI-compatible API base.
pub const FIREWORKS_API_BASE: &str = "https://api.fireworks.ai/inference/v1";
/// Fireworks AI's default model.
pub const FIREWORKS_DEFAULT_MODEL: &str = "accounts/fireworks/models/firefunction-v2";

/// Fireworks AI LLM service — an [`OpenAiLlm`] pointed at the Fireworks AI base URL.
pub struct FireworksLlm {
    inner: OpenAiLlm,
}

impl FireworksLlm {
    /// Construct bound to `api_key`, using [`FIREWORKS_DEFAULT_MODEL`].
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_model(api_key, FIREWORKS_DEFAULT_MODEL)
    }

    /// Construct bound to `api_key` with an explicit `model`.
    pub fn with_model(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let inner = OpenAiLlmBuilder::new(api_key)
            .base_url(FIREWORKS_API_BASE)
            .model(model)
            .build();
        Self { inner }
    }
}

#[async_trait]
impl LlmService for FireworksLlm {
    fn name(&self) -> &str {
        "fireworks"
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
        assert_eq!(FireworksLlm::new("k").name(), "fireworks");
        assert_eq!(FireworksLlm::with_model("k", "custom").name(), "fireworks");
    }
}
