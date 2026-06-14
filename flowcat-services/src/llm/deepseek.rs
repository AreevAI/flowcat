// SPDX-License-Identifier: Apache-2.0
//
//! **DeepSeek** LLM — a thin **(W)rapper** over [`OpenAiLlm`](super::OpenAiLlm).
//!
//! DeepSeek exposes an **OpenAI-compatible** chat-completions endpoint (verified
//! against pipecat `services/deepseek/llm.py`, which is
//! `class DeepSeekLLMService(OpenAILLMService)`), so this is the reference [`OpenAiLlm`]
//! pointed at [`DEEPSEEK_API_BASE`] with the [`DEEPSEEK_DEFAULT_MODEL`] default. Behind the
//! `llm-deepseek` feature (which enables `llm-openai`).

use async_trait::async_trait;
use futures::stream::BoxStream;

use flowcat_core::error::Result;
use flowcat_core::processor::frame::{Frame, LlmContext, StartParams};
use flowcat_core::service::{LlmService, Tool};

use super::{OpenAiLlm, OpenAiLlmBuilder};

/// DeepSeek's OpenAI-compatible API base.
pub const DEEPSEEK_API_BASE: &str = "https://api.deepseek.com/v1";
/// DeepSeek's default model.
pub const DEEPSEEK_DEFAULT_MODEL: &str = "deepseek-chat";

/// DeepSeek LLM service — an [`OpenAiLlm`] pointed at the DeepSeek base URL.
pub struct DeepSeekLlm {
    inner: OpenAiLlm,
}

impl DeepSeekLlm {
    /// Construct bound to `api_key`, using [`DEEPSEEK_DEFAULT_MODEL`].
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_model(api_key, DEEPSEEK_DEFAULT_MODEL)
    }

    /// Construct bound to `api_key` with an explicit `model`.
    pub fn with_model(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let inner = OpenAiLlmBuilder::new(api_key)
            .base_url(DEEPSEEK_API_BASE)
            .model(model)
            .build();
        Self { inner }
    }
}

#[async_trait]
impl LlmService for DeepSeekLlm {
    fn name(&self) -> &str {
        "deepseek"
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
        assert_eq!(DeepSeekLlm::new("k").name(), "deepseek");
        assert_eq!(DeepSeekLlm::with_model("k", "custom").name(), "deepseek");
    }
}
