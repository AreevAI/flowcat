// SPDX-License-Identifier: Apache-2.0
//
//! **Together AI** LLM — a thin **(W)rapper** over [`OpenAiLlm`](super::OpenAiLlm).
//!
//! Together AI exposes an **OpenAI-compatible** chat-completions endpoint (verified
//! against pipecat `services/together/llm.py`, which is
//! `class TogetherLLMService(OpenAILLMService)`), so this is the reference [`OpenAiLlm`]
//! pointed at [`TOGETHER_API_BASE`] with the [`TOGETHER_DEFAULT_MODEL`] default. Behind the
//! `llm-together` feature (which enables `llm-openai`).

use async_trait::async_trait;
use futures::stream::BoxStream;

use flowcat_core::error::Result;
use flowcat_core::processor::frame::{Frame, LlmContext, StartParams};
use flowcat_core::service::{LlmService, Tool};

use super::{OpenAiLlm, OpenAiLlmBuilder};

/// Together AI's OpenAI-compatible API base.
pub const TOGETHER_API_BASE: &str = "https://api.together.xyz/v1";
/// Together AI's default model.
pub const TOGETHER_DEFAULT_MODEL: &str = "openai/gpt-oss-20b";

/// Together AI LLM service — an [`OpenAiLlm`] pointed at the Together AI base URL.
pub struct TogetherLlm {
    inner: OpenAiLlm,
}

impl TogetherLlm {
    /// Construct bound to `api_key`, using [`TOGETHER_DEFAULT_MODEL`].
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_model(api_key, TOGETHER_DEFAULT_MODEL)
    }

    /// Construct bound to `api_key` with an explicit `model`.
    pub fn with_model(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let inner = OpenAiLlmBuilder::new(api_key)
            .base_url(TOGETHER_API_BASE)
            .model(model)
            .build();
        Self { inner }
    }
}

#[async_trait]
impl LlmService for TogetherLlm {
    fn name(&self) -> &str {
        "together"
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
        assert_eq!(TogetherLlm::new("k").name(), "together");
        assert_eq!(TogetherLlm::with_model("k", "custom").name(), "together");
    }
}
