// SPDX-License-Identifier: Apache-2.0
//
//! **Qwen (DashScope)** LLM — a thin **(W)rapper** over [`OpenAiLlm`](super::OpenAiLlm).
//!
//! Qwen (DashScope) exposes an **OpenAI-compatible** chat-completions endpoint (verified
//! against pipecat `services/qwen/llm.py`, which is
//! `class QwenLLMService(OpenAILLMService)`), so this is the reference [`OpenAiLlm`]
//! pointed at [`QWEN_API_BASE`] with the [`QWEN_DEFAULT_MODEL`] default. Behind the
//! `llm-qwen` feature (which enables `llm-openai`).

use async_trait::async_trait;
use futures::stream::BoxStream;

use flowcat_core::error::Result;
use flowcat_core::processor::frame::{Frame, LlmContext, StartParams};
use flowcat_core::service::{LlmService, Tool};

use super::{OpenAiLlm, OpenAiLlmBuilder};

/// Qwen (DashScope)'s OpenAI-compatible API base.
pub const QWEN_API_BASE: &str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1";
/// Qwen (DashScope)'s default model.
pub const QWEN_DEFAULT_MODEL: &str = "qwen-plus";

/// Qwen (DashScope) LLM service — an [`OpenAiLlm`] pointed at the Qwen (DashScope) base URL.
pub struct QwenLlm {
    inner: OpenAiLlm,
}

impl QwenLlm {
    /// Construct bound to `api_key`, using [`QWEN_DEFAULT_MODEL`].
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_model(api_key, QWEN_DEFAULT_MODEL)
    }

    /// Construct bound to `api_key` with an explicit `model`.
    pub fn with_model(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let inner = OpenAiLlmBuilder::new(api_key)
            .base_url(QWEN_API_BASE)
            .model(model)
            .build();
        Self { inner }
    }
}

#[async_trait]
impl LlmService for QwenLlm {
    fn name(&self) -> &str {
        "qwen"
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
        assert_eq!(QwenLlm::new("k").name(), "qwen");
        assert_eq!(QwenLlm::with_model("k", "custom").name(), "qwen");
    }
}
