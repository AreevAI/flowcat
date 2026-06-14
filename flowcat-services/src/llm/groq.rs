// SPDX-License-Identifier: Apache-2.0
//
//! **Groq** LLM — a thin **(W)rapper** over [`OpenAiLlm`](super::OpenAiLlm).
//!
//! Groq exposes an **OpenAI-compatible** chat-completions endpoint (verified
//! against pipecat `services/groq/llm.py`, which is
//! `class GroqLLMService(OpenAILLMService)`), so this is the reference [`OpenAiLlm`]
//! pointed at [`GROQ_API_BASE`] with the [`GROQ_DEFAULT_MODEL`] default. Behind the
//! `llm-groq` feature (which enables `llm-openai`).

use async_trait::async_trait;
use futures::stream::BoxStream;

use flowcat_core::error::Result;
use flowcat_core::processor::frame::{Frame, LlmContext, StartParams};
use flowcat_core::service::{LlmService, Tool};

use super::{OpenAiLlm, OpenAiLlmBuilder};

/// Groq's OpenAI-compatible API base.
pub const GROQ_API_BASE: &str = "https://api.groq.com/openai/v1";
/// Groq's default model.
pub const GROQ_DEFAULT_MODEL: &str = "llama-3.3-70b-versatile";

/// Groq LLM service — an [`OpenAiLlm`] pointed at the Groq base URL.
pub struct GroqLlm {
    inner: OpenAiLlm,
}

impl GroqLlm {
    /// Construct bound to `api_key`, using [`GROQ_DEFAULT_MODEL`].
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_model(api_key, GROQ_DEFAULT_MODEL)
    }

    /// Construct bound to `api_key` with an explicit `model`.
    pub fn with_model(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let inner = OpenAiLlmBuilder::new(api_key)
            .base_url(GROQ_API_BASE)
            .model(model)
            .build();
        Self { inner }
    }
}

#[async_trait]
impl LlmService for GroqLlm {
    fn name(&self) -> &str {
        "groq"
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
        assert_eq!(GroqLlm::new("k").name(), "groq");
        assert_eq!(GroqLlm::with_model("k", "custom").name(), "groq");
    }
}
