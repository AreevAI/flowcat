// SPDX-License-Identifier: Apache-2.0
//
//! **Ollama (local)** LLM — a thin **(W)rapper** over [`OpenAiLlm`](super::OpenAiLlm).
//!
//! Ollama (local) exposes an **OpenAI-compatible** chat-completions endpoint (verified
//! against pipecat `services/ollama/llm.py`, which is
//! `class OLLamaLLMService(OpenAILLMService)`), so this is the reference [`OpenAiLlm`]
//! pointed at [`OLLAMA_API_BASE`] with the [`OLLAMA_DEFAULT_MODEL`] default. Behind the
//! `llm-ollama` feature (which enables `llm-openai`).

use async_trait::async_trait;
use futures::stream::BoxStream;

use flowcat_core::error::Result;
use flowcat_core::processor::frame::{Frame, LlmContext, StartParams};
use flowcat_core::service::{LlmService, Tool};

use super::{OpenAiLlm, OpenAiLlmBuilder};

/// Ollama (local)'s OpenAI-compatible API base.
pub const OLLAMA_API_BASE: &str = "http://localhost:11434/v1";
/// Ollama (local)'s default model.
pub const OLLAMA_DEFAULT_MODEL: &str = "llama2";

/// Ollama (local) LLM service — an [`OpenAiLlm`] pointed at the Ollama (local) base URL.
pub struct OllamaLlm {
    inner: OpenAiLlm,
}

impl OllamaLlm {
    /// Construct bound to `api_key`, using [`OLLAMA_DEFAULT_MODEL`].
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_model(api_key, OLLAMA_DEFAULT_MODEL)
    }

    /// Construct bound to `api_key` with an explicit `model`.
    pub fn with_model(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let inner = OpenAiLlmBuilder::new(api_key)
            .base_url(OLLAMA_API_BASE)
            .model(model)
            .build();
        Self { inner }
    }
}

#[async_trait]
impl LlmService for OllamaLlm {
    fn name(&self) -> &str {
        "ollama"
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
        assert_eq!(OllamaLlm::new("k").name(), "ollama");
        assert_eq!(OllamaLlm::with_model("k", "custom").name(), "ollama");
    }
}
