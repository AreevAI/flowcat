// SPDX-License-Identifier: Apache-2.0
//
//! **Novita AI** LLM — a thin **(W)rapper** over [`OpenAiLlm`](super::OpenAiLlm).
//!
//! Novita AI exposes an **OpenAI-compatible** chat-completions endpoint (verified
//! against pipecat `services/novita/llm.py`, which is
//! `class NovitaLLMService(OpenAILLMService)`), so this is the reference [`OpenAiLlm`]
//! pointed at [`NOVITA_API_BASE`] with the [`NOVITA_DEFAULT_MODEL`] default. Behind the
//! `llm-novita` feature (which enables `llm-openai`).

use async_trait::async_trait;
use futures::stream::BoxStream;

use flowcat_core::error::Result;
use flowcat_core::processor::frame::{Frame, LlmContext, StartParams};
use flowcat_core::service::{LlmService, Tool};

use super::{OpenAiLlm, OpenAiLlmBuilder};

/// Novita AI's OpenAI-compatible API base.
pub const NOVITA_API_BASE: &str = "https://api.novita.ai/openai";
/// Novita AI's default model.
pub const NOVITA_DEFAULT_MODEL: &str = "moonshotai/kimi-k2.5";

/// Novita AI LLM service — an [`OpenAiLlm`] pointed at the Novita AI base URL.
pub struct NovitaLlm {
    inner: OpenAiLlm,
}

impl NovitaLlm {
    /// Construct bound to `api_key`, using [`NOVITA_DEFAULT_MODEL`].
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_model(api_key, NOVITA_DEFAULT_MODEL)
    }

    /// Construct bound to `api_key` with an explicit `model`.
    pub fn with_model(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let inner = OpenAiLlmBuilder::new(api_key)
            .base_url(NOVITA_API_BASE)
            .model(model)
            .build();
        Self { inner }
    }
}

#[async_trait]
impl LlmService for NovitaLlm {
    fn name(&self) -> &str {
        "novita"
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
        assert_eq!(NovitaLlm::new("k").name(), "novita");
        assert_eq!(NovitaLlm::with_model("k", "custom").name(), "novita");
    }
}
