// SPDX-License-Identifier: Apache-2.0
//
//! **MiniMax** LLM — a thin **(W)rapper** over [`OpenAiLlm`](super::OpenAiLlm).
//!
//! MiniMax's international API exposes an **OpenAI-compatible** chat-completions
//! endpoint at [`MINIMAX_API_BASE`] (`{base}/chat/completions`, `Authorization: Bearer`),
//! so — like Cerebras/Groq/… — this is the reference [`OpenAiLlm`] pointed at that base
//! with the [`MINIMAX_DEFAULT_MODEL`] default. (A known-good MiniMax config:
//! `base_url = https://api.minimax.io/v1`, model `MiniMax-M2.7`.) Behind the
//! `llm-minimax` feature (which enables `llm-openai`).

use async_trait::async_trait;
use futures::stream::BoxStream;

use flowcat_core::error::Result;
use flowcat_core::processor::frame::{Frame, LlmContext, StartParams};
use flowcat_core::service::{LlmService, Tool};

use super::{OpenAiLlm, OpenAiLlmBuilder};

/// MiniMax's OpenAI-compatible API base (international endpoint).
pub const MINIMAX_API_BASE: &str = "https://api.minimax.io/v1";
/// MiniMax's default chat model.
pub const MINIMAX_DEFAULT_MODEL: &str = "MiniMax-M2.7";

/// MiniMax LLM service — an [`OpenAiLlm`] pointed at the MiniMax base URL.
pub struct MiniMaxLlm {
    inner: OpenAiLlm,
}

impl MiniMaxLlm {
    /// Construct bound to `api_key`, using [`MINIMAX_DEFAULT_MODEL`].
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_model(api_key, MINIMAX_DEFAULT_MODEL)
    }

    /// Construct bound to `api_key` with an explicit `model`.
    pub fn with_model(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let inner = OpenAiLlmBuilder::new(api_key)
            .base_url(MINIMAX_API_BASE)
            .model(model)
            .build();
        Self { inner }
    }
}

#[async_trait]
impl LlmService for MiniMaxLlm {
    fn name(&self) -> &str {
        "minimax"
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
        assert_eq!(MiniMaxLlm::new("k").name(), "minimax");
        assert_eq!(
            MiniMaxLlm::with_model("k", "MiniMax-Text-01").name(),
            "minimax"
        );
    }
}
