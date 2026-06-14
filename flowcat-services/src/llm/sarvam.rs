// SPDX-License-Identifier: Apache-2.0
//
//! **Sarvam** LLM — a thin **(W)rapper** over [`OpenAiLlm`](super::OpenAiLlm).
//!
//! Sarvam exposes an **OpenAI-compatible** chat-completions endpoint (verified
//! against pipecat `services/sarvam/llm.py`, which is
//! `class SarvamLLMService(OpenAILLMService)`), so this is the reference [`OpenAiLlm`]
//! pointed at [`SARVAM_API_BASE`] with the [`SARVAM_DEFAULT_MODEL`] default. Behind the
//! `llm-sarvam` feature (which enables `llm-openai`).

use async_trait::async_trait;
use futures::stream::BoxStream;

use flowcat_core::error::Result;
use flowcat_core::processor::frame::{Frame, LlmContext, StartParams};
use flowcat_core::service::{LlmService, Tool};

use super::{OpenAiLlm, OpenAiLlmBuilder};

/// Sarvam's OpenAI-compatible API base.
pub const SARVAM_API_BASE: &str = "https://api.sarvam.ai/v1";
/// Sarvam's default model.
pub const SARVAM_DEFAULT_MODEL: &str = "sarvam-30b";

/// Sarvam LLM service — an [`OpenAiLlm`] pointed at the Sarvam base URL.
pub struct SarvamLlm {
    inner: OpenAiLlm,
}

impl SarvamLlm {
    /// Construct bound to `api_key`, using [`SARVAM_DEFAULT_MODEL`].
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_model(api_key, SARVAM_DEFAULT_MODEL)
    }

    /// Construct bound to `api_key` with an explicit `model`.
    pub fn with_model(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let inner = OpenAiLlmBuilder::new(api_key)
            .base_url(SARVAM_API_BASE)
            .model(model)
            .build();
        Self { inner }
    }
}

#[async_trait]
impl LlmService for SarvamLlm {
    fn name(&self) -> &str {
        "sarvam"
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
        assert_eq!(SarvamLlm::new("k").name(), "sarvam");
        assert_eq!(SarvamLlm::with_model("k", "custom").name(), "sarvam");
    }
}
