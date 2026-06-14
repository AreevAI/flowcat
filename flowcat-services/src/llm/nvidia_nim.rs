// SPDX-License-Identifier: Apache-2.0
//
//! **NVIDIA NIM** LLM — a thin **(W)rapper** over [`OpenAiLlm`](super::OpenAiLlm).
//!
//! NVIDIA NIM exposes an **OpenAI-compatible** chat-completions endpoint (verified
//! against pipecat `services/nvidia/llm.py`, which is
//! `class NimLLMService(OpenAILLMService)`), so this is the reference [`OpenAiLlm`]
//! pointed at [`NVIDIA_NIM_API_BASE`] with the [`NVIDIA_NIM_DEFAULT_MODEL`] default. Behind the
//! `llm-nvidia-nim` feature (which enables `llm-openai`).

use async_trait::async_trait;
use futures::stream::BoxStream;

use flowcat_core::error::Result;
use flowcat_core::processor::frame::{Frame, LlmContext, StartParams};
use flowcat_core::service::{LlmService, Tool};

use super::{OpenAiLlm, OpenAiLlmBuilder};

/// NVIDIA NIM's OpenAI-compatible API base.
pub const NVIDIA_NIM_API_BASE: &str = "https://integrate.api.nvidia.com/v1";
/// NVIDIA NIM's default model.
pub const NVIDIA_NIM_DEFAULT_MODEL: &str = "nvidia/nemotron-3-nano-30b-a3b";

/// NVIDIA NIM LLM service — an [`OpenAiLlm`] pointed at the NVIDIA NIM base URL.
pub struct NvidiaNimLlm {
    inner: OpenAiLlm,
}

impl NvidiaNimLlm {
    /// Construct bound to `api_key`, using [`NVIDIA_NIM_DEFAULT_MODEL`].
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_model(api_key, NVIDIA_NIM_DEFAULT_MODEL)
    }

    /// Construct bound to `api_key` with an explicit `model`.
    pub fn with_model(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let inner = OpenAiLlmBuilder::new(api_key)
            .base_url(NVIDIA_NIM_API_BASE)
            .model(model)
            .build();
        Self { inner }
    }
}

#[async_trait]
impl LlmService for NvidiaNimLlm {
    fn name(&self) -> &str {
        "nvidia_nim"
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
        assert_eq!(NvidiaNimLlm::new("k").name(), "nvidia_nim");
        assert_eq!(NvidiaNimLlm::with_model("k", "custom").name(), "nvidia_nim");
    }
}
