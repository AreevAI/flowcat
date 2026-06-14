// SPDX-License-Identifier: Apache-2.0
//
//! **OpenRouter** LLM — a thin **(W)rapper** over [`OpenAiLlm`](super::OpenAiLlm).
//!
//! OpenRouter exposes an **OpenAI-compatible** chat-completions endpoint (verified
//! against pipecat `services/openrouter/llm.py`, which is
//! `class OpenRouterLLMService(OpenAILLMService)`), so this is the reference [`OpenAiLlm`]
//! pointed at [`OPENROUTER_API_BASE`] with the [`OPENROUTER_DEFAULT_MODEL`] default. Behind the
//! `llm-openrouter` feature (which enables `llm-openai`).

use async_trait::async_trait;
use futures::stream::BoxStream;

use flowcat_core::error::Result;
use flowcat_core::processor::frame::{Frame, LlmContext, StartParams};
use flowcat_core::service::{LlmService, Tool};

use super::{OpenAiLlm, OpenAiLlmBuilder};

/// OpenRouter's OpenAI-compatible API base.
pub const OPENROUTER_API_BASE: &str = "https://openrouter.ai/api/v1";
/// OpenRouter's default model.
pub const OPENROUTER_DEFAULT_MODEL: &str = "openai/gpt-4o-2024-11-20";

/// OpenRouter LLM service — an [`OpenAiLlm`] pointed at the OpenRouter base URL.
pub struct OpenRouterLlm {
    inner: OpenAiLlm,
}

impl OpenRouterLlm {
    /// Construct bound to `api_key`, using [`OPENROUTER_DEFAULT_MODEL`].
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_model(api_key, OPENROUTER_DEFAULT_MODEL)
    }

    /// Construct bound to `api_key` with an explicit `model`.
    pub fn with_model(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let inner = OpenAiLlmBuilder::new(api_key)
            .base_url(OPENROUTER_API_BASE)
            .model(model)
            .build();
        Self { inner }
    }
}

#[async_trait]
impl LlmService for OpenRouterLlm {
    fn name(&self) -> &str {
        "openrouter"
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
        assert_eq!(OpenRouterLlm::new("k").name(), "openrouter");
        assert_eq!(
            OpenRouterLlm::with_model("k", "custom").name(),
            "openrouter"
        );
    }

    /// Live smoke (requires `OPENROUTER_API_KEY`): stream a reply through OpenRouter
    /// across several underlying models (it routes to multiple providers), proving
    /// the OpenAI-compatible client works against the OpenRouter catalog.
    /// `OPENROUTER_API_KEY=… cargo test -p flowcat-services --features llm-openrouter -- --ignored openrouter_live`
    #[tokio::test]
    #[ignore = "requires OPENROUTER_API_KEY"]
    async fn openrouter_live_streams_across_models() {
        use futures::StreamExt;
        let key = std::env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY");
        for model in [
            "openai/gpt-4o-mini",
            "anthropic/claude-3.5-haiku",
            "meta-llama/llama-3.3-70b-instruct",
        ] {
            let mut llm = OpenRouterLlm::with_model(key.clone(), model);
            let ctx = LlmContext {
                messages: vec![
                    serde_json::json!({"role":"user","content":"Reply with one word: ok"}),
                ],
                tools: vec![],
            };
            let mut stream = llm.run_llm(&ctx).await.expect("run_llm");
            let mut text = String::new();
            while let Some(f) = stream.next().await {
                if let Frame::LlmText(t) = f {
                    text.push_str(&t);
                }
            }
            eprintln!("openrouter {model} => {text:?}");
            assert!(!text.trim().is_empty(), "no text from {model}");
        }
    }
}
