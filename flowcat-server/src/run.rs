// SPDX-License-Identifier: Apache-2.0
//
//! Call orchestration: assemble + run the flowcat pipeline for one call from the
//! resolved [`TopologyConfig`].
//!
//! - **Realtime:** build the realtime backend via the provider factory and run
//!   `build_s2s_task` (e.g. Gemini Live).
//! - **Cascaded:** build STT/LLM/TTS via the factory and run `build_cascaded_call`.
//!
//! Provider **API keys are resolved from the environment** here (never from the
//! config file or a per-call payload): `<PROVIDER>_API_KEY`, then
//! `FLOWCAT_<PROVIDER>_API_KEY`, plus the well-known `GOOGLE_API_KEY` for the
//! Gemini family.

use std::sync::Arc;

use serde_json::{Map, Value};

use flowcat_core::observer::FrameObserver;
use flowcat_core::pipeline::CascadedConfig;
use flowcat_core::{AgentBrain, FlowcatError, MediaTransport, SessionSource};
use flowcat_services::factory::{self, ProviderSpec};

use crate::config::TopologyConfig;

/// The ordered environment-variable names checked for `provider`'s API key.
///
/// Pure (no env access) so the precedence is unit-testable: the Gemini/Google
/// family shares `GOOGLE_API_KEY`; then `<PROVIDER>_API_KEY` and
/// `FLOWCAT_<PROVIDER>_API_KEY` (upper-cased, non-alphanumerics → `_`).
pub fn key_env_var_names(provider: &str) -> Vec<String> {
    let p = provider.to_ascii_lowercase();
    let upper: String = p
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect();
    let mut names = Vec::new();
    if matches!(p.as_str(), "gemini" | "google" | "google_realtime") {
        names.push("GOOGLE_API_KEY".to_string());
    }
    names.push(format!("{upper}_API_KEY"));
    names.push(format!("FLOWCAT_{upper}_API_KEY"));
    names
}

/// Resolve a provider's API key from the environment (empty string if unset).
pub fn key_from_env(provider: &str) -> String {
    for name in key_env_var_names(provider) {
        if let Ok(v) = std::env::var(&name) {
            if !v.trim().is_empty() {
                return v;
            }
        }
    }
    String::new()
}

/// Build a [`ProviderSpec`] from a provider name + model + options, injecting the
/// API key from the environment.
fn spec_with_env_key(provider: &str, model: &str, options: Map<String, Value>) -> ProviderSpec {
    ProviderSpec {
        provider: provider.to_string(),
        model: model.to_string(),
        api_key: key_from_env(provider),
        options,
    }
}

/// Fill a config [`ProviderSpec`]'s `api_key` from the environment if it's empty
/// (the config file never carries keys).
fn enrich(spec: &ProviderSpec) -> ProviderSpec {
    let mut s = spec.clone();
    if s.api_key.trim().is_empty() {
        s.api_key = key_from_env(&s.provider);
    }
    s
}

/// Assemble + run one call over `transport` per the resolved [`TopologyConfig`].
///
/// The realtime/cascaded providers are built from the topology (keys from env);
/// the pipeline runs to completion (returns when the call ends or errors).
pub async fn run_call<T, B, S>(
    transport: T,
    topology: &TopologyConfig,
    brain: B,
    session: S,
    run_id: i64,
    token: String,
    observers: Vec<Arc<dyn FrameObserver>>,
) -> Result<(), FlowcatError>
where
    T: MediaTransport + 'static,
    B: AgentBrain + 'static,
    S: SessionSource + 'static,
{
    match topology {
        TopologyConfig::Realtime {
            provider,
            model,
            options,
        } => {
            let spec = spec_with_env_key(provider, model, options.clone());
            let realtime = factory::realtime(&spec)?;
            flowcat_core::pipeline::s2s::build_s2s_task_with_observers(
                transport,
                realtime,
                brain,
                session,
                run_id,
                token,
                model.clone(),
                observers,
            )
            .await?
            .run()
            .await
        }
        TopologyConfig::Cascaded { stt, llm, tts } => {
            let (stt_svc, llm_svc, tts_svc) =
                factory::cascaded(&enrich(stt), &enrich(llm), &enrich(tts))?;
            let config = CascadedConfig {
                system_prompt: Some(brain.system_prompt()),
                ..Default::default()
            };
            flowcat_core::pipeline::build_cascaded_call_with_observers(
                transport, stt_svc, llm_svc, tts_svc, brain, session, run_id, token, config,
                observers,
            )
            .await?
            .run()
            .await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemini_family_checks_google_api_key_first() {
        let names = key_env_var_names("gemini");
        assert_eq!(names[0], "GOOGLE_API_KEY");
        assert!(names.contains(&"GEMINI_API_KEY".to_string()));
        assert!(names.contains(&"FLOWCAT_GEMINI_API_KEY".to_string()));

        // `google` and `google_realtime` share the alias too.
        assert_eq!(key_env_var_names("google")[0], "GOOGLE_API_KEY");
        assert_eq!(key_env_var_names("google_realtime")[0], "GOOGLE_API_KEY");
    }

    #[test]
    fn provider_name_is_upper_snake_cased() {
        let names = key_env_var_names("aws_bedrock");
        assert_eq!(
            names,
            vec!["AWS_BEDROCK_API_KEY", "FLOWCAT_AWS_BEDROCK_API_KEY"]
        );

        // Hyphens / odd chars normalize to `_`.
        assert_eq!(
            key_env_var_names("nvidia-nim"),
            vec!["NVIDIA_NIM_API_KEY", "FLOWCAT_NVIDIA_NIM_API_KEY"]
        );
    }

    #[test]
    fn non_gemini_provider_has_no_google_alias() {
        let names = key_env_var_names("deepgram");
        assert!(!names.iter().any(|n| n == "GOOGLE_API_KEY"));
        assert_eq!(names[0], "DEEPGRAM_API_KEY");
    }
}
