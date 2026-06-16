// SPDX-License-Identifier: Apache-2.0
//
//! Call orchestration: assemble + run the flowcat pipeline for one call from the
//! resolved [`TopologyConfig`].
//!
//! - **Realtime:** build the realtime backend via the provider factory and run
//!   `build_s2s_task` (e.g. Gemini Live).
//! - **Cascaded:** build STT/LLM/TTS via the factory and run `build_cascaded_call`.
//!
//! ## Who resolves provider keys
//!
//! [`run_call`] is the standalone-server default: it resolves provider **API keys
//! from the environment** (never the config file or a per-call payload) via
//! [`env_spec_resolver`] — `<PROVIDER>_API_KEY`, then `FLOWCAT_<PROVIDER>_API_KEY`,
//! plus the well-known `GOOGLE_API_KEY` for the Gemini family.
//!
//! A platform with its **own** secret store / per-call provider selection calls
//! [`run_call_with`] instead and supplies a [`SpecResolver`] — a
//! `Fn(&ProviderSpec) -> Result<ProviderSpec>` that fills each leg's key (and may
//! swap provider/model/options) from wherever it likes, with **no env reads** on
//! that path. `run_call` is just `run_call_with` + [`env_spec_resolver`].

use std::sync::Arc;

use flowcat_core::observer::FrameObserver;
use flowcat_core::pipeline::CascadedConfig;
use flowcat_core::{AgentBrain, FlowcatError, MediaTransport, SessionSource};
use flowcat_services::factory::{self, ProviderSpec};

use crate::config::TopologyConfig;

/// Resolves a call leg's final [`ProviderSpec`] (notably its `api_key`) from the
/// "bare" spec the [`TopologyConfig`] implies.
///
/// The standalone server uses [`env_spec_resolver`] (keys from the process env); an
/// embedder supplies its own — a secret-store lookup, a per-call config, etc. —
/// shared across calls behind an `Arc`. Returning a full `ProviderSpec` (not just a
/// key) lets a resolver also override provider/model/options per call.
pub type SpecResolver =
    Arc<dyn Fn(&ProviderSpec) -> Result<ProviderSpec, FlowcatError> + Send + Sync>;

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

/// The default [`SpecResolver`]: fill a leg's `api_key` from the environment if it's
/// empty (the config file never carries keys). A non-empty key is left untouched.
pub fn env_spec_resolver(spec: &ProviderSpec) -> Result<ProviderSpec, FlowcatError> {
    let mut s = spec.clone();
    if s.api_key.trim().is_empty() {
        s.api_key = key_from_env(&s.provider);
    }
    Ok(s)
}

/// The provider specs a call runs with, after resolution — what the factory builds
/// from. Realtime is a single spec; cascaded is the STT/LLM/TTS trio.
///
/// Deliberately not `Debug`: the specs carry resolved `api_key`s, so there is no
/// derive to accidentally log a secret through.
enum ResolvedProviders {
    Realtime(ProviderSpec),
    Cascaded {
        stt: ProviderSpec,
        llm: ProviderSpec,
        tts: ProviderSpec,
    },
}

/// Turn a [`TopologyConfig`] into the per-leg [`ProviderSpec`]s by running each leg
/// through `resolve`. Realtime starts from a **bare** spec (empty key) built from
/// the topology's provider/model/options; cascaded passes the config's specs (whose
/// keys are always empty — the file never carries them). Factored out so the
/// resolver wiring is unit-testable without a live provider connect.
fn resolve_providers<R>(
    topology: &TopologyConfig,
    resolve: R,
) -> Result<ResolvedProviders, FlowcatError>
where
    R: Fn(&ProviderSpec) -> Result<ProviderSpec, FlowcatError>,
{
    match topology {
        TopologyConfig::Realtime {
            provider,
            model,
            options,
        } => {
            let bare = ProviderSpec {
                provider: provider.clone(),
                model: model.clone(),
                api_key: String::new(),
                options: options.clone(),
            };
            Ok(ResolvedProviders::Realtime(resolve(&bare)?))
        }
        TopologyConfig::Cascaded { stt, llm, tts } => Ok(ResolvedProviders::Cascaded {
            stt: resolve(stt)?,
            llm: resolve(llm)?,
            tts: resolve(tts)?,
        }),
    }
}

/// Assemble + run one call over `transport` per the resolved [`TopologyConfig`],
/// resolving provider API keys **from the environment** ([`env_spec_resolver`]).
///
/// This is the standalone-server default; an embedder with its own secret store
/// uses [`run_call_with`] to supply a [`SpecResolver`] instead. The pipeline runs
/// to completion (returns when the call ends or errors).
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
    run_call_with(
        transport,
        topology,
        env_spec_resolver,
        brain,
        session,
        run_id,
        token,
        observers,
    )
    .await
}

/// [`run_call`] with an embedder-supplied [`SpecResolver`]: each provider leg's
/// final [`ProviderSpec`] comes from `resolve` (a secret store, per-call config, …)
/// rather than the environment, so a platform reuses flowcat-server's orchestration
/// without adopting its env convention. **No env reads happen on this path** unless
/// `resolve` makes them itself.
#[allow(clippy::too_many_arguments)]
pub async fn run_call_with<T, B, S, R>(
    transport: T,
    topology: &TopologyConfig,
    resolve: R,
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
    R: Fn(&ProviderSpec) -> Result<ProviderSpec, FlowcatError>,
{
    match resolve_providers(topology, resolve)? {
        ResolvedProviders::Realtime(spec) => {
            let model = spec.model.clone();
            let realtime = factory::realtime(&spec)?;
            flowcat_core::pipeline::s2s::build_s2s_task_with_observers(
                transport, realtime, brain, session, run_id, token, model, observers,
            )
            .await?
            .run()
            .await
        }
        ResolvedProviders::Cascaded { stt, llm, tts } => {
            let (stt_svc, llm_svc, tts_svc) = factory::cascaded(&stt, &llm, &tts)?;
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

    // --- Provider/spec resolution (the #21 embedder seam). ----------------------

    fn cfg_spec(provider: &str, model: &str) -> ProviderSpec {
        ProviderSpec {
            provider: provider.into(),
            model: model.into(),
            api_key: String::new(),
            options: Default::default(),
        }
    }

    #[test]
    fn env_spec_resolver_leaves_a_preset_key_untouched() {
        // A spec that already carries a key is passed through verbatim — the env is
        // never consulted (so an embedder that pre-fills keys gets no env reads).
        let spec = ProviderSpec {
            api_key: "preset-key".into(),
            ..cfg_spec("deepgram", "nova-3")
        };
        let out = env_spec_resolver(&spec).unwrap();
        assert_eq!(out.api_key, "preset-key");
        assert_eq!(out.provider, "deepgram");
    }

    #[test]
    fn resolve_providers_realtime_uses_the_supplied_resolver() {
        let topology = TopologyConfig::Realtime {
            provider: "gemini".into(),
            model: "gemini-2.0-flash".into(),
            options: Default::default(),
        };
        // A custom resolver standing in for an embedder secret store — no env reads.
        let resolved = resolve_providers(&topology, |spec: &ProviderSpec| {
            assert!(spec.api_key.is_empty(), "realtime starts from a bare spec");
            Ok(ProviderSpec {
                api_key: "from-store".into(),
                ..spec.clone()
            })
        })
        .unwrap();
        match resolved {
            ResolvedProviders::Realtime(spec) => {
                assert_eq!(spec.provider, "gemini");
                assert_eq!(spec.model, "gemini-2.0-flash");
                assert_eq!(spec.api_key, "from-store");
            }
            _ => panic!("expected realtime"),
        }
    }

    #[test]
    fn resolve_providers_cascaded_resolves_every_leg() {
        let topology = TopologyConfig::Cascaded {
            stt: cfg_spec("deepgram", "nova-3"),
            llm: cfg_spec("openai", "gpt-4o"),
            tts: cfg_spec("cartesia", "voice-xyz"),
        };
        // Resolver keys each leg from its provider name (embedder-owned, no env).
        let resolved = resolve_providers(&topology, |spec: &ProviderSpec| {
            Ok(ProviderSpec {
                api_key: format!("{}-key", spec.provider),
                ..spec.clone()
            })
        })
        .unwrap();
        match resolved {
            ResolvedProviders::Cascaded { stt, llm, tts } => {
                assert_eq!(stt.api_key, "deepgram-key");
                assert_eq!(llm.api_key, "openai-key");
                assert_eq!(tts.api_key, "cartesia-key");
            }
            _ => panic!("expected cascaded"),
        }
    }

    #[test]
    fn resolve_providers_propagates_a_resolver_error() {
        let topology = TopologyConfig::Cascaded {
            stt: cfg_spec("deepgram", "nova-3"),
            llm: cfg_spec("openai", "gpt-4o"),
            tts: cfg_spec("cartesia", "voice-xyz"),
        };
        // A resolver that fails for one leg (e.g. a missing secret) aborts cleanly.
        // (Matched out rather than `unwrap_err`'d so `ResolvedProviders` need not be
        // `Debug` — it carries resolved api keys.)
        let result = resolve_providers(&topology, |spec: &ProviderSpec| {
            if spec.provider == "openai" {
                Err(FlowcatError::Other("no secret for openai".into()))
            } else {
                Ok(spec.clone())
            }
        });
        let err = match result {
            Err(e) => e,
            Ok(_) => panic!("expected the resolver error to propagate"),
        };
        assert!(err.to_string().contains("no secret for openai"));
    }
}
