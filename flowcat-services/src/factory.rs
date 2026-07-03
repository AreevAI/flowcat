// SPDX-License-Identifier: Apache-2.0
//
//! Provider **factory**: construct a boxed service from a provider *name* + options,
//! instead of naming a concrete type at the call site. This is what lets a
//! config-driven host (e.g. `flowcat-server`) pick providers from a YAML/JSON file
//! — `factory::stt("deepgram", …)` rather than `DeepgramSttBuilder::new(…)`.
//!
//! Every provider arm is `#[cfg(feature = "…")]`-gated on the same flowcat-services
//! feature that pulls the connector, so the **default build (`features = []`) builds
//! none of them** and every name falls through to a clean "not built" error. Turning
//! a provider on is a one-line Cargo feature add — no code change here.
//!
//! ## [`ProviderSpec`]
//!
//! A host-agnostic description of one provider selection: `provider`, `model`
//! (for TTS this is the voice id), `api_key`, and an `options` map for the
//! provider-specific extras some connectors need (region, base_url, project,
//! location, endpoint, secret, group_id, url, join_url, language). The host fills
//! these from its own config/env; flowcat-services never reads the environment.
//!
//! **Security:** the `api_key` is whatever the host resolves (typically from its
//! own env/secret store). The factory never reads keys itself, so a per-call
//! payload can't smuggle one in unless the host puts it in the `ProviderSpec`.

use serde::Deserialize;
use serde_json::{Map, Value};

use flowcat_core::service::{LlmService, SttService, TtsService};
use flowcat_core::{FlowcatError, GeminiLive, RealtimeBackend};
// `ServiceRealtimeAdapter` is only referenced from the feature-gated realtime arms;
// import it conditionally so the default (no-feature) build has no unused import.
#[cfg(any(
    feature = "realtime-openai",
    feature = "realtime-grok",
    feature = "realtime-inworld",
    feature = "realtime-azure",
    feature = "realtime-ultravox"
))]
use flowcat_core::ServiceRealtimeAdapter;

/// A host-agnostic description of one provider selection.
///
/// `model` is the model id for STT/LLM/realtime; for **TTS it is the voice id**.
/// `options` carries provider-specific extras (see the module docs for the keys
/// each connector reads).
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ProviderSpec {
    /// Provider name, e.g. `"deepgram"`, `"elevenlabs"`, `"openai"`, `"gemini"`.
    pub provider: String,
    /// Model id (STT/LLM/realtime) or **voice id** (TTS). Empty → the connector's
    /// own default.
    #[serde(default)]
    pub model: String,
    /// API key / token resolved by the host (never read from the environment here).
    #[serde(default)]
    pub api_key: String,
    /// Provider-specific extras (region, base_url, project, location, endpoint,
    /// secret, group_id, url, join_url, language, …).
    #[serde(default)]
    pub options: Map<String, Value>,
}

impl ProviderSpec {
    /// A spec with just a provider name (everything else default).
    pub fn new(provider: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            ..Default::default()
        }
    }

    /// Set the model (or, for TTS, the voice id).
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the API key / token.
    pub fn with_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = api_key.into();
        self
    }

    /// Set one provider-specific option.
    pub fn with_option(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.options.insert(key.into(), Value::String(value.into()));
        self
    }

    /// Read a string option (empty string if absent), trimmed.
    fn opt(&self, key: &str) -> String {
        self.options
            .get(key)
            .and_then(Value::as_str)
            .map(str::trim)
            .unwrap_or_default()
            .to_string()
    }
}

/// "provider not built" error for an unknown / feature-disabled provider string.
fn not_built(kind: &str, provider: &str) -> FlowcatError {
    FlowcatError::Session(format!(
        "{kind} provider {provider:?} is not built (enable its flowcat-services feature, \
         or select a built provider)"
    ))
}

/// Require a non-empty provider name + key for a cascaded leg (an empty key is a
/// config error, not a silent fallback).
fn require_key<'a>(kind: &str, spec: &'a ProviderSpec) -> Result<&'a str, FlowcatError> {
    if spec.provider.trim().is_empty() {
        return Err(FlowcatError::Session(format!(
            "no {kind} provider configured"
        )));
    }
    if spec.api_key.trim().is_empty() {
        return Err(FlowcatError::Session(format!(
            "no API key configured for the {kind} provider {:?}",
            spec.provider
        )));
    }
    Ok(spec.api_key.as_str())
}

/// Build the three cascaded services (STT, LLM, TTS) in one call. Boxed trait
/// objects so they unify and hand straight to `build_cascaded_task`.
#[allow(clippy::type_complexity)]
pub fn cascaded(
    stt_spec: &ProviderSpec,
    llm_spec: &ProviderSpec,
    tts_spec: &ProviderSpec,
) -> Result<
    (
        Box<dyn SttService>,
        Box<dyn LlmService>,
        Box<dyn TtsService>,
    ),
    FlowcatError,
> {
    Ok((stt(stt_spec)?, llm(llm_spec)?, tts(tts_spec)?))
}

/// Build the STT service for `spec.provider`. Most clients take `new(api_key)`;
/// a few read extras from `spec.options` (Azure `region`, Google `project`/
/// `location`, Speaches `base_url`).
#[allow(unused_variables)] // `key` is unused when no stt-* feature is enabled
pub fn stt(spec: &ProviderSpec) -> Result<Box<dyn SttService>, FlowcatError> {
    let key = require_key("stt", spec)?;
    match spec.provider.to_ascii_lowercase().as_str() {
        #[cfg(feature = "stt-deepgram")]
        "deepgram" => {
            let mut b = crate::stt::DeepgramSttBuilder::new(key);
            if !spec.model.is_empty() {
                b = b.model(&spec.model);
            }
            Ok(Box::new(b.build()))
        }
        #[cfg(feature = "stt-assemblyai")]
        "assemblyai" => Ok(Box::new(crate::stt::AssemblyAiStt::new(key))),
        #[cfg(feature = "stt-gladia")]
        "gladia" => Ok(Box::new(crate::stt::GladiaStt::new(key))),
        #[cfg(feature = "stt-soniox")]
        "soniox" => Ok(Box::new(crate::stt::SonioxStt::new(key))),
        #[cfg(feature = "stt-speechmatics")]
        "speechmatics" => Ok(Box::new(crate::stt::SpeechmaticsStt::new(key))),
        #[cfg(feature = "stt-cartesia")]
        "cartesia" => Ok(Box::new(crate::stt::CartesiaStt::new(key))),
        #[cfg(feature = "stt-elevenlabs")]
        "elevenlabs" => {
            let c = crate::stt::ElevenLabsStt::new(key);
            Ok(Box::new(if spec.model.is_empty() {
                c
            } else {
                c.model(&spec.model)
            }))
        }
        #[cfg(feature = "stt-sarvam")]
        "sarvam" => {
            // Honor the pinned model (`saarika:v2.5` vs `saaras:v2.5`) and the
            // `language` option (BCP-47, e.g. `hi-IN`; absent → auto-detect).
            let mut c = crate::stt::SarvamStt::new(key);
            if !spec.model.is_empty() {
                c = c.model(&spec.model);
            }
            let language = spec.opt("language");
            if !language.is_empty() {
                c = c.language(language);
            }
            Ok(Box::new(c))
        }
        #[cfg(feature = "stt-openai")]
        "openai" | "whisper" => Ok(Box::new(crate::stt::OpenAiStt::new(key))),
        #[cfg(feature = "stt-groq")]
        "groq" => Ok(Box::new(crate::stt::GroqStt::new(key))),
        #[cfg(feature = "stt-fal")]
        "fal" => Ok(Box::new(crate::stt::FalStt::new(key))),
        #[cfg(feature = "stt-gradium")]
        "gradium" => Ok(Box::new(crate::stt::GradiumStt::new(key))),
        #[cfg(feature = "stt-mistral")]
        "mistral" => Ok(Box::new(crate::stt::MistralStt::new(key))),
        #[cfg(feature = "stt-speaches")]
        "speaches" => {
            let base = spec.opt("base_url");
            Ok(Box::new(if base.is_empty() {
                crate::stt::SpeachesStt::new(key)
            } else {
                crate::stt::SpeachesStt::with_base_url(key, base)
            }))
        }
        #[cfg(feature = "stt-xai")]
        "xai" => Ok(Box::new(crate::stt::XaiStt::new(key))),
        #[cfg(feature = "stt-azure")]
        "azure" | "azure_speech" => {
            Ok(Box::new(crate::stt::AzureStt::new(key, spec.opt("region"))))
        }
        #[cfg(feature = "stt-google")]
        "google" => {
            let project = spec.opt("project");
            if project.is_empty() {
                return Err(FlowcatError::Session(
                    "google stt requires the `project` option (your GCP project id)".to_string(),
                ));
            }
            let location = {
                let l = spec.opt("location");
                if l.is_empty() {
                    "global".to_string()
                } else {
                    l
                }
            };
            let recognizer = format!("projects/{project}/locations/{location}/recognizers/_");
            Ok(Box::new(crate::stt::GoogleStt::new(key, recognizer)))
        }
        #[cfg(feature = "stt-nvidia")]
        "nvidia" => {
            // NVIDIA Riva ASR over gRPC. `key` = the NVCF bearer / NIM key; the
            // `function_id` option routes to a hosted ASR model (NVCF only — empty
            // for a self-hosted Riva, which sets its own `host`). `model` is optional.
            let mut c = crate::stt::NvidiaStt::new(key);
            let function_id = spec.opt("function_id");
            if !function_id.is_empty() {
                c = c.function_id(function_id);
            }
            let host = spec.opt("host");
            if !host.is_empty() {
                c = c.endpoint(host);
            }
            if !spec.model.is_empty() {
                c = c.model(&spec.model);
            }
            Ok(Box::new(c))
        }
        #[cfg(feature = "stt-whisper-local")]
        "whisper_local" => {
            // Local whisper.cpp (in-process, no network). `model_path` is required —
            // the path to a ggml model file (the key is an unused keyless placeholder).
            let path = spec.opt("model_path");
            if path.is_empty() {
                return Err(FlowcatError::Session(
                    "whisper_local stt requires the `model_path` option (path to a ggml model)"
                        .to_string(),
                ));
            }
            Ok(Box::new(crate::stt::WhisperLocalStt::new(path)))
        }
        other => Err(not_built("stt", other)),
    }
}

/// Build the TTS service for `spec.provider`. Clients take `new(api_key, voice)`;
/// the voice is `spec.model` (empty → the provider's default voice). A few read
/// extras from `spec.options` (MiniMax `group_id`, Azure `region`, Speaches
/// `base_url`).
#[allow(unused_variables)] // `key`/`voice` are unused when no tts-* feature is enabled
pub fn tts(spec: &ProviderSpec) -> Result<Box<dyn TtsService>, FlowcatError> {
    let key = require_key("tts", spec)?;
    let voice = spec.model.clone();
    match spec.provider.to_ascii_lowercase().as_str() {
        #[cfg(feature = "tts-cartesia")]
        "cartesia" => Ok(Box::new(crate::tts::CartesiaTts::new(key, voice))),
        #[cfg(feature = "tts-elevenlabs")]
        "elevenlabs" => Ok(Box::new(crate::tts::ElevenLabsTts::new(key, voice))),
        #[cfg(feature = "tts-deepgram")]
        "deepgram" => Ok(Box::new(crate::tts::DeepgramTts::new(key, voice))),
        #[cfg(feature = "tts-rime")]
        "rime" => Ok(Box::new(crate::tts::RimeTts::new(key, voice))),
        #[cfg(feature = "tts-openai")]
        "openai" => Ok(Box::new(crate::tts::OpenAiTts::new(key, voice))),
        #[cfg(feature = "tts-sarvam")]
        "sarvam" => {
            // `spec.model` is the voice (speaker); the actual Bulbul model and
            // the `target_language_code` ride as options (`model`, `language`) —
            // Sarvam rejects synthesis without a real language code, so the
            // caller should pass one (default `en-IN`).
            let mut c = crate::tts::SarvamTts::new(key, voice);
            let model = spec.opt("model");
            if !model.is_empty() {
                c = c.model(model);
            }
            let language = spec.opt("language");
            if !language.is_empty() {
                c = c.language(language);
            }
            Ok(Box::new(c))
        }
        #[cfg(feature = "tts-hume")]
        "hume" => Ok(Box::new(crate::tts::HumeTts::new(key, voice))),
        #[cfg(feature = "tts-inworld")]
        "inworld" => Ok(Box::new(crate::tts::InworldTts::new(key, voice))),
        #[cfg(feature = "tts-asyncai")]
        "asyncai" => Ok(Box::new(crate::tts::AsyncAiTts::new(key, voice))),
        #[cfg(feature = "tts-camb")]
        "camb" => Ok(Box::new(crate::tts::CambTts::new(key, voice))),
        #[cfg(feature = "tts-fish")]
        "fish" => Ok(Box::new(crate::tts::FishTts::new(key, voice))),
        #[cfg(feature = "tts-gradium")]
        "gradium" => Ok(Box::new(crate::tts::GradiumTts::new(key, voice))),
        #[cfg(feature = "tts-groq")]
        "groq" => Ok(Box::new(crate::tts::GroqTts::new(key, voice))),
        #[cfg(feature = "tts-kokoro")]
        "kokoro" => Ok(Box::new(crate::tts::KokoroTts::new(key, voice))),
        #[cfg(feature = "tts-lmnt")]
        "lmnt" => Ok(Box::new(crate::tts::LmntTts::new(key, voice))),
        #[cfg(feature = "tts-mistral")]
        "mistral" => Ok(Box::new(crate::tts::MistralTts::new(key, voice))),
        #[cfg(feature = "tts-neuphonic")]
        "neuphonic" => Ok(Box::new(crate::tts::NeuphonicTts::new(key, voice))),
        #[cfg(feature = "tts-piper")]
        "piper" => Ok(Box::new(crate::tts::PiperTts::new(key, voice))),
        #[cfg(feature = "tts-resemble")]
        "resemble" => Ok(Box::new(crate::tts::ResembleTts::new(key, voice))),
        #[cfg(feature = "tts-smallest")]
        "smallest" => Ok(Box::new(crate::tts::SmallestTts::new(key, voice))),
        #[cfg(feature = "tts-soniox")]
        "soniox" => Ok(Box::new(crate::tts::SonioxTts::new(key, voice))),
        #[cfg(feature = "tts-speechmatics")]
        "speechmatics" => Ok(Box::new(crate::tts::SpeechmaticsTts::new(key, voice))),
        #[cfg(feature = "tts-xai")]
        "xai" => Ok(Box::new(crate::tts::XaiTts::new(key, voice))),
        #[cfg(feature = "tts-xtts")]
        "xtts" => Ok(Box::new(crate::tts::XttsTts::new(key, voice))),
        #[cfg(feature = "tts-minimax")]
        "minimax" => Ok(Box::new(crate::tts::MiniMaxTts::new(
            key,
            spec.opt("group_id"),
            voice,
        ))),
        #[cfg(feature = "tts-google")]
        "google" => Ok(Box::new(crate::tts::GoogleTts::new(key, voice))),
        #[cfg(feature = "tts-azure")]
        "azure" | "azure_speech" => Ok(Box::new(crate::tts::AzureTts::new(
            key,
            spec.opt("region"),
            voice,
        ))),
        #[cfg(feature = "tts-speaches")]
        "speaches" => Ok(Box::new(crate::tts::SpeachesTts::new(
            key,
            spec.opt("base_url"),
            voice,
        ))),
        other => Err(not_built("tts", other)),
    }
}

/// Build an OpenAI-compatible LLM exposing `new(key)` + `with_model(key, model)`,
/// applying the model override only when non-empty.
#[allow(unused_macros)] // unused when no OpenAI-compatible llm-* feature is enabled
macro_rules! oai_compat_llm {
    ($ty:ty, $key:expr, $model:expr) => {
        if $model.is_empty() {
            <$ty>::new($key)
        } else {
            <$ty>::with_model($key, $model)
        }
    };
}

/// Build the LLM service for `spec.provider`. Most clients accept `new(api_key)`
/// plus a `.model(m)` / `with_model(key, m)` override; a few read extras from
/// `spec.options` (Vertex `project`/`location`, Azure `endpoint`, Bedrock
/// `secret`/`region`).
#[allow(unused_variables)] // `key`/`model` are unused when no llm-* feature is enabled
pub fn llm(spec: &ProviderSpec) -> Result<Box<dyn LlmService>, FlowcatError> {
    let key = require_key("llm", spec)?;
    let model = spec.model.clone();
    match spec.provider.to_ascii_lowercase().as_str() {
        #[cfg(feature = "llm-openai")]
        "openai" => {
            let mut b = crate::llm::OpenAiLlmBuilder::new(key);
            if !model.is_empty() {
                b = b.model(model);
            }
            Ok(Box::new(b.build()))
        }
        #[cfg(feature = "llm-anthropic")]
        "anthropic" => {
            let c = crate::llm::AnthropicLlm::new(key);
            Ok(Box::new(if model.is_empty() { c } else { c.model(model) }))
        }
        #[cfg(feature = "llm-google")]
        "google" | "gemini" => {
            let c = crate::llm::GoogleLlm::new(key);
            Ok(Box::new(if model.is_empty() { c } else { c.model(model) }))
        }
        #[cfg(feature = "llm-google-vertex")]
        "google_vertex" | "vertex" => {
            let c =
                crate::llm::GoogleVertexLlm::new(key, spec.opt("project"), spec.opt("location"));
            Ok(Box::new(if model.is_empty() { c } else { c.model(model) }))
        }
        #[cfg(feature = "llm-groq")]
        "groq" => Ok(Box::new(oai_compat_llm!(crate::llm::GroqLlm, key, model))),
        #[cfg(feature = "llm-together")]
        "together" => Ok(Box::new(oai_compat_llm!(
            crate::llm::TogetherLlm,
            key,
            model
        ))),
        #[cfg(feature = "llm-openrouter")]
        "openrouter" => Ok(Box::new(oai_compat_llm!(
            crate::llm::OpenRouterLlm,
            key,
            model
        ))),
        #[cfg(feature = "llm-deepseek")]
        "deepseek" => Ok(Box::new(oai_compat_llm!(
            crate::llm::DeepSeekLlm,
            key,
            model
        ))),
        #[cfg(feature = "llm-mistral")]
        "mistral" => Ok(Box::new(oai_compat_llm!(
            crate::llm::MistralLlm,
            key,
            model
        ))),
        #[cfg(feature = "llm-cerebras")]
        "cerebras" => Ok(Box::new(oai_compat_llm!(
            crate::llm::CerebrasLlm,
            key,
            model
        ))),
        #[cfg(feature = "llm-minimax")]
        "minimax" => Ok(Box::new(oai_compat_llm!(
            crate::llm::MiniMaxLlm,
            key,
            model
        ))),
        #[cfg(feature = "llm-fireworks")]
        "fireworks" => Ok(Box::new(oai_compat_llm!(
            crate::llm::FireworksLlm,
            key,
            model
        ))),
        #[cfg(feature = "llm-nebius")]
        "nebius" => Ok(Box::new(oai_compat_llm!(crate::llm::NebiusLlm, key, model))),
        #[cfg(feature = "llm-novita")]
        "novita" => Ok(Box::new(oai_compat_llm!(crate::llm::NovitaLlm, key, model))),
        #[cfg(feature = "llm-perplexity")]
        "perplexity" => Ok(Box::new(oai_compat_llm!(
            crate::llm::PerplexityLlm,
            key,
            model
        ))),
        #[cfg(feature = "llm-qwen")]
        "qwen" => Ok(Box::new(oai_compat_llm!(crate::llm::QwenLlm, key, model))),
        #[cfg(feature = "llm-sambanova")]
        "sambanova" => Ok(Box::new(oai_compat_llm!(
            crate::llm::SambaNovaLlm,
            key,
            model
        ))),
        #[cfg(feature = "llm-nvidia-nim")]
        "nvidia_nim" | "nvidia-nim" => Ok(Box::new(oai_compat_llm!(
            crate::llm::NvidiaNimLlm,
            key,
            model
        ))),
        #[cfg(feature = "llm-grok")]
        "grok" => Ok(Box::new(oai_compat_llm!(crate::llm::GrokLlm, key, model))),
        #[cfg(feature = "llm-ollama")]
        "ollama" => Ok(Box::new(oai_compat_llm!(crate::llm::OllamaLlm, key, model))),
        #[cfg(feature = "llm-sarvam")]
        "sarvam" => Ok(Box::new(oai_compat_llm!(crate::llm::SarvamLlm, key, model))),
        #[cfg(feature = "llm-speaches")]
        "speaches" => Ok(Box::new(oai_compat_llm!(
            crate::llm::SpeachesLlm,
            key,
            model
        ))),
        #[cfg(feature = "llm-openai-responses")]
        "openai_responses" | "openai-responses" => {
            let c = crate::llm::OpenAiResponsesLlm::new(key);
            Ok(Box::new(if model.is_empty() { c } else { c.model(model) }))
        }
        #[cfg(feature = "llm-azure")]
        "azure" => Ok(Box::new(crate::llm::AzureLlm::new(
            key,
            spec.opt("endpoint"),
            model,
        ))),
        #[cfg(feature = "llm-aws-bedrock")]
        "aws_bedrock" | "bedrock" => Ok(Box::new(crate::llm::AwsBedrockLlm::new(
            key,
            spec.opt("secret"),
            spec.opt("region"),
            model,
        ))),
        other => Err(not_built("llm", other)),
    }
}

/// Build the realtime (speech-to-speech) backend for `spec.provider`.
///
/// `gemini` is flowcat-core's [`GeminiLive`] (implements the pipeline traits
/// directly); every flowcat-services realtime connector is wrapped in
/// [`ServiceRealtimeAdapter`], which presents it as the pipeline's
/// `RealtimeLlm + RealtimeKickoff`. Both coerce to one `Box<dyn RealtimeBackend>`.
///
/// Option keys: `language` (input-transcription hint, OpenAI-family), `url`
/// (Azure deployment WSS URL), `join_url` (Ultravox per-call URL), `project` /
/// `location` (Google Vertex).
#[allow(unused_variables)] // `language` is unused when no realtime-openai feature is enabled
pub fn realtime(spec: &ProviderSpec) -> Result<Box<dyn RealtimeBackend>, FlowcatError> {
    let key = spec.api_key.as_str();
    let language = {
        let l = spec.opt("language");
        if l.is_empty() {
            None
        } else {
            Some(l)
        }
    };
    match spec.provider.to_ascii_lowercase().as_str() {
        "gemini" | "google" | "google_realtime" => {
            // Validate upfront (like the feature-gated realtime providers below) so a
            // missing key is a clean error here, not a cryptic failure at connect time.
            if key.trim().is_empty() {
                return Err(FlowcatError::Realtime(
                    "gemini realtime selected but no API key configured (set GOOGLE_API_KEY)"
                        .to_string(),
                ));
            }
            Ok(Box::new(GeminiLive::new(key.to_string())))
        }
        #[cfg(feature = "realtime-openai")]
        "openai" | "openai_realtime" => {
            let key = require_realtime_key("openai", spec)?;
            Ok(Box::new(ServiceRealtimeAdapter::new(
                crate::realtime::OpenAiRealtime::new(key).with_input_language(language),
            )))
        }
        #[cfg(feature = "realtime-grok")]
        "grok" | "grok_realtime" => {
            let key = require_realtime_key("grok", spec)?;
            Ok(Box::new(ServiceRealtimeAdapter::new(
                crate::realtime::GrokRealtime::new(key),
            )))
        }
        #[cfg(feature = "realtime-inworld")]
        "inworld" | "inworld_realtime" => {
            let key = require_realtime_key("inworld", spec)?;
            Ok(Box::new(ServiceRealtimeAdapter::new(
                crate::realtime::InworldRealtime::new(key),
            )))
        }
        #[cfg(feature = "realtime-azure")]
        "azure" | "azure_realtime" => {
            let key = require_realtime_key("azure", spec)?;
            let url = spec.opt("url");
            if url.is_empty() {
                return Err(FlowcatError::Realtime(
                    "azure realtime selected but no endpoint URL configured \
                     (set option `url` to the full deployment WSS URL)"
                        .to_string(),
                ));
            }
            Ok(Box::new(ServiceRealtimeAdapter::new(
                crate::realtime::AzureRealtime::new(key, url),
            )))
        }
        #[cfg(feature = "realtime-ultravox")]
        "ultravox" | "ultravox_realtime" => {
            let join_url = spec.opt("join_url");
            if join_url.is_empty() {
                return Err(FlowcatError::Realtime(
                    "ultravox realtime selected but no join URL configured \
                     (set option `join_url`; mint one per call via the Ultravox REST API)"
                        .to_string(),
                ));
            }
            Ok(Box::new(ServiceRealtimeAdapter::new(
                crate::realtime::UltravoxRealtime::new(join_url),
            )))
        }
        "google_vertex" | "google_vertex_realtime" | "vertex_realtime" => {
            if key.trim().is_empty() {
                return Err(FlowcatError::Realtime(
                    "google_vertex realtime selected but no access token configured \
                     (set api_key to an OAuth2 access token)"
                        .to_string(),
                ));
            }
            Ok(Box::new(GeminiLive::new_vertex(
                key.to_string(),
                spec.opt("project"),
                spec.opt("location"),
            )))
        }
        other => Err(FlowcatError::Realtime(format!(
            "realtime provider {other:?} is not built (gemini, openai, grok, inworld, azure, \
             ultravox, google_vertex are wired; enable the matching flowcat-services realtime \
             feature to add more)"
        ))),
    }
}

/// Require a non-empty key for a realtime provider; clean error naming the field.
#[cfg(any(
    feature = "realtime-openai",
    feature = "realtime-grok",
    feature = "realtime-inworld",
    feature = "realtime-azure"
))]
fn require_realtime_key(provider: &str, spec: &ProviderSpec) -> Result<String, FlowcatError> {
    if spec.api_key.trim().is_empty() {
        return Err(FlowcatError::Realtime(format!(
            "{provider} realtime selected but no key configured (set api_key)"
        )));
    }
    Ok(spec.api_key.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(provider: &str, key: &str) -> ProviderSpec {
        ProviderSpec::new(provider).with_key(key)
    }

    fn cascaded_err(stt_p: &str, tts_p: &str, llm_p: &str) -> String {
        match cascaded(&spec(stt_p, "k"), &spec(llm_p, "k"), &spec(tts_p, "k")) {
            Ok(_) => panic!("expected the factory to error"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn unknown_provider_is_a_clean_not_built_error() {
        let err = cascaded_err("nope-stt", "cartesia", "openai");
        assert!(
            err.contains("not built"),
            "expected 'not built', got: {err}"
        );
        assert!(err.contains("nope-stt"));
    }

    #[test]
    fn empty_key_is_a_config_error_not_a_panic() {
        let err = match stt(&ProviderSpec::new("deepgram")) {
            Ok(_) => panic!("expected a key error"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.to_lowercase().contains("key"),
            "expected a key error, got: {err}"
        );
    }

    #[test]
    fn empty_provider_is_a_config_error() {
        let err = match llm(&ProviderSpec::default()) {
            Ok(_) => panic!("expected a provider error"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("no llm provider configured"), "got: {err}");
    }

    #[test]
    #[cfg(all(feature = "stt-sarvam", feature = "tts-sarvam"))]
    fn sarvam_builds_with_and_without_model_language_options() {
        // Bare spec (host defaults: saarika:v2.5 / bulbul:v2 / en-IN).
        assert!(stt(&spec("sarvam", "k")).is_ok());
        assert!(tts(&spec("sarvam", "k")).is_ok());
        // Pinned model + language options (the vaais Models-page path).
        let s = spec("sarvam", "k")
            .with_model("saaras:v2.5")
            .with_option("language", "hi-IN");
        assert!(stt(&s).is_ok());
        let t = spec("sarvam", "k")
            .with_model("anushka") // TTS spec.model = the voice id
            .with_option("model", "bulbul:v3")
            .with_option("language", "hi-IN");
        assert!(tts(&t).is_ok());
    }

    #[test]
    fn realtime_gemini_is_key_gated_and_unknown_is_not_built() {
        // With a key, gemini builds.
        assert!(realtime(&spec("gemini", "k")).is_ok());
        assert!(realtime(&spec("GEMINI", "k")).is_ok());
        // Without a key it is a clean error (not a cryptic connect-time failure).
        let err = match realtime(&ProviderSpec::new("gemini")) {
            Ok(_) => panic!("gemini realtime with no key should error"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("no API key configured"), "got: {err}");
        // An unknown provider is a clean "not built" error naming the provider.
        let err = match realtime(&spec("banana", "k")) {
            Ok(_) => panic!("unknown realtime should not be built"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("not built"), "got: {err}");
        assert!(err.contains("banana"), "got: {err}");
    }

    #[cfg(feature = "stt-google")]
    #[test]
    fn google_stt_requires_a_project_option() {
        // No `project` → a clean config error, not a malformed recognizer path.
        let err = match stt(&spec("google", "token")) {
            Ok(_) => panic!("google stt with no project should error"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("project"), "got: {err}");
        // With a project it builds.
        assert!(stt(&spec("google", "token").with_option("project", "my-proj")).is_ok());
    }

    // ── feature-gated provider coverage (run under the connector feature set) ──

    #[cfg(all(
        feature = "stt-deepgram",
        feature = "tts-cartesia",
        feature = "llm-openai"
    ))]
    #[test]
    fn factory_builds_the_reference_trio() {
        assert!(cascaded(
            &spec("deepgram", "k"),
            &spec("openai", "k"),
            &spec("cartesia", "k")
        )
        .is_ok());
    }

    #[cfg(feature = "stt-elevenlabs")]
    #[test]
    fn elevenlabs_stt_builds_with_and_without_model() {
        assert!(stt(&spec("elevenlabs", "k")).is_ok());
        assert!(stt(&spec("elevenlabs", "k").with_model("scribe_v1")).is_ok());
    }

    #[cfg(feature = "tts-minimax")]
    #[test]
    fn minimax_tts_builds_with_group_id_option() {
        assert!(tts(&spec("minimax", "k")
            .with_option("group_id", "grp-1")
            .with_model("voice-1"))
        .is_ok());
    }

    #[cfg(feature = "stt-azure")]
    #[test]
    fn azure_stt_reads_region_option() {
        assert!(stt(&spec("azure", "k").with_option("region", "eastus")).is_ok());
    }

    #[cfg(feature = "realtime-openai")]
    #[test]
    fn openai_realtime_is_key_gated() {
        // Wired, but key-gated: no key → a "no key" error, not "not built".
        let err = match realtime(&ProviderSpec::new("openai")) {
            Ok(_) => panic!("openai realtime should be key-gated"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("no key configured"), "got: {err}");
    }
}
