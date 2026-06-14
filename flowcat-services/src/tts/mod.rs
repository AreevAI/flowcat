// SPDX-License-Identifier: Apache-2.0
//
//! Streaming text-to-speech providers.
//!
//! Impls of [`TtsService`](flowcat_core::service::TtsService). Cartesia is the
//! reference streaming-WebSocket impl behind `tts-cartesia`; the remaining providers
//! fan out from there. Each provider is `dep:`-feature-gated so the default build
//! pulls no TTS dep.
//!
//! ## Provider fan-out homes (PROVIDERS.md §3)
//!
//! TTS is the least wrapper-friendly category — almost every vendor has a bespoke
//! request body, so nearly all are **(D)istinct** (the Cartesia WS impl + a future
//! `HttpTtsClient` helper are the templates). Only groq/xai are **(W)** over the
//! OpenAI-TTS-HTTP client ([`OpenAiTts`](super::tts::OpenAiTts)). Every provider has
//! a stub module below so the fan-out fills one body and never edits this list.

#[cfg(feature = "tts-cartesia")]
pub mod cartesia;
#[cfg(feature = "tts-cartesia")]
pub use cartesia::{CartesiaTts, CartesiaTtsBuilder};

// --- (D)istinct streaming-WebSocket TTS (PROVIDERS.md §3) ---
#[cfg(feature = "tts-elevenlabs")]
pub mod elevenlabs;
#[cfg(feature = "tts-elevenlabs")]
pub use elevenlabs::ElevenLabsTts;
#[cfg(feature = "tts-deepgram")]
pub mod deepgram;
#[cfg(feature = "tts-deepgram")]
pub use deepgram::DeepgramTts;
#[cfg(feature = "tts-rime")]
pub mod rime;
#[cfg(feature = "tts-rime")]
pub use rime::RimeTts;
#[cfg(feature = "tts-asyncai")]
pub mod asyncai;
#[cfg(feature = "tts-asyncai")]
pub use asyncai::AsyncAiTts;
#[cfg(feature = "tts-gradium")]
pub mod gradium;
#[cfg(feature = "tts-gradium")]
pub use gradium::GradiumTts;
#[cfg(feature = "tts-soniox")]
pub mod soniox;
#[cfg(feature = "tts-soniox")]
pub use soniox::SonioxTts;
#[cfg(feature = "tts-resemble")]
pub mod resemble;
#[cfg(feature = "tts-resemble")]
pub use resemble::ResembleTts;

// --- OpenAI-TTS-HTTP family: openai is the (D) client; groq/xai are (W) ---
#[cfg(feature = "tts-openai")]
pub mod openai;
#[cfg(feature = "tts-openai")]
pub use openai::OpenAiTts;
#[cfg(feature = "tts-groq")]
pub mod groq;
#[cfg(feature = "tts-groq")]
pub use groq::GroqTts;
#[cfg(feature = "tts-xai")]
pub mod xai;
#[cfg(feature = "tts-xai")]
pub use xai::XaiTts;
#[cfg(feature = "tts-speaches")]
pub mod speaches;
#[cfg(feature = "tts-speaches")]
pub use speaches::SpeachesTts;

// --- (D)istinct HTTP cloud TTS (PROVIDERS.md §3) ---
#[cfg(feature = "tts-azure")]
pub mod azure;
#[cfg(feature = "tts-azure")]
pub use azure::AzureTts;
#[cfg(feature = "tts-sarvam")]
pub mod sarvam;
#[cfg(feature = "tts-sarvam")]
pub use sarvam::SarvamTts;
#[cfg(feature = "tts-mistral")]
pub mod mistral;
#[cfg(feature = "tts-mistral")]
pub use mistral::MistralTts;
#[cfg(feature = "tts-hume")]
pub mod hume;
#[cfg(feature = "tts-hume")]
pub use hume::HumeTts;
#[cfg(feature = "tts-inworld")]
pub mod inworld;
#[cfg(feature = "tts-inworld")]
pub use inworld::InworldTts;
#[cfg(feature = "tts-minimax")]
pub mod minimax;
#[cfg(feature = "tts-minimax")]
pub use minimax::MiniMaxTts;
#[cfg(feature = "tts-camb")]
pub mod camb;
#[cfg(feature = "tts-camb")]
pub use camb::CambTts;
#[cfg(feature = "tts-speechmatics")]
pub mod speechmatics;
#[cfg(feature = "tts-speechmatics")]
pub use speechmatics::SpeechmaticsTts;

// --- (D)istinct interruptible-HTTP TTS ---
#[cfg(feature = "tts-fish")]
pub mod fish;
#[cfg(feature = "tts-fish")]
pub use fish::FishTts;
#[cfg(feature = "tts-lmnt")]
pub mod lmnt;
#[cfg(feature = "tts-lmnt")]
pub use lmnt::LmntTts;
#[cfg(feature = "tts-neuphonic")]
pub mod neuphonic;
#[cfg(feature = "tts-neuphonic")]
pub use neuphonic::NeuphonicTts;
#[cfg(feature = "tts-smallest")]
pub mod smallest;
#[cfg(feature = "tts-smallest")]
pub use smallest::SmallestTts;

// --- Local-model TTS ---
#[cfg(feature = "tts-kokoro")]
pub mod kokoro;
#[cfg(feature = "tts-kokoro")]
pub use kokoro::KokoroTts;
#[cfg(feature = "tts-piper")]
pub mod piper;
#[cfg(feature = "tts-piper")]
pub use piper::PiperTts;
#[cfg(feature = "tts-xtts")]
pub mod xtts;
#[cfg(feature = "tts-xtts")]
pub use xtts::XttsTts;

// --- gRPC + AWS TTS (toolchain-heavy — PROVIDERS.md §5) ---
#[cfg(feature = "tts-google")]
pub mod google;
#[cfg(feature = "tts-google")]
pub use google::GoogleTts;
#[cfg(feature = "tts-nvidia")]
pub mod nvidia;
#[cfg(feature = "tts-nvidia")]
pub use nvidia::NvidiaTts;
#[cfg(feature = "tts-aws-polly")]
pub mod aws_polly;
#[cfg(feature = "tts-aws-polly")]
pub use aws_polly::AwsPollyTts;
