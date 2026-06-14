// SPDX-License-Identifier: Apache-2.0
//
//! Streaming speech-to-text providers.
//!
//! Impls of [`SttService`](flowcat_core::service::SttService). The reference impl
//! (**Deepgram**, streaming WebSocket) sits behind `stt-deepgram`; all other providers
//! follow the same pattern. Each provider is `dep:`-feature-gated so the default build
//! pulls no STT dep.
//!
//! ## Provider homes (PROVIDERS.md §2)
//!
//! Protocol families: streaming-WebSocket (each its own JSON schema, the Deepgram
//! template), Whisper-HTTP segmented (`stt-openai` is the **(D)** family client;
//! groq/fal/speaches/xai are **(W)**), gRPC (google/nvidia-riva via `tonic`), and
//! whisper-local (`whisper-rs`, C build). Every provider has a stub module below so
//! adding a provider fills one body and never edits this `mod`/`use` list.

#[cfg(feature = "stt-deepgram")]
pub mod deepgram;
#[cfg(feature = "stt-deepgram")]
pub use deepgram::{DeepgramStt, DeepgramSttBuilder};

// --- (D)istinct streaming-WS / REST STT clients (PROVIDERS.md §2) ---
#[cfg(feature = "stt-assemblyai")]
pub mod assemblyai;
#[cfg(feature = "stt-assemblyai")]
pub use assemblyai::AssemblyAiStt;
#[cfg(feature = "stt-gladia")]
pub mod gladia;
#[cfg(feature = "stt-gladia")]
pub use gladia::GladiaStt;
#[cfg(feature = "stt-soniox")]
pub mod soniox;
#[cfg(feature = "stt-soniox")]
pub use soniox::SonioxStt;
#[cfg(feature = "stt-speechmatics")]
pub mod speechmatics;
#[cfg(feature = "stt-speechmatics")]
pub use speechmatics::SpeechmaticsStt;
#[cfg(feature = "stt-cartesia")]
pub mod cartesia;
#[cfg(feature = "stt-cartesia")]
pub use cartesia::CartesiaStt;
#[cfg(feature = "stt-azure")]
pub mod azure;
#[cfg(feature = "stt-azure")]
pub use azure::AzureStt;
#[cfg(feature = "stt-gradium")]
pub mod gradium;
#[cfg(feature = "stt-gradium")]
pub use gradium::GradiumStt;
#[cfg(feature = "stt-elevenlabs")]
pub mod elevenlabs;
#[cfg(feature = "stt-elevenlabs")]
pub use elevenlabs::ElevenLabsStt;
#[cfg(feature = "stt-sarvam")]
pub mod sarvam;
#[cfg(feature = "stt-sarvam")]
pub use sarvam::SarvamStt;
#[cfg(feature = "stt-mistral")]
pub mod mistral;
#[cfg(feature = "stt-mistral")]
pub use mistral::MistralStt;

// --- Whisper-HTTP family: openai is the (D) client; the rest are (W) ---
#[cfg(feature = "stt-openai")]
pub mod openai;
#[cfg(feature = "stt-openai")]
pub use openai::OpenAiStt;
#[cfg(feature = "stt-groq")]
pub mod groq;
#[cfg(feature = "stt-groq")]
pub use groq::GroqStt;
#[cfg(feature = "stt-fal")]
pub mod fal;
#[cfg(feature = "stt-fal")]
pub use fal::FalStt;
#[cfg(feature = "stt-speaches")]
pub mod speaches;
#[cfg(feature = "stt-speaches")]
pub use speaches::SpeachesStt;
#[cfg(feature = "stt-xai")]
pub mod xai;
#[cfg(feature = "stt-xai")]
pub use xai::XaiStt;

// --- gRPC + AWS + local STT (toolchain-heavy — PROVIDERS.md §5) ---
#[cfg(feature = "stt-google")]
pub mod google;
#[cfg(feature = "stt-google")]
pub use google::GoogleStt;
#[cfg(feature = "stt-nvidia")]
pub mod nvidia;
#[cfg(feature = "stt-nvidia")]
pub use nvidia::NvidiaStt;
#[cfg(feature = "stt-aws-transcribe")]
pub mod aws_transcribe;
#[cfg(feature = "stt-aws-transcribe")]
pub use aws_transcribe::AwsTranscribeStt;
#[cfg(feature = "stt-whisper-local")]
pub mod whisper_local;
#[cfg(feature = "stt-whisper-local")]
pub use whisper_local::WhisperLocalStt;
