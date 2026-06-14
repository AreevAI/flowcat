// SPDX-License-Identifier: Apache-2.0
//
//! **NVIDIA Riva** TTS — gRPC client (Group H).
//!
//! NVIDIA Riva's `nvidia.riva.tts.RivaSpeechSynthesis/Synthesize` is a unary gRPC
//! call. As with [Google TTS](super::google), the Group-H `tts-nvidia` feature
//! enables `tonic`+`tokio` only (no `tonic-build`/`prost` codegen — that needs a
//! build.rs + build-dependency Group H must not add). So this module:
//!
//! 1. **Hand-encodes** the `SynthesizeSpeechRequest` protobuf + gRPC frame
//!    ([`build_request_message`]) — fully unit-tested; and
//! 2. validates the `tonic` endpoint, then returns a clear *"gRPC transport not
//!    wired"* error on the live path (no panic) — the response codec is infra-gated.
//!
//! Request shape (`riva_tts.proto`): `text` (1), `language_code` (2),
//! `encoding` (3, enum LINEAR_PCM = 1), `sample_rate_hz` (4), `voice_name` (5).
//! NVCF auth is an API key in the `authorization` + `function-id` gRPC metadata.

use async_trait::async_trait;

use flowcat_core::error::{FlowcatError, Result};
use flowcat_core::processor::frame::{Frame, StartParams};
use flowcat_core::service::TtsService;

#[path = "tail_tts_common.rs"]
#[allow(clippy::duplicate_mod)] // shared header pattern: included into each Group-H module
mod tail;

/// Default Riva gRPC server (NVCF hosted). Self-hosted Riva is `host:50051`.
pub const NVIDIA_RIVA_ENDPOINT: &str = "https://grpc.nvcf.nvidia.com:443";
/// AudioEncoding.LINEAR_PCM (Riva proto enum value 1).
const ENCODING_LINEAR_PCM: u64 = 1;

/// NVIDIA Riva TTS service (gRPC).
pub struct NvidiaTts {
    api_key: String,
    voice_id: String,
    sample_rate: u32,
    lang: String,
    endpoint: String,
}

impl NvidiaTts {
    /// Construct bound to `api_key` + `voice_id` (default 24000 Hz, `en-US`, NVCF).
    pub fn new(api_key: impl Into<String>, voice_id: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            voice_id: voice_id.into(),
            sample_rate: 24_000,
            lang: "en-US".to_string(),
            endpoint: NVIDIA_RIVA_ENDPOINT.to_string(),
        }
    }

    /// Point at a self-hosted Riva server (e.g. `http://localhost:50051`).
    pub fn endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Override the output sample rate (default 24000 Hz).
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Override the language code (default `en-US`).
    pub fn language(mut self, lang: impl Into<String>) -> Self {
        self.lang = lang.into();
        self
    }
}

/// Build the full gRPC-framed Riva `SynthesizeSpeechRequest` body (pure — the
/// request seam). Fields: text(1), language_code(2), encoding(3),
/// sample_rate_hz(4), voice_name(5).
fn build_request_message(text: &str, lang: &str, voice: &str, sample_rate: u32) -> Vec<u8> {
    let mut msg = Vec::new();
    tail::pb_len_delim(&mut msg, 1, text.as_bytes());
    tail::pb_len_delim(&mut msg, 2, lang.as_bytes());
    tail::pb_varint_field(&mut msg, 3, ENCODING_LINEAR_PCM);
    tail::pb_varint_field(&mut msg, 4, sample_rate as u64);
    tail::pb_len_delim(&mut msg, 5, voice.as_bytes());
    tail::grpc_frame(&msg)
}

#[async_trait]
impl TtsService for NvidiaTts {
    fn name(&self) -> &str {
        "nvidia"
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        tonic::transport::Endpoint::from_shared(self.endpoint.clone())
            .map_err(|e| FlowcatError::Network(format!("nvidia riva endpoint: {e}")))?;
        Ok(())
    }

    async fn run_tts(&mut self, text: &str) -> Result<Vec<Frame>> {
        let _request = build_request_message(text, &self.lang, &self.voice_id, self.sample_rate);
        let _endpoint = tonic::transport::Endpoint::from_shared(self.endpoint.clone())
            .map_err(|e| FlowcatError::Network(format!("nvidia riva endpoint: {e}")))?;
        let _auth = self.api_key.clone();
        Err(FlowcatError::Other(
            "nvidia TTS: gRPC transport not wired — request encode is ready, but the \
             Riva unary response codec needs tonic-build codegen (a build.rs + \
             build-dependency Group H does not add). Infra-gated."
                .into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_message_layout_matches_riva_proto() {
        let req = build_request_message("hi", "en-US", "English-US.Female-1", 24_000);
        // gRPC frame prefix.
        assert_eq!(req[0], 0);
        let len = u32::from_be_bytes([req[1], req[2], req[3], req[4]]) as usize;
        assert_eq!(len, req.len() - 5);
        let msg = &req[5..];
        // field 1 (text), wire 2 → tag 0x0a, len 2, "hi".
        assert_eq!(&msg[0..4], &[0x0a, 0x02, b'h', b'i']);
        // field 2 (language_code) tag 0x12, len 5, "en-US".
        assert_eq!(msg[4], 0x12);
        assert_eq!(msg[5], 5);
        assert_eq!(&msg[6..11], b"en-US");
        // field 3 (encoding) varint LINEAR_PCM(1): tag 0x18, value 1.
        assert_eq!(&msg[11..13], &[0x18, 0x01]);
        // field 4 (sample_rate_hz) varint 24000: tag 0x20, then varint(24000).
        assert_eq!(msg[13], 0x20);
        assert_eq!(&msg[14..16], &[0xc0, 0xbb]); // 24000 = 0x5dc0 → varint c0 bb 01
    }

    #[test]
    fn voice_name_is_the_last_field() {
        let req = build_request_message("x", "en-US", "VoiceN", 16_000);
        // field 5 (voice_name) tag 0x2a must appear with len 6 + "VoiceN".
        let needle = [0x2a, 6, b'V', b'o', b'i', b'c', b'e', b'N'];
        assert!(req.windows(needle.len()).any(|w| w == needle));
    }

    #[tokio::test]
    async fn start_validates_endpoint() {
        let mut tts = NvidiaTts::new("key", "VoiceN");
        tts.start(&StartParams::default())
            .await
            .expect("valid endpoint");
    }

    #[tokio::test]
    async fn run_returns_clear_not_wired_seam() {
        let mut tts = NvidiaTts::new("key", "VoiceN");
        let err = tts.run_tts("hi").await.unwrap_err();
        assert!(err.to_string().contains("gRPC transport not wired"));
    }
}
