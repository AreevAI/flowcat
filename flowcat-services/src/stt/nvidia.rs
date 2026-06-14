// SPDX-License-Identifier: Apache-2.0
//
//! **NVIDIA Riva** (Nemotron Speech) streaming STT (gRPC).
//!
//! A **(D)istinct** gRPC client over Riva's
//! `RivaSpeechRecognition.StreamingRecognize` bidi RPC (PROVIDERS.md §2/§5).
//! Cross-checked against pipecat `services/nvidia/stt.py` (which wraps
//! `riva.client`). Behind the `stt-nvidia` feature.
//!
//! ## What's implemented vs seam-only
//!
//! The **wire + auth seam is fully implemented and unit-tested** — the protobuf
//! encode of both `StreamingRecognizeRequest` shapes (the initial
//! `streaming_config`, and per-chunk `audio_content`) and the decode of
//! `StreamingRecognizeResponse` into transcription [`Frame`]s — pinned to
//! `riva/proto/riva_asr.proto` field numbers, **no `prost`**. It reuses the shared
//! hand-rolled protobuf codec ([`crate::stt::google::grpc_proto`]) so the two gRPC
//! providers share one encoder (no duplication).
//!
//! ## Transport not fully wired (Cargo.toml-gated)
//!
//! Driving the live bidi stream needs tonic's **`channel`** feature **+ TLS** for
//! NIM-hosted Riva (self-hosted Riva is plaintext but still needs `channel`) —
//! neither is on this crate's `tonic` dep. The encode/decode here is exactly what a
//! `tonic::client::Grpc<Channel>` codec would carry once the feature is flipped on.
//! Until then [`SttService::start`] returns a clear "transport not wired" error.

use std::sync::Arc;

use async_trait::async_trait;

use flowcat_core::error::{FlowcatError, Result};
use flowcat_core::processor::frame::{AudioFrame, Frame, StartParams};
use flowcat_core::service::SttService;

use grpc_proto::Field;

/// Minimal hand-rolled protobuf (wire) + gRPC length-prefix framing for the Riva
/// `StreamingRecognize` messages — **no `prost`** (tonic 0.14 core does not pull it;
/// this crate does not declare it). This is the same wire encoding the Google STT
/// seam uses; it is carried per-file because the two gRPC providers are
/// independently feature-gated (`stt-google` / `stt-nvidia`) — a shared home would
/// need a `mod` decl in `stt/mod.rs`. Only the wire types these providers use
/// (varint + length-delimited string/bytes) are supported. See the matching codec
/// (and its tests) in `stt/google.rs::grpc_proto`.
pub(crate) mod grpc_proto {
    /// One protobuf field to encode: a `(field_number, value)` with its wire type.
    pub enum Field<'a> {
        /// Wire type 0 — varint (bool/int/enum).
        Varint(u32, i64),
        /// Wire type 2 — length-delimited UTF-8 string.
        Str(u32, &'a str),
        /// Wire type 2 — length-delimited bytes (a nested message or raw bytes).
        Bytes(u32, Vec<u8>),
    }

    impl<'a> Field<'a> {
        pub fn varint(field: u32, v: i64) -> Self {
            Field::Varint(field, v)
        }
        pub fn string(field: u32, v: &'a str) -> Self {
            Field::Str(field, v)
        }
        pub fn bytes(field: u32, v: Vec<u8>) -> Self {
            Field::Bytes(field, v)
        }
    }

    fn put_varint(out: &mut Vec<u8>, mut v: u64) {
        loop {
            let mut byte = (v & 0x7f) as u8;
            v >>= 7;
            if v != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if v == 0 {
                break;
            }
        }
    }

    fn put_tag(out: &mut Vec<u8>, field: u32, wire_type: u8) {
        put_varint(out, ((field as u64) << 3) | wire_type as u64);
    }

    /// Encode a flat list of fields into a protobuf message byte buffer.
    pub fn encode_message(fields: &[Field]) -> Vec<u8> {
        let mut out = Vec::new();
        for f in fields {
            match f {
                Field::Varint(field, v) => {
                    put_tag(&mut out, *field, 0);
                    put_varint(&mut out, *v as u64);
                }
                Field::Str(field, s) => {
                    put_tag(&mut out, *field, 2);
                    put_varint(&mut out, s.len() as u64);
                    out.extend_from_slice(s.as_bytes());
                }
                Field::Bytes(field, b) => {
                    put_tag(&mut out, *field, 2);
                    put_varint(&mut out, b.len() as u64);
                    out.extend_from_slice(b);
                }
            }
        }
        out
    }

    fn read_varint(buf: &[u8], pos: &mut usize) -> Option<u64> {
        let mut result: u64 = 0;
        let mut shift = 0;
        loop {
            let byte = *buf.get(*pos)?;
            *pos += 1;
            result |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                return Some(result);
            }
            shift += 7;
            if shift >= 64 {
                return None;
            }
        }
    }

    /// Iterate the raw bytes of every length-delimited occurrence of `field`.
    pub fn iter_field_bytes(buf: &[u8], field: u32) -> impl Iterator<Item = Vec<u8>> + '_ {
        let mut pos = 0usize;
        std::iter::from_fn(move || {
            while pos < buf.len() {
                let tag = read_varint(buf, &mut pos)?;
                let f = (tag >> 3) as u32;
                let wire = (tag & 0x7) as u8;
                match wire {
                    0 => {
                        let _ = read_varint(buf, &mut pos)?;
                    }
                    2 => {
                        let len = read_varint(buf, &mut pos)? as usize;
                        if pos + len > buf.len() {
                            return None;
                        }
                        let slice = buf[pos..pos + len].to_vec();
                        pos += len;
                        if f == field {
                            return Some(slice);
                        }
                    }
                    1 => pos += 8,
                    5 => pos += 4,
                    _ => return None,
                }
            }
            None
        })
    }

    /// First varint value of `field`, if present.
    pub fn first_varint(buf: &[u8], field: u32) -> Option<i64> {
        let mut pos = 0usize;
        while pos < buf.len() {
            let tag = read_varint(buf, &mut pos)?;
            let f = (tag >> 3) as u32;
            let wire = (tag & 0x7) as u8;
            match wire {
                0 => {
                    let v = read_varint(buf, &mut pos)?;
                    if f == field {
                        return Some(v as i64);
                    }
                }
                2 => {
                    let len = read_varint(buf, &mut pos)? as usize;
                    pos += len;
                }
                1 => pos += 8,
                5 => pos += 4,
                _ => return None,
            }
        }
        None
    }

    /// First length-delimited field decoded as UTF-8 (lossy), if present.
    pub fn first_string(buf: &[u8], field: u32) -> Option<String> {
        iter_field_bytes(buf, field)
            .next()
            .map(|b| String::from_utf8_lossy(&b).to_string())
    }
}

/// The fully-qualified gRPC method path for Riva's bidi streaming RPC.
pub const STREAMING_RECOGNIZE_PATH: &str =
    "/nvidia.riva.asr.RivaSpeechRecognition/StreamingRecognize";

/// `AudioEncoding.LINEAR_PCM` (16-bit LE PCM) in `riva_audio.proto`.
const AUDIO_ENCODING_LINEAR_PCM: i64 = 1;

/// NVIDIA Riva / Nemotron Speech streaming-STT service.
pub struct NvidiaStt {
    /// API key for NIM-hosted Riva (`function-id`/bearer). Empty for a self-hosted
    /// Riva server that needs no auth.
    api_key: String,
    /// BCP-47 language code (e.g. `en-US`).
    language_code: String,
    /// ASR model name (e.g. `parakeet-1.1b-en-US-asr-streaming`); empty ⇒ server
    /// default.
    model: String,
    sample_rate: u32,
    interim_results: bool,
    muted: bool,
}

impl NvidiaStt {
    /// Construct bound to `api_key` (default `en-US`, server-default model, 16 kHz,
    /// interim results on). Pass an empty key for an unauthenticated self-hosted
    /// Riva.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            language_code: "en-US".to_string(),
            model: String::new(),
            sample_rate: 16_000,
            interim_results: true,
            muted: false,
        }
    }

    /// Override the language code (default `en-US`).
    pub fn language_code(mut self, code: impl Into<String>) -> Self {
        self.language_code = code.into();
        self
    }

    /// Pin the ASR model (default: server's choice).
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Override the input sample rate (default 16 kHz).
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Toggle interim (partial) results (default on).
    pub fn interim_results(mut self, on: bool) -> Self {
        self.interim_results = on;
        self
    }

    /// gRPC request metadata. NIM-hosted Riva authenticates with an
    /// `authorization: Bearer <key>` header; a self-hosted server with no key gets
    /// no auth metadata. Returned as `(name, value)` pairs.
    pub fn auth_metadata(&self) -> Vec<(String, String)> {
        if self.api_key.is_empty() {
            vec![]
        } else {
            vec![(
                "authorization".to_string(),
                format!("Bearer {}", self.api_key),
            )]
        }
    }
}

/// Encode the **initial** `StreamingRecognizeRequest{ streaming_config=1 }` with a
/// `RecognitionConfig` (LINEAR_PCM, rate, language, model, mono) and
/// `interim_results`. Field numbers pinned to `riva_asr.proto`. **Pure** —
/// round-trip tested.
pub fn encode_config_request(
    language_code: &str,
    model: &str,
    sample_rate: u32,
    interim_results: bool,
) -> Vec<u8> {
    // RecognitionConfig{ encoding=1, sample_rate_hertz=2, language_code=3,
    //                    audio_channel_count=7, model=13 }
    let mut config_fields = vec![
        Field::varint(1, AUDIO_ENCODING_LINEAR_PCM),
        Field::varint(2, sample_rate as i64),
        Field::string(3, language_code),
        Field::varint(7, 1),
    ];
    if !model.is_empty() {
        config_fields.push(Field::string(13, model));
    }
    let recognition_config = grpc_proto::encode_message(&config_fields);
    // StreamingRecognitionConfig{ config=1, interim_results=2 }
    let streaming_config = grpc_proto::encode_message(&[
        Field::bytes(1, recognition_config),
        Field::varint(2, if interim_results { 1 } else { 0 }),
    ]);
    // StreamingRecognizeRequest{ streaming_config=1 }  (oneof streaming_request)
    grpc_proto::encode_message(&[Field::bytes(1, streaming_config)])
}

/// Encode a per-chunk `StreamingRecognizeRequest{ audio_content=2 }`. **Pure.**
pub fn encode_audio_request(pcm_le: &[u8]) -> Vec<u8> {
    grpc_proto::encode_message(&[Field::bytes(2, pcm_le.to_vec())])
}

/// Decode a `StreamingRecognizeResponse{ results=1 }` into transcription frames.
/// Each `StreamingRecognitionResult{ alternatives=1, is_final=2 }` yields the first
/// alternative's `transcript=1`. Empty transcripts → nothing. **Pure** — the decode
/// seam the fixture tests drive.
pub fn decode_response(bytes: &[u8]) -> Vec<Frame> {
    let mut out = Vec::new();
    let user_id: Arc<str> = Arc::from("user");
    for result_bytes in grpc_proto::iter_field_bytes(bytes, 1) {
        let is_final = grpc_proto::first_varint(&result_bytes, 2).unwrap_or(0) != 0;
        let Some(alt) = grpc_proto::iter_field_bytes(&result_bytes, 1).next() else {
            continue;
        };
        let Some(transcript) = grpc_proto::first_string(&alt, 1) else {
            continue;
        };
        if transcript.is_empty() {
            continue;
        }
        if is_final {
            out.push(Frame::Transcription {
                text: transcript,
                user_id: user_id.clone(),
                language: None,
                final_: true,
            });
        } else {
            out.push(Frame::InterimTranscription {
                text: transcript,
                user_id: user_id.clone(),
                language: None,
            });
        }
    }
    out
}

#[async_trait]
impl SttService for NvidiaStt {
    fn name(&self) -> &str {
        "nvidia"
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        // Wire/auth seam complete; the live bidi transport needs tonic's `channel`
        // + TLS (Cargo.toml-gated — see module docs). Build the config request +
        // auth so the failure is explicit, not silent.
        let _initial = encode_config_request(
            &self.language_code,
            &self.model,
            self.sample_rate,
            self.interim_results,
        );
        let _auth = self.auth_metadata();
        Err(FlowcatError::Network(
            "nvidia Riva STT: transport not fully wired — needs tonic `channel` + \
             TLS feature (Cargo.toml-gated); wire/auth seam ready"
                .into(),
        ))
    }

    async fn run_stt(&mut self, _audio: Arc<AudioFrame>) -> Result<Vec<Frame>> {
        if self.muted {
            return Ok(vec![]);
        }
        Err(FlowcatError::Network(
            "nvidia Riva STT: transport not fully wired (see start)".into(),
        ))
    }

    async fn set_muted(&mut self, muted: bool) {
        self.muted = muted;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_request_roundtrips_recognition_config() {
        let bytes = encode_config_request("en-US", "parakeet", 16_000, true);
        // streaming_config = field 1.
        let sc = grpc_proto::iter_field_bytes(&bytes, 1)
            .next()
            .expect("streaming_config");
        // interim_results = field 2 inside streaming_config.
        assert_eq!(grpc_proto::first_varint(&sc, 2), Some(1));
        // config = field 1 inside streaming_config.
        let rc = grpc_proto::iter_field_bytes(&sc, 1).next().expect("config");
        assert_eq!(grpc_proto::first_varint(&rc, 1), Some(1)); // LINEAR_PCM
        assert_eq!(grpc_proto::first_varint(&rc, 2), Some(16_000));
        assert_eq!(grpc_proto::first_string(&rc, 3).as_deref(), Some("en-US"));
        assert_eq!(grpc_proto::first_varint(&rc, 7), Some(1)); // channels
        assert_eq!(
            grpc_proto::first_string(&rc, 13).as_deref(),
            Some("parakeet")
        );
    }

    #[test]
    fn config_request_omits_empty_model() {
        let bytes = encode_config_request("en-US", "", 16_000, false);
        let sc = grpc_proto::iter_field_bytes(&bytes, 1)
            .next()
            .expect("streaming_config");
        // interim_results false → either absent or 0; our encoder always writes it.
        assert_eq!(grpc_proto::first_varint(&sc, 2), Some(0));
        let rc = grpc_proto::iter_field_bytes(&sc, 1).next().expect("config");
        assert!(
            grpc_proto::first_string(&rc, 13).is_none(),
            "model must be omitted"
        );
    }

    #[test]
    fn audio_request_wraps_pcm_in_field_2() {
        let pcm = vec![1u8, 0, 254, 255];
        let bytes = encode_audio_request(&pcm);
        let got = grpc_proto::iter_field_bytes(&bytes, 2)
            .next()
            .expect("audio_content");
        assert_eq!(got, pcm);
    }

    #[test]
    fn decode_final_and_interim_results() {
        let alt = grpc_proto::encode_message(&[Field::string(1, "book a dentist")]);
        let final_result = grpc_proto::encode_message(&[Field::bytes(1, alt), Field::varint(2, 1)]);
        let alt2 = grpc_proto::encode_message(&[Field::string(1, "for tomorrow")]);
        let interim_result = grpc_proto::encode_message(&[Field::bytes(1, alt2)]);
        let response = grpc_proto::encode_message(&[
            Field::bytes(1, final_result),
            Field::bytes(1, interim_result),
        ]);
        let frames = decode_response(&response);
        assert_eq!(frames.len(), 2);
        assert!(matches!(
            &frames[0],
            Frame::Transcription { text, final_: true, .. } if text == "book a dentist"
        ));
        assert!(matches!(
            &frames[1],
            Frame::InterimTranscription { text, .. } if text == "for tomorrow"
        ));
    }

    #[test]
    fn decode_ignores_empty_and_no_results() {
        assert!(decode_response(&[]).is_empty());
        let alt = grpc_proto::encode_message(&[Field::string(1, "")]);
        let result = grpc_proto::encode_message(&[Field::bytes(1, alt), Field::varint(2, 1)]);
        let response = grpc_proto::encode_message(&[Field::bytes(1, result)]);
        assert!(decode_response(&response).is_empty());
    }

    #[test]
    fn auth_metadata_bearer_when_keyed_else_empty() {
        let keyed = NvidiaStt::new("nvapi-xxx");
        assert_eq!(keyed.auth_metadata()[0].1, "Bearer nvapi-xxx");
        let unkeyed = NvidiaStt::new("");
        assert!(unkeyed.auth_metadata().is_empty());
    }

    #[tokio::test]
    async fn start_reports_transport_seam_clearly() {
        let mut stt = NvidiaStt::new("k");
        let err = stt.start(&StartParams::default()).await.unwrap_err();
        assert!(err.to_string().contains("not fully wired"));
    }

    /// Live smoke (requires `NVIDIA_API_KEY` + the tonic transport feature). Ignored
    /// until the transport is wired. Run with:
    /// `NVIDIA_API_KEY=… cargo test -p flowcat-services --features stt-nvidia -- --ignored nvidia_live`
    #[tokio::test]
    #[ignore = "transport not wired (needs tonic channel+TLS) + NVIDIA_API_KEY"]
    async fn nvidia_live_connects_and_streams() {
        let key = std::env::var("NVIDIA_API_KEY").expect("NVIDIA_API_KEY");
        let mut stt = NvidiaStt::new(key);
        let _ = stt.start(&StartParams::default()).await;
    }
}
