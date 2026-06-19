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
use bytes::{Buf, BufMut};
use futures::stream::{self, StreamExt};
use http::uri::PathAndQuery;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tonic::client::Grpc;
use tonic::codec::{Codec, DecodeBuf, Decoder, EncodeBuf, Encoder};
use tonic::metadata::MetadataValue;
use tonic::transport::{ClientTlsConfig, Endpoint};
use tonic::{Request, Status};

use flowcat_core::error::{FlowcatError, Result};
use flowcat_core::processor::frame::{AudioFrame, Frame, StartParams};
use flowcat_core::service::SttService;

use grpc_proto::Field;

/// NVIDIA NIM's hosted Riva ASR gRPC host (NVCF). A self-hosted Riva server can be
/// targeted with [`NvidiaStt::endpoint`].
pub const NVIDIA_NVCF_HOST: &str = "grpc.nvcf.nvidia.com";

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

/// End-of-utterance gap: NVCF parakeet streams cumulative interims and (in practice)
/// no `is_final`, so once an interim exists this much further audio with no new
/// interim closes the user turn (one final transcript).
const TURN_GAP_MS: u64 = 700;

/// Live gRPC bidi connection state (present once [`SttService::start`] succeeds).
struct Connection {
    /// Sender into the outbound request stream; `run_stt` pushes encoded audio here.
    /// Dropping it half-closes the request stream (clean teardown).
    audio_tx: mpsc::UnboundedSender<Vec<u8>>,
    /// Transcription frames the reader task has decoded so far (drained non-blocking).
    frames: mpsc::UnboundedReceiver<Frame>,
    /// The response-reader task; aborted on drop.
    reader: JoinHandle<()>,
}

impl Drop for Connection {
    fn drop(&mut self) {
        self.reader.abort();
    }
}

/// NVIDIA Riva / Nemotron Speech streaming-STT service.
pub struct NvidiaStt {
    /// API key for NIM-hosted Riva (`function-id`/bearer). Empty for a self-hosted
    /// Riva server that needs no auth.
    api_key: String,
    /// gRPC host (default NVCF; a self-hosted Riva sets its own host).
    host: String,
    /// NVCF function id — the hosted ASR model to route to (NVCF only; empty for a
    /// self-hosted Riva, which has no function routing).
    function_id: String,
    /// BCP-47 language code (e.g. `en-US`).
    language_code: String,
    /// ASR model name (e.g. `parakeet-1.1b-en-US-asr-streaming`); empty ⇒ server
    /// default.
    model: String,
    sample_rate: u32,
    interim_results: bool,
    muted: bool,
    /// Live bidi connection — `None` until `start` opens it.
    conn: Option<Connection>,
    /// Latest cumulative interim text of the in-progress turn (Riva interims replace,
    /// not append). Emitted as the turn-final once transcription goes quiet.
    turn_text: String,
    /// Audio ms since the last new interim — the end-of-utterance clock.
    quiet_ms: u64,
}

impl NvidiaStt {
    /// Construct bound to `api_key` (default NVCF host, `en-US`, server-default model,
    /// 16 kHz, interim results on). Pass an empty key for an unauthenticated
    /// self-hosted Riva.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            host: NVIDIA_NVCF_HOST.to_string(),
            function_id: String::new(),
            language_code: "en-US".to_string(),
            model: String::new(),
            sample_rate: 16_000,
            interim_results: true,
            muted: false,
            conn: None,
            turn_text: String::new(),
            quiet_ms: 0,
        }
    }

    /// Override the gRPC host (default `grpc.nvcf.nvidia.com`; set for a self-hosted
    /// Riva server, e.g. `localhost:50051`).
    pub fn endpoint(mut self, host: impl Into<String>) -> Self {
        self.host = host.into();
        self
    }

    /// Set the NVCF `function-id` (the hosted ASR model to route to). Required for
    /// NIM-hosted Riva; leave empty for a self-hosted server.
    pub fn function_id(mut self, id: impl Into<String>) -> Self {
        self.function_id = id.into();
        self
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
    /// `authorization: Bearer <key>` header **and** a `function-id` routing the call
    /// to the chosen ASR model; a self-hosted server with no key gets neither.
    /// Returned as `(name, value)` pairs.
    pub fn auth_metadata(&self) -> Vec<(String, String)> {
        let mut md = Vec::new();
        if !self.api_key.is_empty() {
            md.push((
                "authorization".to_string(),
                format!("Bearer {}", self.api_key),
            ));
        }
        if !self.function_id.is_empty() {
            md.push(("function-id".to_string(), self.function_id.clone()));
        }
        md
    }
}

/// Raw-bytes [`tonic::codec::Codec`]: messages ARE the already-encoded
/// `StreamingRecognizeRequest` bytes ([`encode_config_request`]/[`encode_audio_request`])
/// and the raw `StreamingRecognizeResponse` bytes handed to [`decode_response`]. tonic
/// owns the gRPC length-prefix framing, so the pure encode/decode fns are reused over
/// the live stream verbatim. (Mirrors the Google STT codec — the two share the wire.)
#[derive(Clone, Copy, Default)]
pub(crate) struct RivaCodec;

impl Codec for RivaCodec {
    type Encode = Vec<u8>;
    type Decode = Vec<u8>;
    type Encoder = RawEncoder;
    type Decoder = RawDecoder;
    fn encoder(&mut self) -> RawEncoder {
        RawEncoder
    }
    fn decoder(&mut self) -> RawDecoder {
        RawDecoder
    }
}

pub(crate) struct RawEncoder;
impl Encoder for RawEncoder {
    type Item = Vec<u8>;
    type Error = Status;
    fn encode(
        &mut self,
        item: Vec<u8>,
        dst: &mut EncodeBuf<'_>,
    ) -> std::result::Result<(), Status> {
        dst.put_slice(&item);
        Ok(())
    }
}

pub(crate) struct RawDecoder;
impl Decoder for RawDecoder {
    type Item = Vec<u8>;
    type Error = Status;
    fn decode(&mut self, src: &mut DecodeBuf<'_>) -> std::result::Result<Option<Vec<u8>>, Status> {
        let n = src.remaining();
        if n == 0 {
            return Ok(None);
        }
        let mut v = vec![0u8; n];
        src.copy_to_slice(&mut v);
        Ok(Some(v))
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
        // The gRPC dial + bidi stream run in a **background task**: NVCF cold-starts
        // can take many seconds, and doing it inline (in `start` or `run_stt`) would
        // stall the pipeline. `run_stt` just queues audio (buffered until the stream
        // is up) and drains decoded frames.
        let (audio_tx, audio_rx) = mpsc::unbounded_channel::<Vec<u8>>();
        let (frame_tx, frame_rx) = mpsc::unbounded_channel::<Frame>();
        let config_req = encode_config_request(
            &self.language_code,
            &self.model,
            self.sample_rate,
            self.interim_results,
        );
        let host = self.host.clone();
        let metadata = self.auth_metadata();
        let reader = tokio::spawn(async move {
            if let Err(e) = run_session(host, metadata, config_req, audio_rx, frame_tx).await {
                tracing::error!(target: "flowcat_services::stt", error = %e, "nvidia STT session ended");
            }
        });
        self.conn = Some(Connection {
            audio_tx,
            frames: frame_rx,
            reader,
        });
        Ok(())
    }

    async fn run_stt(&mut self, audio: Arc<AudioFrame>) -> Result<Vec<Frame>> {
        if self.muted {
            return Ok(vec![]);
        }
        let chunk_ms = if audio.sample_rate > 0 {
            (audio.pcm.len() as u64 * 1000) / audio.sample_rate as u64
        } else {
            0
        };
        let conn = self
            .conn
            .as_mut()
            .ok_or_else(|| FlowcatError::Network("nvidia STT: run_stt before start".into()))?;
        // PCM i16 → little-endian bytes (the LINEAR_PCM wire format Riva expects).
        let mut pcm_le = Vec::with_capacity(audio.pcm.len() * 2);
        for s in &audio.pcm {
            pcm_le.extend_from_slice(&s.to_le_bytes());
        }
        // Buffered: queues during NVCF cold-start, flushed once the stream is up.
        let _ = conn.audio_tx.send(encode_audio_request(&pcm_le));

        // Drain decoded results. Riva interims are cumulative (full hypothesis each
        // time), so the latest replaces `turn_text`; a real `is_final` (rare on NVCF)
        // closes the turn immediately.
        let mut got_interim = false;
        let mut finals = Vec::new();
        while let Ok(f) = conn.frames.try_recv() {
            match f {
                Frame::InterimTranscription { text, .. } => {
                    if !text.trim().is_empty() {
                        self.turn_text = text;
                        got_interim = true;
                    }
                }
                Frame::Transcription { .. } => {
                    self.turn_text.clear();
                    self.quiet_ms = 0;
                    finals.push(f);
                }
                other => finals.push(other),
            }
        }
        if !finals.is_empty() {
            return Ok(finals);
        }
        if got_interim {
            self.quiet_ms = 0;
        } else {
            self.quiet_ms = self.quiet_ms.saturating_add(chunk_ms);
        }
        // End of utterance: an interim exists and transcription has gone quiet —
        // emit the latest hypothesis as the single turn-final.
        if !self.turn_text.is_empty() && self.quiet_ms >= TURN_GAP_MS {
            let text = std::mem::take(&mut self.turn_text);
            self.quiet_ms = 0;
            return Ok(vec![Frame::Transcription {
                text,
                user_id: Arc::from("user"),
                language: None,
                final_: true,
            }]);
        }
        Ok(vec![])
    }

    async fn set_muted(&mut self, muted: bool) {
        // Bot speaking / turn closed: drop any half-turn remainder so the next user
        // turn starts clean.
        if muted {
            self.turn_text.clear();
            self.quiet_ms = 0;
        }
        self.muted = muted;
    }
}

/// Background session: TLS-dial the gRPC host, open the bidi `StreamingRecognize`
/// stream (config request first, then the buffered audio chunks), and forward every
/// decoded `StreamingRecognizeResponse` as transcription [`Frame`]s to `frame_tx`.
/// Runs for the life of the call; ends on stream close/error.
async fn run_session(
    host: String,
    metadata: Vec<(String, String)>,
    config_req: Vec<u8>,
    audio_rx: mpsc::UnboundedReceiver<Vec<u8>>,
    frame_tx: mpsc::UnboundedSender<Frame>,
) -> Result<()> {
    let tls = ClientTlsConfig::new()
        .domain_name(host.clone())
        .with_webpki_roots();
    let endpoint = Endpoint::from_shared(format!("https://{host}"))
        .map_err(|e| FlowcatError::Network(format!("nvidia endpoint: {e}")))?
        .tls_config(tls)
        .map_err(|e| FlowcatError::Network(format!("nvidia tls: {e}")))?;
    let channel = endpoint.connect().await.map_err(|e| {
        // tonic's Display is just "transport error"; surface the source chain so
        // a TLS / h2 / DNS / refused-connection cause is diagnosable.
        let mut detail = format!("{e}");
        let mut src = std::error::Error::source(&e);
        while let Some(s) = src {
            detail.push_str(&format!(" → {s}"));
            src = s.source();
        }
        FlowcatError::Network(format!("nvidia connect: {detail}"))
    })?;

    let audio_stream = stream::unfold(audio_rx, |mut rx| async move {
        rx.recv().await.map(|m| (m, rx))
    });
    let out_stream = stream::once(async move { config_req }).chain(audio_stream);

    let mut request = Request::new(out_stream);
    for (name, value) in metadata {
        let key: tonic::metadata::MetadataKey<_> = name
            .parse()
            .map_err(|e| FlowcatError::Network(format!("nvidia metadata key: {e}")))?;
        let val: MetadataValue<_> = value
            .parse()
            .map_err(|e| FlowcatError::Network(format!("nvidia metadata value: {e}")))?;
        request.metadata_mut().insert(key, val);
    }

    let path = PathAndQuery::from_static(STREAMING_RECOGNIZE_PATH);
    let mut grpc = Grpc::new(channel);
    grpc.ready()
        .await
        .map_err(|e| FlowcatError::Network(format!("nvidia grpc not ready: {e}")))?;
    let response = grpc
        .streaming(request, path, RivaCodec)
        .await
        .map_err(|e| FlowcatError::Network(format!("nvidia streaming: {e}")))?;

    let mut inbound = response.into_inner();
    loop {
        match inbound.message().await {
            Ok(Some(bytes)) => {
                for frame in decode_response(&bytes) {
                    if frame_tx.send(frame).is_err() {
                        return Ok(()); // consumer gone
                    }
                }
            }
            Ok(None) => break,
            Err(status) => return Err(FlowcatError::Network(format!("nvidia stream: {status}"))),
        }
    }
    Ok(())
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

    #[test]
    fn auth_metadata_carries_bearer_and_function_id() {
        let stt = NvidiaStt::new("k").function_id("fn-123");
        let md = stt.auth_metadata();
        assert!(md
            .iter()
            .any(|(n, v)| n == "authorization" && v == "Bearer k"));
        assert!(md.iter().any(|(n, v)| n == "function-id" && v == "fn-123"));
        // Self-hosted (no key, no function) → no metadata.
        assert!(NvidiaStt::new("").auth_metadata().is_empty());
    }

    /// Live smoke (requires `NVIDIA_API_KEY` + a NIM ASR `NVIDIA_FUNCTION_ID`). Run:
    /// `NVIDIA_API_KEY=… NVIDIA_FUNCTION_ID=… cargo test -p flowcat-services \
    ///   --features stt-nvidia -- --ignored nvidia_live`
    #[tokio::test]
    #[ignore = "requires NVIDIA_API_KEY + NVIDIA_FUNCTION_ID (NVCF ASR)"]
    async fn nvidia_live_connects_and_streams() {
        let key = std::env::var("NVIDIA_API_KEY").expect("NVIDIA_API_KEY");
        let function_id = std::env::var("NVIDIA_FUNCTION_ID").expect("NVIDIA_FUNCTION_ID");
        let mut stt = NvidiaStt::new(key).function_id(function_id);
        stt.start(&StartParams::default()).await.expect("start");
        let silence = Arc::new(AudioFrame::mono(vec![0i16; 1600], 16_000));
        let _ = stt.run_stt(silence).await.expect("run_stt");
    }
}
