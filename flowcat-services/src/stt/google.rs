// SPDX-License-Identifier: Apache-2.0
//
//! **Google Cloud Speech-to-Text V2** streaming STT (gRPC).
//!
//! A **(D)istinct** gRPC client over Google's `Speech.StreamingRecognize` bidi RPC
//! (PROVIDERS.md §2/§5). Cross-checked against pipecat
//! `services/google/stt.py`. Behind the `stt-google` feature.
//!
//! ## What's implemented vs seam-only
//!
//! The **wire + auth seam is fully implemented and unit-tested** — it is the part a
//! gRPC client actually has to get right and the part `tonic`'s generated code would
//! merely wrap:
//!
//! - the protobuf **encode** of the two `StreamingRecognizeRequest` messages — the
//!   initial config ([`encode_config_request`]) and per-chunk audio
//!   ([`encode_audio_request`]) — by hand (field numbers pinned to
//!   `google/cloud/speech/v2/cloud_speech.proto`), no `prost`;
//! - the protobuf **decode** of `StreamingRecognizeResponse` into transcription
//!   [`Frame`]s ([`decode_response`]);
//! - the gRPC **length-prefix framing** ([`grpc_frame`] / [`grpc_deframe`]);
//! - the **OAuth2 Bearer** auth metadata ([`GoogleStt::auth_metadata`]) — the caller
//!   supplies a pre-minted access token (the standard ADC/service-account flow mints
//!   it out of band; see the transport note below).
//!
//! ## Transport (live gRPC bidi — tonic `channel` + rustls/aws-lc-rs + webpki roots)
//!
//! [`SttService::start`] opens a TLS [`Channel`](tonic::transport::Channel) to the
//! **regional** host derived from the recognizer's `locations/<loc>` segment
//! (`global` → `speech.googleapis.com`, else `<loc>-speech.googleapis.com`), then
//! drives `StreamingRecognize` with a [`tonic::client::Grpc`] over the raw-bytes
//! [`SpeechCodec`] — tonic owns the gRPC length-prefix framing, so the pure
//! `encode_*`/`decode_response` fns ride the live stream unchanged ([`grpc_proto::grpc_frame`]
//! is kept only for its round-trip unit test). The outbound request stream emits the
//! config request then each audio chunk; a reader task decodes responses into a
//! channel that [`SttService::run_stt`] drains non-blocking (the same shape as the
//! WS-STT connectors). Auth is the `authorization: Bearer <token>` request metadata.
//! OAuth2 **token minting** (service-account JWT → access token) stays the caller's
//! job — the connector takes a pre-minted, short-lived access token.

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

/// Minimal hand-rolled protobuf (wire) + gRPC length-prefix framing — enough to
/// encode the two `StreamingRecognizeRequest` shapes and decode
/// `StreamingRecognizeResponse`, with **no `prost`** (tonic 0.14 core does not pull
/// prost; this crate does not declare it). Shared by the NVIDIA Riva STT seam via
/// `crate::stt::google::grpc_proto` (both gRPC providers reuse one encoder — no
/// duplication). Only the wire types these two providers use are supported
/// (varint, length-delimited string/bytes).
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

    /// Encode a base-128 varint.
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

    /// Encode a field tag `(field_number << 3) | wire_type`.
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

    /// Read a base-128 varint at `*pos`, advancing it. `None` on truncation.
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

    /// Iterate the raw bytes of every length-delimited (wire type 2) occurrence of
    /// `field` in `buf` (repeated fields yield multiple). Unknown fields are skipped.
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
                    1 => pos += 8, // 64-bit
                    5 => pos += 4, // 32-bit
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

    /// First length-delimited field decoded as UTF-8 (lossless), if present.
    pub fn first_string(buf: &[u8], field: u32) -> Option<String> {
        iter_field_bytes(buf, field)
            .next()
            .map(|b| String::from_utf8_lossy(&b).to_string())
    }

    /// Frame a message in the gRPC length-prefix format: a 1-byte compression flag
    /// (0 = uncompressed) + a 4-byte big-endian length + the message bytes. This is
    /// what rides inside each HTTP/2 DATA frame on a gRPC stream. Part of the
    /// documented transport seam — exercised by the round-trip test; the live
    /// transport (Cargo.toml-gated tonic `channel`) is what will call it in prod.
    #[allow(dead_code)] // seam: used once the tonic transport feature is enabled
    pub fn grpc_frame(message: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(message.len() + 5);
        out.push(0); // not compressed
        out.extend_from_slice(&(message.len() as u32).to_be_bytes());
        out.extend_from_slice(message);
        out
    }

    /// Strip a single gRPC length-prefix frame, returning the inner message bytes.
    /// `None` if the buffer is too short or claims a length past its end. Seam pair
    /// of [`grpc_frame`].
    #[allow(dead_code)] // seam: used once the tonic transport feature is enabled
    pub fn grpc_deframe(framed: &[u8]) -> Option<&[u8]> {
        if framed.len() < 5 {
            return None;
        }
        let len = u32::from_be_bytes([framed[1], framed[2], framed[3], framed[4]]) as usize;
        framed.get(5..5 + len)
    }
}

/// `Speech.StreamingRecognize` is served from this host (global endpoint). Regional
/// endpoints (`<region>-speech.googleapis.com`) are also valid; the host is fixed
/// per region, never caller-controlled (no SSRF surface).
pub const GOOGLE_SPEECH_HOST: &str = "speech.googleapis.com";

/// The fully-qualified gRPC method path for the bidi streaming RPC.
pub const STREAMING_RECOGNIZE_PATH: &str = "/google.cloud.speech.v2.Speech/StreamingRecognize";

/// LINEAR16 in `ExplicitDecodingConfig.AudioEncoding`.
const AUDIO_ENCODING_LINEAR16: i64 = 1;

/// Live gRPC bidi connection state (present once [`SttService::start`] succeeds).
struct Connection {
    /// Sender into the outbound request stream; `run_stt` pushes encoded audio
    /// requests here. Dropping it half-closes the request stream (clean teardown).
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

/// Google Cloud Speech V2 streaming-STT service.
pub struct GoogleStt {
    /// A pre-minted OAuth2 access token (Bearer). The ADC / service-account flow
    /// mints this out of band (token minting is not wired here — see module docs).
    access_token: String,
    /// `projects/<project>/locations/<location>/recognizers/_` — the recognizer
    /// resource the config request targets.
    recognizer: String,
    /// BCP-47 language code(s) (e.g. `en-US`).
    language_code: String,
    /// Recognition model (e.g. `long`, `telephony`, `chirp`).
    model: String,
    sample_rate: u32,
    interim_results: bool,
    muted: bool,
    /// Live bidi connection — `None` until `start` opens it.
    conn: Option<Connection>,
}

impl GoogleStt {
    /// Construct bound to a pre-minted access token + project recognizer path.
    /// Defaults: `en-US`, model `long`, 16 kHz, interim results on.
    pub fn new(access_token: impl Into<String>, recognizer: impl Into<String>) -> Self {
        Self {
            access_token: access_token.into(),
            recognizer: recognizer.into(),
            language_code: "en-US".to_string(),
            model: "long".to_string(),
            sample_rate: 16_000,
            interim_results: true,
            muted: false,
            conn: None,
        }
    }

    /// Override the language code (default `en-US`).
    pub fn language_code(mut self, code: impl Into<String>) -> Self {
        self.language_code = code.into();
        self
    }

    /// Override the recognition model (default `long`).
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

    /// The gRPC request metadata: `authorization: Bearer <token>`. (A real client
    /// also sets `x-goog-user-project`/`x-goog-request-params`; the Bearer is the
    /// security-relevant one.) Returned as `(name, value)` pairs.
    pub fn auth_metadata(&self) -> Vec<(String, String)> {
        vec![(
            "authorization".to_string(),
            format!("Bearer {}", self.access_token),
        )]
    }

    /// The gRPC host for this recognizer's location. The recognizer is
    /// `projects/<p>/locations/<loc>/recognizers/_`; `global` → the global host,
    /// any other `<loc>` → the regional `<loc>-speech.googleapis.com` (a regional
    /// recognizer MUST be reached on its regional endpoint). Host is recognizer-
    /// derived (operator config, never request input) → no SSRF surface.
    fn endpoint_host(&self) -> String {
        let loc = self
            .recognizer
            .split('/')
            .skip_while(|s| *s != "locations")
            .nth(1)
            .unwrap_or("global");
        if loc == "global" || loc.is_empty() {
            GOOGLE_SPEECH_HOST.to_string()
        } else {
            format!("{loc}-speech.googleapis.com")
        }
    }
}

/// Encode the **initial** `StreamingRecognizeRequest` (recognizer + streaming_config
/// with an explicit LINEAR16 decoding config, language, model, and interim-results).
/// Field numbers pinned to `cloud_speech.proto`. **Pure** — round-trip tested.
pub fn encode_config_request(
    recognizer: &str,
    language_code: &str,
    model: &str,
    sample_rate: u32,
    interim_results: bool,
) -> Vec<u8> {
    // ExplicitDecodingConfig{ encoding=1 LINEAR16, sample_rate_hertz=2, channels=3 }
    let explicit_decoding = grpc_proto::encode_message(&[
        Field::varint(1, AUDIO_ENCODING_LINEAR16),
        Field::varint(2, sample_rate as i64),
        Field::varint(3, 1),
    ]);
    // RecognitionConfig{ explicit_decoding_config=8, model=9, language_codes=10 }
    let recognition_config = grpc_proto::encode_message(&[
        Field::bytes(8, explicit_decoding),
        Field::string(9, model),
        Field::string(10, language_code),
    ]);
    // StreamingRecognitionFeatures{ interim_results=2 }
    let streaming_features =
        grpc_proto::encode_message(&[Field::varint(2, if interim_results { 1 } else { 0 })]);
    // StreamingRecognitionConfig{ config=1, streaming_features=2 }
    let streaming_config = grpc_proto::encode_message(&[
        Field::bytes(1, recognition_config),
        Field::bytes(2, streaming_features),
    ]);
    // StreamingRecognizeRequest{ recognizer=3, streaming_config=6 }
    grpc_proto::encode_message(&[
        Field::string(3, recognizer),
        Field::bytes(6, streaming_config),
    ])
}

/// Encode a per-chunk audio `StreamingRecognizeRequest{ audio=5 }`. **Pure.**
pub fn encode_audio_request(pcm_le: &[u8]) -> Vec<u8> {
    grpc_proto::encode_message(&[Field::bytes(5, pcm_le.to_vec())])
}

/// Decode a `StreamingRecognizeResponse` into transcription frames. Walks
/// `results=6 → StreamingRecognitionResult{ alternatives=1, is_final=2 }`, taking
/// the first alternative's `transcript=1`. Empty transcripts → nothing. **Pure** —
/// the decode seam the fixture tests drive.
pub fn decode_response(bytes: &[u8]) -> Vec<Frame> {
    let mut out = Vec::new();
    let user_id: Arc<str> = Arc::from("user");
    for result_bytes in grpc_proto::iter_field_bytes(bytes, 6) {
        // StreamingRecognitionResult.
        let is_final = grpc_proto::first_varint(&result_bytes, 2).unwrap_or(0) != 0;
        let Some(alt) = grpc_proto::iter_field_bytes(&result_bytes, 1).next() else {
            continue;
        };
        // SpeechRecognitionAlternative.transcript = 1.
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

/// Raw-bytes [`tonic::codec::Codec`]: messages ARE the already-encoded
/// `StreamingRecognizeRequest` bytes ([`encode_config_request`]/[`encode_audio_request`])
/// and the raw `StreamingRecognizeResponse` bytes handed to [`decode_response`]. tonic
/// owns the gRPC length-prefix framing, so the pure encode/decode fns are reused over
/// the live stream verbatim.
#[derive(Clone, Copy, Default)]
pub(crate) struct SpeechCodec;

impl Codec for SpeechCodec {
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
        // tonic deframes one full message into `src`; copy it out as raw bytes.
        let n = src.remaining();
        if n == 0 {
            return Ok(None);
        }
        let mut v = vec![0u8; n];
        src.copy_to_slice(&mut v);
        Ok(Some(v))
    }
}

#[async_trait]
impl SttService for GoogleStt {
    fn name(&self) -> &str {
        "google"
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        if self.access_token.trim().is_empty() {
            return Err(FlowcatError::Network(
                "google STT: empty access token (mint one, e.g. `gcloud auth print-access-token`)"
                    .into(),
            ));
        }
        let host = self.endpoint_host();

        // 1. TLS channel to the (regional) host — rustls/aws-lc-rs + webpki roots.
        let tls = ClientTlsConfig::new()
            .domain_name(host.clone())
            .with_webpki_roots();
        let channel = Endpoint::from_shared(format!("https://{host}"))
            .map_err(|e| FlowcatError::Network(format!("google endpoint: {e}")))?
            .tls_config(tls)
            .map_err(|e| FlowcatError::Network(format!("google tls: {e}")))?
            .connect()
            .await
            .map_err(|e| FlowcatError::Network(format!("google connect: {e}")))?;

        // 2. Outbound request stream: the config request, then queued audio chunks.
        let (audio_tx, audio_rx) = mpsc::unbounded_channel::<Vec<u8>>();
        let config_req = encode_config_request(
            &self.recognizer,
            &self.language_code,
            &self.model,
            self.sample_rate,
            self.interim_results,
        );
        let audio_stream = stream::unfold(audio_rx, |mut rx| async move {
            rx.recv().await.map(|m| (m, rx))
        });
        let out_stream = stream::once(async move { config_req }).chain(audio_stream);

        // 3. Build the request + attach the Bearer auth metadata.
        let mut request = Request::new(out_stream);
        let bearer: MetadataValue<_> = format!("Bearer {}", self.access_token)
            .parse()
            .map_err(|e| FlowcatError::Network(format!("google auth metadata: {e}")))?;
        request.metadata_mut().insert("authorization", bearer);

        // 4. Open the bidi StreamingRecognize call over the raw codec.
        let path = PathAndQuery::from_static(STREAMING_RECOGNIZE_PATH);
        let mut grpc = Grpc::new(channel);
        grpc.ready()
            .await
            .map_err(|e| FlowcatError::Network(format!("google grpc not ready: {e}")))?;
        let response = grpc
            .streaming(request, path, SpeechCodec)
            .await
            .map_err(|e| FlowcatError::Network(format!("google streaming: {e}")))?;

        // 5. Spawn the reader: decode each response message → frames → channel.
        let (frame_tx, frame_rx) = mpsc::unbounded_channel();
        let reader = tokio::spawn(reader_task(response.into_inner(), frame_tx));

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
        let conn = self
            .conn
            .as_mut()
            .ok_or_else(|| FlowcatError::Network("google STT: run_stt before start".into()))?;
        // PCM i16 → little-endian bytes (the wire format Google expects for LINEAR16).
        let mut pcm_le = Vec::with_capacity(audio.pcm.len() * 2);
        for s in &audio.pcm {
            pcm_le.extend_from_slice(&s.to_le_bytes());
        }
        conn.audio_tx
            .send(encode_audio_request(&pcm_le))
            .map_err(|_| FlowcatError::Network("google STT: stream closed".into()))?;
        // Non-blocking drain of whatever the reader has decoded so far.
        let mut out = Vec::new();
        while let Ok(f) = conn.frames.try_recv() {
            out.push(f);
        }
        Ok(out)
    }

    async fn set_muted(&mut self, muted: bool) {
        self.muted = muted;
    }
}

/// Read the inbound `StreamingRecognizeResponse` stream, decoding each message into
/// transcription [`Frame`]s and forwarding them to `tx`. Ends on stream close/error
/// (the per-chunk `run_stt` drains whatever has arrived).
async fn reader_task(mut inbound: tonic::Streaming<Vec<u8>>, tx: mpsc::UnboundedSender<Frame>) {
    loop {
        match inbound.message().await {
            Ok(Some(bytes)) => {
                for frame in decode_response(&bytes) {
                    if tx.send(frame).is_err() {
                        return; // consumer gone
                    }
                }
            }
            Ok(None) => break,     // server closed the stream
            Err(_status) => break, // transport/status error → end the reader
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_request_roundtrips_top_level_fields() {
        let bytes = encode_config_request(
            "projects/p/locations/global/recognizers/_",
            "en-US",
            "long",
            16_000,
            true,
        );
        // recognizer = field 3 (string).
        assert_eq!(
            grpc_proto::first_string(&bytes, 3).as_deref(),
            Some("projects/p/locations/global/recognizers/_")
        );
        // streaming_config = field 6 (message) is present.
        let sc = grpc_proto::iter_field_bytes(&bytes, 6)
            .next()
            .expect("streaming_config");
        // config = field 1 inside streaming_config.
        let rc = grpc_proto::iter_field_bytes(&sc, 1).next().expect("config");
        assert_eq!(grpc_proto::first_string(&rc, 9).as_deref(), Some("long"));
        assert_eq!(grpc_proto::first_string(&rc, 10).as_deref(), Some("en-US"));
        // explicit_decoding_config = field 8: encoding=1 LINEAR16, rate=2.
        let edc = grpc_proto::iter_field_bytes(&rc, 8)
            .next()
            .expect("explicit_decoding");
        assert_eq!(grpc_proto::first_varint(&edc, 1), Some(1)); // LINEAR16
        assert_eq!(grpc_proto::first_varint(&edc, 2), Some(16_000));
        // streaming_features = field 2: interim_results = 2 = true.
        let sf = grpc_proto::iter_field_bytes(&sc, 2)
            .next()
            .expect("streaming_features");
        assert_eq!(grpc_proto::first_varint(&sf, 2), Some(1));
    }

    #[test]
    fn audio_request_wraps_pcm_in_field_5() {
        let pcm = vec![1u8, 0, 254, 255];
        let bytes = encode_audio_request(&pcm);
        let got = grpc_proto::iter_field_bytes(&bytes, 5)
            .next()
            .expect("audio");
        assert_eq!(got, pcm);
    }

    #[test]
    fn decode_final_and_interim_results() {
        // Build a StreamingRecognizeResponse by hand from the same encoder.
        let alt = grpc_proto::encode_message(&[Field::string(1, "book a dentist")]);
        let final_result = grpc_proto::encode_message(&[
            Field::bytes(1, alt),
            Field::varint(2, 1), // is_final = true
        ]);
        let alt2 = grpc_proto::encode_message(&[Field::string(1, "for tomorrow")]);
        let interim_result = grpc_proto::encode_message(&[
            Field::bytes(1, alt2),
            // is_final omitted → default false
        ]);
        let response = grpc_proto::encode_message(&[
            Field::bytes(6, final_result),
            Field::bytes(6, interim_result),
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
    fn decode_ignores_empty_transcripts_and_no_results() {
        assert!(decode_response(&[]).is_empty());
        let alt = grpc_proto::encode_message(&[Field::string(1, "")]);
        let result = grpc_proto::encode_message(&[Field::bytes(1, alt), Field::varint(2, 1)]);
        let response = grpc_proto::encode_message(&[Field::bytes(6, result)]);
        assert!(decode_response(&response).is_empty());
    }

    #[test]
    fn grpc_framing_roundtrips() {
        let msg = encode_audio_request(&[9, 9, 9]);
        let framed = grpc_proto::grpc_frame(&msg);
        // 1-byte flag + 4-byte length prefix.
        assert_eq!(framed[0], 0);
        assert_eq!(framed.len(), msg.len() + 5);
        let back = grpc_proto::grpc_deframe(&framed).expect("deframe");
        assert_eq!(back, msg.as_slice());
        // A short buffer deframes to None.
        assert!(grpc_proto::grpc_deframe(&[0, 0, 0]).is_none());
    }

    #[test]
    fn auth_metadata_is_bearer() {
        let stt = GoogleStt::new("ya29.token", "projects/p/locations/global/recognizers/_");
        let md = stt.auth_metadata();
        assert_eq!(md[0].0, "authorization");
        assert_eq!(md[0].1, "Bearer ya29.token");
    }

    #[test]
    fn endpoint_host_is_global_or_regional() {
        // global recognizer → the global host.
        let g = GoogleStt::new("t", "projects/p/locations/global/recognizers/_");
        assert_eq!(g.endpoint_host(), "speech.googleapis.com");
        // regional recognizer → the regional host (a regional recognizer can ONLY be
        // reached on its regional endpoint — this is the fix for the prior global-only bug).
        let r = GoogleStt::new("t", "projects/p/locations/us-central1/recognizers/_");
        assert_eq!(r.endpoint_host(), "us-central1-speech.googleapis.com");
        // malformed recognizer → falls back to global, never panics.
        let m = GoogleStt::new("t", "garbage");
        assert_eq!(m.endpoint_host(), "speech.googleapis.com");
    }

    #[tokio::test]
    async fn start_rejects_empty_token_without_touching_network() {
        // Deterministic, offline: an empty token is rejected before any connect.
        let mut stt = GoogleStt::new("", "projects/p/locations/global/recognizers/_");
        let err = stt.start(&StartParams::default()).await.unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("token"),
            "got: {err}"
        );
    }

    /// Live smoke (requires `GOOGLE_ACCESS_TOKEN` + `GOOGLE_RECOGNIZER` + network):
    /// connects the bidi stream and pushes a short silence buffer. Run with:
    /// `GOOGLE_ACCESS_TOKEN=… GOOGLE_RECOGNIZER=projects/…/recognizers/_ \
    ///   cargo test -p flowcat-services --features stt-google -- --ignored google_live`
    #[tokio::test]
    #[ignore = "live: needs GOOGLE_ACCESS_TOKEN + GOOGLE_RECOGNIZER + network"]
    async fn google_live_connects_and_streams() {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let token = std::env::var("GOOGLE_ACCESS_TOKEN").expect("GOOGLE_ACCESS_TOKEN");
        let recognizer = std::env::var("GOOGLE_RECOGNIZER").expect("GOOGLE_RECOGNIZER");
        let mut stt = GoogleStt::new(token, recognizer);
        stt.start(&StartParams::default()).await.expect("start");
        let silence = Arc::new(AudioFrame::mono(vec![0i16; 8_000], 16_000));
        let _ = stt.run_stt(silence).await.expect("run_stt");
    }
}
