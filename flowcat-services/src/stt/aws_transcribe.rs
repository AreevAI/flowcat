// SPDX-License-Identifier: Apache-2.0
//
//! **AWS Transcribe** streaming STT.
//!
//! A **(D)istinct** client over AWS Transcribe's streaming-WebSocket API. **Two
//! security-sensitive seams, both hand-rolled — no AWS SDK:**
//!
//! 1. **Connect** with a **SigV4 query-string-presigned** `wss://` URL
//!    ([`presign_transcribe_url`]) — the `transcribe` service, region host
//!    `transcribestreaming.<region>.amazonaws.com:8443`, canonical URI
//!    `/stream-transcription-websocket`, `UNSIGNED-PAYLOAD`-equivalent empty-body
//!    hash, optional `X-Amz-Security-Token` for STS temporary creds.
//! 2. **Frame** audio as AWS **event-stream** `AudioEvent` messages
//!    ([`build_audio_event`]) and **decode** server event-stream messages
//!    ([`decode_event`] → [`transcripts_from_payload`]) into [`Frame`]s. The
//!    event-stream wire format is prelude(8) + prelude-CRC32(4) + headers + payload
//!    + message-CRC32(4); CRC32 is IEEE/zlib ([`crc32`]).
//!
//! Cross-checked against pipecat `services/aws/{stt,utils}.py`. The PCM/auth/decode
//! are **pure functions** with a SigV4 **known-answer test** and event-stream
//! round-trip tests — no live AWS needed. Behind `stt-aws-transcribe`.

use std::sync::Arc;

use async_trait::async_trait;
use futures::{SinkExt, StreamExt};
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;
use tokio::sync::Mutex as AsyncMutex;
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message;

use flowcat_core::error::{FlowcatError, Result};
use flowcat_core::processor::frame::{AudioFrame, Frame, StartParams};
use flowcat_core::service::SttService;

type HmacSha256 = Hmac<Sha256>;

/// AWS credentials for the SigV4 presign. `session_token` is set only for STS
/// temporary credentials (assumed roles, IRSA, instance profiles).
#[derive(Debug, Clone)]
pub struct AwsCredentials {
    pub access_key: String,
    pub secret_key: String,
    pub session_token: Option<String>,
    pub region: String,
}

/// Builder for [`AwsTranscribeStt`]: credentials + region + language + rate.
#[derive(Debug, Clone)]
pub struct AwsTranscribeSttBuilder {
    creds: AwsCredentials,
    language_code: String,
    sample_rate: u32,
}

impl AwsTranscribeSttBuilder {
    /// Start a builder bound to `creds` (default `en-US`, 16 kHz).
    pub fn new(creds: AwsCredentials) -> Self {
        Self {
            creds,
            language_code: "en-US".to_string(),
            sample_rate: 16_000,
        }
    }

    /// Override the transcription language code (e.g. `en-US`, `es-ES`).
    pub fn language_code(mut self, code: impl Into<String>) -> Self {
        self.language_code = code.into();
        self
    }

    /// Override the input sample rate (AWS Transcribe accepts 8000 or 16000;
    /// other values are clamped to 16000 at connect time).
    pub fn sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = rate;
        self
    }

    /// Build the (not-yet-connected) client.
    pub fn build(self) -> AwsTranscribeStt {
        AwsTranscribeStt {
            creds: self.creds,
            language_code: self.language_code,
            sample_rate: self.sample_rate,
            conn: None,
            muted: false,
        }
    }
}

/// The connect-time sample rate AWS will accept (8000 or 16000; else 16000).
fn connect_sample_rate(rate: u32) -> u32 {
    if rate == 8000 || rate == 16000 {
        rate
    } else {
        16000
    }
}

// ---------------------------------------------------------------------------
// SigV4 query-string presign for the Transcribe streaming WebSocket. Hand-rolled
// (HMAC-SHA256 + SHA-256), no AWS SDK.
// SECURITY-REVIEW: this is the auth path; the secret_key never appears in the URL
// (only the derived signature does), and the empty-body hash is signed.
// ---------------------------------------------------------------------------

/// Parameters for [`presign_transcribe_url`] — split out so the signing is a pure,
/// deterministic function (the known-answer test pins it).
pub struct PresignParams<'a> {
    pub creds: &'a AwsCredentials,
    pub language_code: &'a str,
    pub media_encoding: &'a str, // "pcm" for linear16
    pub sample_rate: u32,
    pub number_of_channels: u32,
    /// Timestamp components (`YYYYMMDDTHHMMSSZ`, `YYYYMMDD`) — injected so the test
    /// is deterministic; production passes the current UTC time.
    pub amz_date: &'a str,
    pub date_stamp: &'a str,
    pub expires_secs: u32,
}

/// Build the SigV4-presigned `wss://` URL for AWS Transcribe streaming. **Pure** —
/// the known-answer test drives it. The query string is built in the exact
/// (already-sorted) order AWS Transcribe's sample uses, so it is both the canonical
/// query for signing **and** the URL query (they must match byte-for-byte).
pub fn presign_transcribe_url(p: &PresignParams) -> String {
    let service = "transcribe";
    let host = format!("transcribestreaming.{}.amazonaws.com:8443", p.creds.region);
    let canonical_uri = "/stream-transcription-websocket";
    let signed_headers = "host";
    let algorithm = "AWS4-HMAC-SHA256";

    let credential_scope = format!("{}/{}/{service}/aws4_request", p.date_stamp, p.creds.region);
    let credential = format!("{}/{credential_scope}", p.creds.access_key);

    // Canonical query string. AWS query params sort lexicographically by key; the
    // ordering below is already sorted (X-Amz-* before the lowercase request
    // params). Values are URI-encoded (RFC 3986, '/' escaped in query values).
    let mut qs = String::new();
    qs.push_str(&format!("X-Amz-Algorithm={}", uri_encode(algorithm, true)));
    qs.push_str(&format!(
        "&X-Amz-Credential={}",
        uri_encode(&credential, true)
    ));
    qs.push_str(&format!("&X-Amz-Date={}", uri_encode(p.amz_date, true)));
    qs.push_str(&format!("&X-Amz-Expires={}", p.expires_secs));
    if let Some(token) = &p.creds.session_token {
        qs.push_str(&format!(
            "&X-Amz-Security-Token={}",
            uri_encode(token, true)
        ));
    }
    qs.push_str(&format!(
        "&X-Amz-SignedHeaders={}",
        uri_encode(signed_headers, true)
    ));
    // Request params (already in lexicographic order: language-code, media-encoding,
    // sample-rate). number-of-channels only if > 1.
    qs.push_str(&format!(
        "&language-code={}",
        uri_encode(p.language_code, true)
    ));
    qs.push_str(&format!(
        "&media-encoding={}",
        uri_encode(p.media_encoding, true)
    ));
    if p.number_of_channels > 1 {
        qs.push_str(&format!("&number-of-channels={}", p.number_of_channels));
    }
    qs.push_str(&format!("&sample-rate={}", p.sample_rate));

    let canonical_headers = format!("host:{host}\n");
    let payload_hash = hex(sha256(b""));
    let canonical_request = format!(
        "GET\n{canonical_uri}\n{qs}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
    );

    let string_to_sign = format!(
        "{algorithm}\n{}\n{credential_scope}\n{}",
        p.amz_date,
        hex(sha256(canonical_request.as_bytes()))
    );

    let signature = hex(signing_signature(
        &p.creds.secret_key,
        p.date_stamp,
        &p.creds.region,
        service,
        &string_to_sign,
    ));

    format!("wss://{host}{canonical_uri}?{qs}&X-Amz-Signature={signature}")
}

fn signing_signature(
    secret: &str,
    date_stamp: &str,
    region: &str,
    service: &str,
    string_to_sign: &str,
) -> Vec<u8> {
    let k_date = hmac(format!("AWS4{secret}").as_bytes(), date_stamp.as_bytes());
    let k_region = hmac(&k_date, region.as_bytes());
    let k_service = hmac(&k_region, service.as_bytes());
    let k_signing = hmac(&k_service, b"aws4_request");
    hmac(&k_signing, string_to_sign.as_bytes())
}

fn hmac(key: &[u8], data: &[u8]) -> Vec<u8> {
    let mut mac = HmacSha256::new_from_slice(key).expect("hmac accepts any key length");
    mac.update(data);
    mac.finalize().into_bytes().to_vec()
}

fn sha256(data: &[u8]) -> Vec<u8> {
    let mut h = Sha256::new();
    h.update(data);
    h.finalize().to_vec()
}

fn hex(bytes: Vec<u8>) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// RFC 3986 URI encoding. With `encode_slash = true` everything non-unreserved is
/// escaped (query-value rules); `false` leaves '/' intact (path segments).
fn uri_encode(input: &str, encode_slash: bool) -> String {
    let mut out = String::with_capacity(input.len());
    for byte in input.bytes() {
        let c = byte as char;
        if c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '~' | '.') {
            out.push(c);
        } else if c == '/' && !encode_slash {
            out.push('/');
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// AWS event-stream framing. Pure; round-trip tested.
// ---------------------------------------------------------------------------

/// IEEE/zlib CRC-32 (same polynomial as Python `binascii.crc32`). Bit-reflected,
/// init 0xFFFFFFFF, final XOR 0xFFFFFFFF. Pure — used for both prelude + message
/// checksums.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// Encode one event-stream header (`name`, string value, type 7). Mirrors the AWS
/// sample's `get_headers`.
fn encode_header(name: &str, value: &str) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(name.len() as u8);
    out.extend_from_slice(name.as_bytes());
    out.push(7u8); // value type 7 = string
    out.extend_from_slice(&(value.len() as u16).to_be_bytes());
    out.extend_from_slice(value.as_bytes());
    out
}

/// Build an AWS event-stream `AudioEvent` message wrapping `payload` (raw PCM
/// bytes). Layout: prelude(total_len:u32, headers_len:u32) + prelude_crc(u32) +
/// headers + payload + message_crc(u32). **Pure** — round-trip tested.
pub fn build_audio_event(payload: &[u8]) -> Vec<u8> {
    let mut headers = Vec::new();
    headers.extend_from_slice(&encode_header(":content-type", "application/octet-stream"));
    headers.extend_from_slice(&encode_header(":event-type", "AudioEvent"));
    headers.extend_from_slice(&encode_header(":message-type", "event"));

    // 16 = 8-byte prelude + 4-byte prelude CRC + 4-byte message CRC.
    let total_len = (headers.len() + payload.len() + 16) as u32;
    let headers_len = headers.len() as u32;

    let mut prelude = Vec::with_capacity(8);
    prelude.extend_from_slice(&total_len.to_be_bytes());
    prelude.extend_from_slice(&headers_len.to_be_bytes());
    let prelude_crc = crc32(&prelude);

    let mut msg = Vec::with_capacity(total_len as usize);
    msg.extend_from_slice(&prelude);
    msg.extend_from_slice(&prelude_crc.to_be_bytes());
    msg.extend_from_slice(&headers);
    msg.extend_from_slice(payload);
    let message_crc = crc32(&msg);
    msg.extend_from_slice(&message_crc.to_be_bytes());
    msg
}

/// Parsed event-stream message: string headers + raw payload bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventMessage {
    pub headers: Vec<(String, String)>,
    pub payload: Vec<u8>,
}

impl EventMessage {
    /// Header value by name (e.g. `":message-type"`).
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
}

/// Decode one AWS event-stream message into headers + payload, verifying both
/// CRC-32 checksums. Only string-typed (type 7) headers are decoded (every header
/// AWS Transcribe sends on the response path is a string). **Pure.**
pub fn decode_event(msg: &[u8]) -> Result<EventMessage> {
    if msg.len() < 16 {
        return Err(FlowcatError::Protocol(
            "aws_transcribe: event-stream message too short".into(),
        ));
    }
    let total_len = u32::from_be_bytes([msg[0], msg[1], msg[2], msg[3]]) as usize;
    let headers_len = u32::from_be_bytes([msg[4], msg[5], msg[6], msg[7]]) as usize;
    if total_len != msg.len() {
        return Err(FlowcatError::Protocol(
            "aws_transcribe: event-stream length mismatch".into(),
        ));
    }
    let prelude_crc = u32::from_be_bytes([msg[8], msg[9], msg[10], msg[11]]);
    if crc32(&msg[0..8]) != prelude_crc {
        return Err(FlowcatError::Protocol(
            "aws_transcribe: prelude CRC mismatch".into(),
        ));
    }
    let msg_crc = u32::from_be_bytes([
        msg[total_len - 4],
        msg[total_len - 3],
        msg[total_len - 2],
        msg[total_len - 1],
    ]);
    if crc32(&msg[0..total_len - 4]) != msg_crc {
        return Err(FlowcatError::Protocol(
            "aws_transcribe: message CRC mismatch".into(),
        ));
    }

    let headers_start = 12;
    let headers_end = headers_start + headers_len;
    if headers_end > total_len - 4 {
        return Err(FlowcatError::Protocol(
            "aws_transcribe: headers overrun".into(),
        ));
    }
    let mut headers = Vec::new();
    let mut i = headers_start;
    while i < headers_end {
        let name_len = msg[i] as usize;
        i += 1;
        if i + name_len > headers_end {
            break;
        }
        let name = String::from_utf8_lossy(&msg[i..i + name_len]).to_string();
        i += name_len;
        if i >= headers_end {
            break;
        }
        let value_type = msg[i];
        i += 1;
        // We only need string (type 7) headers; bail on anything else defensively.
        if value_type != 7 {
            break;
        }
        if i + 2 > headers_end {
            break;
        }
        let value_len = u16::from_be_bytes([msg[i], msg[i + 1]]) as usize;
        i += 2;
        if i + value_len > headers_end {
            break;
        }
        let value = String::from_utf8_lossy(&msg[i..i + value_len]).to_string();
        i += value_len;
        headers.push((name, value));
    }

    let payload = msg[headers_end..total_len - 4].to_vec();
    Ok(EventMessage { headers, payload })
}

/// Turn a decoded Transcribe `TranscriptEvent` JSON payload into transcription
/// frames. Shape: `{ "Transcript": { "Results": [ { "Alternatives": [ {
/// "Transcript": "…" } ], "IsPartial": false } ] } }`. Empty transcripts → nothing.
/// **Pure** — the decode seam the fixture tests drive.
pub fn transcripts_from_payload(payload: &serde_json::Value) -> Vec<Frame> {
    let results = payload
        .get("Transcript")
        .and_then(|t| t.get("Results"))
        .and_then(|r| r.as_array());
    let Some(results) = results else {
        return vec![];
    };
    let mut out = Vec::new();
    let user_id: Arc<str> = Arc::from("user");
    for result in results {
        let transcript = result
            .get("Alternatives")
            .and_then(|a| a.as_array())
            .and_then(|a| a.first())
            .and_then(|alt| alt.get("Transcript"))
            .and_then(|t| t.as_str())
            .unwrap_or("");
        if transcript.is_empty() {
            continue;
        }
        // `IsPartial` defaults to true (an interim) when absent.
        let is_partial = result
            .get("IsPartial")
            .and_then(|p| p.as_bool())
            .unwrap_or(true);
        if is_partial {
            out.push(Frame::InterimTranscription {
                text: transcript.to_string(),
                user_id: user_id.clone(),
                language: None,
            });
        } else {
            out.push(Frame::Transcription {
                text: transcript.to_string(),
                user_id: user_id.clone(),
                language: None,
                final_: true,
            });
        }
    }
    out
}

/// Decode a full server event-stream message into transcription frames: parse the
/// envelope, route by `:message-type`, and (for `event`s) extract transcripts.
/// `exception` messages surface as an error; other types yield nothing. **Pure.**
pub fn decode_server_message(msg: &[u8]) -> Result<Vec<Frame>> {
    let event = decode_event(msg)?;
    match event.header(":message-type") {
        Some("event") => {
            let payload: serde_json::Value = serde_json::from_slice(&event.payload)?;
            Ok(transcripts_from_payload(&payload))
        }
        Some("exception") => {
            let payload: serde_json::Value =
                serde_json::from_slice(&event.payload).unwrap_or_default();
            let m = payload
                .get("Message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown");
            Err(FlowcatError::Network(format!(
                "aws_transcribe exception: {m}"
            )))
        }
        _ => Ok(vec![]),
    }
}

// ---------------------------------------------------------------------------
// Live client (transport).
// ---------------------------------------------------------------------------

type ClientSocket =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// Live socket state, present once [`SttService::start`] connected.
struct Connection {
    sink: Arc<AsyncMutex<futures::stream::SplitSink<ClientSocket, Message>>>,
    frames: mpsc::UnboundedReceiver<Frame>,
    reader: JoinHandle<()>,
}

impl Drop for Connection {
    fn drop(&mut self) {
        self.reader.abort();
    }
}

/// An AWS Transcribe streaming-STT session.
pub struct AwsTranscribeStt {
    creds: AwsCredentials,
    language_code: String,
    sample_rate: u32,
    conn: Option<Connection>,
    muted: bool,
}

impl AwsTranscribeStt {
    /// Construct with `en-US` / 16 kHz defaults. Use [`AwsTranscribeSttBuilder`]
    /// for non-default settings.
    pub fn new(creds: AwsCredentials) -> Self {
        AwsTranscribeSttBuilder::new(creds).build()
    }

    /// Current UTC time as the `(amz_date, date_stamp)` SigV4 needs.
    fn now_stamps() -> (String, String) {
        // Seconds since the Unix epoch → broken-down UTC, formatted by hand to
        // avoid pulling `chrono` (not declared for this feature).
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let (y, mo, d, h, mi, s) = civil_from_unix(secs as i64);
        (
            format!("{y:04}{mo:02}{d:02}T{h:02}{mi:02}{s:02}Z"),
            format!("{y:04}{mo:02}{d:02}"),
        )
    }

    /// Open the socket (SigV4-presigned URL) + spawn the event-stream decode reader.
    async fn open(&mut self) -> Result<()> {
        let rate = connect_sample_rate(self.sample_rate);
        let (amz_date, date_stamp) = Self::now_stamps();
        let url = presign_transcribe_url(&PresignParams {
            creds: &self.creds,
            language_code: &self.language_code,
            media_encoding: "pcm",
            sample_rate: rate,
            number_of_channels: 1,
            amz_date: &amz_date,
            date_stamp: &date_stamp,
            expires_secs: 300,
        });
        let request = url
            .into_client_request()
            .map_err(|e| FlowcatError::Network(format!("aws_transcribe url: {e}")))?;
        let (socket, _resp) = tokio_tungstenite::connect_async(request)
            .await
            .map_err(|e| FlowcatError::Network(format!("aws_transcribe connect: {e}")))?;
        let (sink, stream) = socket.split();
        let (tx, rx) = mpsc::unbounded_channel();
        let reader = tokio::spawn(reader_task(stream, tx));
        self.conn = Some(Connection {
            sink: Arc::new(AsyncMutex::new(sink)),
            frames: rx,
            reader,
        });
        Ok(())
    }

    /// PCM → little-endian bytes (AWS `pcm`/linear16 is 16-bit LE).
    fn pcm_bytes(audio: &AudioFrame) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(audio.pcm.len() * 2);
        for s in &audio.pcm {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        bytes
    }
}

#[async_trait]
impl SttService for AwsTranscribeStt {
    fn name(&self) -> &str {
        "aws_transcribe"
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        self.open().await
    }

    async fn run_stt(&mut self, audio: Arc<AudioFrame>) -> Result<Vec<Frame>> {
        if self.muted {
            return Ok(vec![]);
        }
        let conn = self
            .conn
            .as_mut()
            .ok_or_else(|| FlowcatError::Network("aws_transcribe: run_stt before start".into()))?;
        // Wrap this chunk's PCM in an AudioEvent event-stream message.
        let pcm = Self::pcm_bytes(&audio);
        let event = build_audio_event(&pcm);
        {
            let mut sink = conn.sink.lock().await;
            sink.send(Message::binary(event))
                .await
                .map_err(|e| FlowcatError::Network(format!("aws_transcribe send: {e}")))?;
        }
        // Drain whatever the reader has decoded so far (the reader runs ahead).
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

/// The persistent reader: decode each binary event-stream message into
/// transcription [`Frame`]s and queue them. Mirrors the Deepgram reader-task.
async fn reader_task(
    mut stream: futures::stream::SplitStream<ClientSocket>,
    tx: mpsc::UnboundedSender<Frame>,
) {
    while let Some(msg) = stream.next().await {
        let bytes = match msg {
            Ok(Message::Binary(b)) => b,
            Ok(Message::Ping(_)) | Ok(Message::Pong(_)) | Ok(Message::Text(_)) => continue,
            Ok(Message::Close(_)) | Err(_) => break,
            Ok(_) => continue,
        };
        match decode_server_message(&bytes) {
            Ok(frames) => {
                for f in frames {
                    if tx.send(f).is_err() {
                        return; // consumer gone
                    }
                }
            }
            // A decode/exception error ends the reader (the socket is unusable).
            Err(_) => break,
        }
    }
}

/// Convert a Unix timestamp (seconds) to a civil UTC `(year, month, day, hour,
/// minute, second)`. Howard Hinnant's `civil_from_days` algorithm — pure + no deps.
fn civil_from_unix(unix_secs: i64) -> (i64, u32, u32, u32, u32, u32) {
    let days = unix_secs.div_euclid(86_400);
    let secs_of_day = unix_secs.rem_euclid(86_400);
    let h = (secs_of_day / 3600) as u32;
    let mi = ((secs_of_day % 3600) / 60) as u32;
    let s = (secs_of_day % 60) as u32;

    // days since 1970-01-01 → civil date.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    let year = if m <= 2 { y + 1 } else { y };
    (year, m, d, h, mi, s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn creds() -> AwsCredentials {
        AwsCredentials {
            access_key: "AKIDEXAMPLE".into(),
            secret_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".into(),
            session_token: None,
            region: "us-east-1".into(),
        }
    }

    /// SigV4 known-answer test: a fixed time + fixed credentials must produce a
    /// byte-stable presigned URL. This pins the canonical-request / string-to-sign /
    /// signing-key derivation (the security-critical path). If the signing changes,
    /// this signature changes — a deliberate tripwire for the security review.
    #[test]
    fn presign_known_answer_is_stable() {
        let url = presign_transcribe_url(&PresignParams {
            creds: &creds(),
            language_code: "en-US",
            media_encoding: "pcm",
            sample_rate: 16000,
            number_of_channels: 1,
            amz_date: "20260101T000000Z",
            date_stamp: "20260101",
            expires_secs: 300,
        });
        // Host + path + the non-secret query are exact.
        assert!(url.starts_with(
            "wss://transcribestreaming.us-east-1.amazonaws.com:8443/stream-transcription-websocket?"
        ));
        assert!(url.contains("X-Amz-Algorithm=AWS4-HMAC-SHA256"));
        assert!(url.contains(
            "X-Amz-Credential=AKIDEXAMPLE%2F20260101%2Fus-east-1%2Ftranscribe%2Faws4_request"
        ));
        assert!(url.contains("X-Amz-Date=20260101T000000Z"));
        assert!(url.contains("X-Amz-Expires=300"));
        assert!(url.contains("X-Amz-SignedHeaders=host"));
        assert!(url.contains("language-code=en-US"));
        assert!(url.contains("media-encoding=pcm"));
        assert!(url.contains("sample-rate=16000"));
        // The secret never appears in the URL — only the derived signature does.
        assert!(!url.contains("wJalrXUtnFEMI"));
        // Known-answer signature for these exact inputs (regenerate intentionally
        // only if the signing algorithm changes).
        assert!(
            url.contains(
                "X-Amz-Signature=446fa0f7e4ffc124cdd654af46f0630e3fdad959604307ac8b42f54631cdc787"
            ),
            "signature changed — {url}"
        );
    }

    #[test]
    fn presign_includes_security_token_when_present() {
        let mut c = creds();
        c.session_token = Some("FQoGZ+token/with=specials".into());
        let url = presign_transcribe_url(&PresignParams {
            creds: &c,
            language_code: "en-US",
            media_encoding: "pcm",
            sample_rate: 16000,
            number_of_channels: 1,
            amz_date: "20260101T000000Z",
            date_stamp: "20260101",
            expires_secs: 300,
        });
        // Token is URI-encoded into the signed query.
        assert!(url.contains("X-Amz-Security-Token=FQoGZ%2Btoken%2Fwith%3Dspecials"));
    }

    /// CRC32 matches the IEEE/zlib reference (`crc32("123456789") == 0xCBF43926`).
    #[test]
    fn crc32_matches_zlib_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0x0000_0000);
    }

    /// build → decode round-trip recovers the exact payload + headers.
    #[test]
    fn audio_event_roundtrips() {
        let payload = b"\x01\x02\x03\x04rawpcm";
        let msg = build_audio_event(payload);
        let decoded = decode_event(&msg).expect("decode");
        assert_eq!(decoded.payload, payload);
        assert_eq!(decoded.header(":message-type"), Some("event"));
        assert_eq!(decoded.header(":event-type"), Some("AudioEvent"));
        assert_eq!(
            decoded.header(":content-type"),
            Some("application/octet-stream")
        );
    }

    #[test]
    fn decode_rejects_corrupt_crc() {
        let mut msg = build_audio_event(b"abc");
        // Flip a payload byte without fixing the CRC → message CRC check fails.
        let len = msg.len();
        msg[len - 6] ^= 0xFF;
        assert!(decode_event(&msg).is_err());
    }

    /// A server `event` message decodes into the right transcription frames.
    #[test]
    fn server_event_decodes_to_transcripts() {
        let payload = json!({
            "Transcript": { "Results": [
                { "Alternatives": [ { "Transcript": "book a dentist" } ], "IsPartial": false },
                { "Alternatives": [ { "Transcript": "for tomorrow" } ], "IsPartial": true }
            ]}
        });
        let frames = transcripts_from_payload(&payload);
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
    fn empty_and_no_results_yield_nothing() {
        assert!(transcripts_from_payload(&json!({})).is_empty());
        let empty = json!({ "Transcript": { "Results": [
            { "Alternatives": [ { "Transcript": "" } ], "IsPartial": false }
        ]}});
        assert!(transcripts_from_payload(&empty).is_empty());
    }

    /// A full server `event` envelope (the wire bytes) round-trips to frames.
    #[test]
    fn decode_server_message_event_envelope() {
        let payload = json!({
            "Transcript": { "Results": [
                { "Alternatives": [ { "Transcript": "hello" } ], "IsPartial": false }
            ]}
        });
        let body = serde_json::to_vec(&payload).unwrap();
        // Re-frame as an event-stream "event" message (build_audio_event tags it
        // :message-type=event), then decode.
        let msg = build_audio_event(&body);
        let frames = decode_server_message(&msg).expect("decode");
        assert_eq!(frames.len(), 1);
        assert!(matches!(&frames[0], Frame::Transcription { text, .. } if text == "hello"));
    }

    #[test]
    fn pcm_bytes_are_little_endian() {
        let af = AudioFrame::mono(vec![1, -2, 256], 16_000);
        assert_eq!(AwsTranscribeStt::pcm_bytes(&af), vec![1, 0, 254, 255, 0, 1]);
    }

    #[test]
    fn connect_sample_rate_clamps_to_supported() {
        assert_eq!(connect_sample_rate(8000), 8000);
        assert_eq!(connect_sample_rate(16000), 16000);
        assert_eq!(connect_sample_rate(24000), 16000);
        assert_eq!(connect_sample_rate(48000), 16000);
    }

    #[test]
    fn civil_from_unix_known_dates() {
        // 2026-01-01T00:00:00Z = 1767225600.
        assert_eq!(civil_from_unix(1_767_225_600), (2026, 1, 1, 0, 0, 0));
        // Epoch.
        assert_eq!(civil_from_unix(0), (1970, 1, 1, 0, 0, 0));
        // 2024-02-29T12:34:56Z (leap day) = 1709210096.
        assert_eq!(civil_from_unix(1_709_210_096), (2024, 2, 29, 12, 34, 56));
    }

    /// Live smoke (requires real AWS creds + region in env): connect with a
    /// presigned URL, send a beat of silence, confirm the socket opens. Run with:
    /// `AWS_ACCESS_KEY_ID=… AWS_SECRET_ACCESS_KEY=… AWS_REGION=us-east-1 \
    ///   cargo test -p flowcat-services --features stt-aws-transcribe -- --ignored aws_transcribe_live`
    #[tokio::test]
    #[ignore = "requires AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION"]
    async fn aws_transcribe_live_connects_and_streams() {
        let creds = AwsCredentials {
            access_key: std::env::var("AWS_ACCESS_KEY_ID").expect("AWS_ACCESS_KEY_ID"),
            secret_key: std::env::var("AWS_SECRET_ACCESS_KEY").expect("AWS_SECRET_ACCESS_KEY"),
            session_token: std::env::var("AWS_SESSION_TOKEN").ok(),
            region: std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".into()),
        };
        let mut stt = AwsTranscribeStt::new(creds);
        stt.start(&StartParams::default()).await.expect("connect");
        let silence = Arc::new(AudioFrame::mono(vec![0i16; 1600], 16_000));
        let _ = stt.run_stt(silence).await.expect("run_stt");
    }
}
