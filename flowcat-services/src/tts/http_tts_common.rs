// SPDX-License-Identifier: Apache-2.0
//
//! Shared HTTP-cloud TTS transport (Group G).
//!
//! The one transport seam reused by every HTTP-POST-audio TTS provider in this
//! group (openai/groq/xai, hume, inworld, minimax, camb, sarvam, mistral,
//! speechmatics, azure). It is **not** a `mod` in `tts/mod.rs` (which the fan-out
//! must not edit) — each provider file pulls it in with
//! `#[path = "http_tts_common.rs"] mod http;` under
//! `#[allow(clippy::duplicate_mod)]`, so a single-feature build (e.g.
//! `--features tts-hume`) compiles its own private copy and never depends on a
//! sibling provider's feature.
//!
//! Shape (the request/response analogue of the Cartesia WS reference): a provider
//! builds an [`HttpTtsRequest`] (fixed host URL + headers + a JSON or raw body),
//! [`HttpTtsClient::post`] POSTs it and buffers the response body, and the
//! provider's **pure** decoder turns those bytes into audio. The
//! [`TtsService::run_tts`](flowcat_core::service::TtsService::run_tts) contract
//! returns *all* frames for one utterance at once, so buffering the body (rather
//! than streaming) matches the trait exactly and keeps the decode a pure function.
//!
//! Audio comes back in one of a handful of containers; this file owns the pure
//! decoders for each so they are written and tested once:
//!
//! - raw little-endian `pcm_s16le` bytes → [`pcm_from_le_bytes`] (openai/xai/
//!   speechmatics/camb),
//! - a WAV file (RIFF header + PCM) → [`strip_wav_header`] then `pcm_from_le_bytes`
//!   (groq/sarvam),
//! - little-endian **float32** PCM → [`float32_le_to_i16`] (mistral),
//! - base64-of-PCM inside a JSON/JSONL body → [`base64_decode`] (hume/inworld),
//! - a **hex** string of PCM inside an SSE `data:` body → [`hex_to_bytes`]
//!   (minimax).
//!
//! [`tts_frames`] wraps the decoded PCM bytes in the standard
//! [`Frame::TtsStarted`] / [`Frame::TtsAudio`] / [`Frame::TtsStopped`] framing.
//!
//! **Security.** The HTTP response is untrusted: every decoder tolerates any byte
//! shape (a short/odd/garbage body decodes to as many whole samples as it can and
//! never panics or indexes out of bounds), a non-2xx status surfaces as a
//! [`FlowcatError::Network`], and malformed JSON/hex/base64 simply yields no audio.
//! Hosts are fixed by the calling provider; the API key travels only in a header,
//! never in the URL — no SSRF surface.
#![allow(dead_code)]

use std::sync::Arc;

use flowcat_core::error::{FlowcatError, Result};
use flowcat_core::processor::frame::{AudioFrame, Frame};

/// The HTTP request body a provider POSTs: either a JSON value (the common case)
/// or a raw string payload with an explicit content type (Azure SSML).
pub enum HttpTtsBody {
    /// `application/json` — serialized from the provided value.
    Json(serde_json::Value),
    /// A raw text body with a caller-supplied `Content-Type` (e.g.
    /// `application/ssml+xml` for Azure).
    Raw {
        content_type: &'static str,
        body: String,
    },
}

/// A fully-described one-shot TTS HTTP request. The provider supplies the fixed
/// host URL, its auth + format headers, and the body; the key only ever rides a
/// header here (never the URL).
pub struct HttpTtsRequest {
    /// The absolute endpoint URL (host fixed per provider).
    pub url: String,
    /// Extra request headers (auth, output-format, tracking), inserted verbatim.
    pub headers: Vec<(String, String)>,
    /// The request body (JSON or raw).
    pub body: HttpTtsBody,
}

/// A reusable HTTP client for the buffered POST-audio providers. Thin wrapper over
/// a rustls [`reqwest::Client`]; cloning is cheap (the client is `Arc` inside).
#[derive(Clone)]
pub struct HttpTtsClient {
    http: reqwest::Client,
    /// The provider name, for error messages.
    provider: &'static str,
}

impl HttpTtsClient {
    /// A client tagged with `provider` (used only in error messages).
    pub fn new(provider: &'static str) -> Self {
        Self {
            http: reqwest::Client::new(),
            provider,
        }
    }

    /// POST `req` and return the response body bytes. A non-success status is an
    /// error carrying the status + (truncated) body so failures are diagnosable.
    pub async fn post(&self, req: HttpTtsRequest) -> Result<Vec<u8>> {
        let mut builder = self.http.post(&req.url);
        for (name, value) in &req.headers {
            builder = builder.header(name, value);
        }
        builder = match req.body {
            HttpTtsBody::Json(value) => builder
                .header("Content-Type", "application/json")
                .body(serde_json::to_vec(&value).map_err(FlowcatError::from)?),
            HttpTtsBody::Raw { content_type, body } => {
                builder.header("Content-Type", content_type).body(body)
            }
        };

        let resp = builder
            .send()
            .await
            .map_err(|e| FlowcatError::Network(format!("{} tts send: {e}", self.provider)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let mut text = resp.text().await.unwrap_or_default();
            text.truncate(512);
            return Err(FlowcatError::Network(format!(
                "{} tts {status}: {text}",
                self.provider
            )));
        }

        let bytes = resp
            .bytes()
            .await
            .map_err(|e| FlowcatError::Network(format!("{} tts body: {e}", self.provider)))?;
        Ok(bytes.to_vec())
    }
}

/// Frame a single utterance's decoded PCM into the standard
/// `TtsStarted` → `TtsAudio` → `TtsStopped` sequence at `sample_rate`. An empty
/// `pcm` still emits the start/stop framing (so an empty synthesis is observable).
pub fn tts_frames(pcm: Vec<i16>, sample_rate: u32, context_id: Arc<str>) -> Vec<Frame> {
    // Chunk the whole-utterance PCM into ~20 ms `TtsAudio` frames, like the streaming
    // providers do. The transport paces realtime audio in small frames; a single giant
    // frame for the whole utterance is mishandled (only its tail plays).
    let chunk = (sample_rate as usize / 50).max(1); // ~20 ms at `sample_rate`
    let mut out = Vec::with_capacity(pcm.len() / chunk + 2);
    out.push(Frame::TtsStarted {
        context_id: Some(context_id.clone()),
    });
    for samples in pcm.chunks(chunk) {
        out.push(Frame::TtsAudio {
            audio: Arc::new(AudioFrame::mono(samples.to_vec(), sample_rate)),
            context_id: Some(context_id.clone()),
        });
    }
    out.push(Frame::TtsStopped {
        context_id: Some(context_id),
    });
    out
}

/// Decode little-endian i16 PCM bytes into samples (drops a trailing odd byte —
/// never panics on an odd-length body).
pub fn pcm_from_le_bytes(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// If `bytes` begins with a canonical RIFF/WAVE header, return the PCM data after
/// the 44-byte header; otherwise return the bytes unchanged (already-raw PCM).
/// Defensive: a `RIFF`-prefixed but too-short buffer falls through to the original.
pub fn strip_wav_header(bytes: &[u8]) -> &[u8] {
    if bytes.len() > 44 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WAVE" {
        &bytes[44..]
    } else {
        bytes
    }
}

/// Decode little-endian **float32** PCM bytes (samples in `[-1.0, 1.0]`) into
/// clamped i16 samples (Mistral streams float32). A trailing partial sample
/// (`< 4` bytes) is dropped — never panics.
pub fn float32_le_to_i16(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(4)
        .map(|c| {
            let f = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            (f.clamp(-1.0, 1.0) * 32767.0) as i16
        })
        .collect()
}

/// Decode a hex string into bytes (MiniMax returns PCM as a hex string in JSON).
/// Tolerant: a trailing odd nibble or any non-hex character ends the decode and
/// returns what parsed so far — never panics on malformed input.
pub fn hex_to_bytes(hex: &str) -> Vec<u8> {
    let h = hex.as_bytes();
    let mut out = Vec::with_capacity(h.len() / 2);
    let mut i = 0;
    while i + 1 < h.len() {
        let (Some(hi), Some(lo)) = (hex_nibble(h[i]), hex_nibble(h[i + 1])) else {
            break;
        };
        out.push((hi << 4) | lo);
        i += 2;
    }
    out
}

/// Standard (RFC 4648) base64-**encode** — used by the providers' fixture tests
/// to build a known audio body, and available if a provider ever needs to embed
/// PCM in a request. Kept here (the one shared file) so it isn't duplicated.
pub fn base64_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[((n >> 18) & 63) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}

fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Standard (RFC 4648) base64-decode, tolerating padding and embedded whitespace
/// (hume/inworld carry base64 PCM in a JSON field). Invalid input ends the decode
/// gracefully and returns what was parsed — never panics. Kept here (the one
/// shared file) so the HTTP features need not pull the optional `base64` crate.
pub fn base64_decode(input: &str) -> Vec<u8> {
    let mut bits: u32 = 0;
    let mut nbits: u8 = 0;
    let mut out = Vec::with_capacity(input.len() / 4 * 3);
    for &b in input.as_bytes() {
        let v = match b {
            b'A'..=b'Z' => b - b'A',
            b'a'..=b'z' => b - b'a' + 26,
            b'0'..=b'9' => b - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => break,
            b'\r' | b'\n' | b' ' | b'\t' => continue,
            _ => break,
        } as u32;
        bits = (bits << 6) | v;
        nbits += 6;
        if nbits >= 8 {
            nbits -= 8;
            out.push((bits >> nbits) as u8);
        }
    }
    out
}

#[cfg(test)]
mod http_common_tests {
    use super::*;

    #[test]
    fn pcm_from_le_bytes_drops_trailing_odd_byte() {
        // 1, -1, then a stray byte.
        assert_eq!(pcm_from_le_bytes(&[1, 0, 255, 255, 9]), vec![1, -1]);
        assert_eq!(pcm_from_le_bytes(&[]), Vec::<i16>::new());
    }

    #[test]
    fn strip_wav_header_only_when_present() {
        // A minimal RIFF/WAVE header (44 bytes) + 2 PCM bytes.
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&[0u8; 4]);
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(&[0u8; 32]); // pad to 44
        wav.extend_from_slice(&[7, 0]);
        assert_eq!(strip_wav_header(&wav), &[7, 0]);
        // Raw (non-RIFF) bytes are returned unchanged.
        let raw = [1u8, 2, 3, 4];
        assert_eq!(strip_wav_header(&raw), &raw);
        // A truncated RIFF buffer is not mistaken for a header.
        let short = b"RIFFshort";
        assert_eq!(strip_wav_header(short), short);
    }

    #[test]
    fn float32_le_to_i16_clamps_and_scales() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1.0f32.to_le_bytes()); // → 32767
        bytes.extend_from_slice(&(-1.0f32).to_le_bytes()); // → -32767
        bytes.extend_from_slice(&0.0f32.to_le_bytes()); // → 0
        bytes.extend_from_slice(&2.0f32.to_le_bytes()); // clamps → 32767
        bytes.push(0xAB); // trailing partial sample, dropped
        assert_eq!(float32_le_to_i16(&bytes), vec![32767, -32767, 0, 32767]);
    }

    #[test]
    fn hex_to_bytes_decodes_and_tolerates_garbage() {
        assert_eq!(hex_to_bytes("01ff00"), vec![1, 255, 0]);
        assert_eq!(hex_to_bytes("ABCD"), vec![0xAB, 0xCD]);
        // Odd trailing nibble + non-hex char stop the decode gracefully.
        assert_eq!(hex_to_bytes("0102zz"), vec![1, 2]);
        assert_eq!(hex_to_bytes("0"), Vec::<u8>::new());
    }

    #[test]
    fn base64_decode_roundtrips_and_ignores_whitespace() {
        // "Man" → "TWFu"; the classic RFC 4648 vectors.
        assert_eq!(base64_decode("TWFu"), b"Man");
        assert_eq!(base64_decode("TWE="), b"Ma");
        assert_eq!(base64_decode("TQ=="), b"M");
        // Embedded newlines (JSONL wrapping) are tolerated.
        assert_eq!(base64_decode("TW\nFu"), b"Man");
        // Garbage ends the decode without panicking.
        assert_eq!(base64_decode(""), Vec::<u8>::new());
    }

    #[test]
    fn base64_encode_matches_known_vectors_and_roundtrips() {
        assert_eq!(base64_encode(b"Man"), "TWFu");
        assert_eq!(base64_encode(b"Ma"), "TWE=");
        assert_eq!(base64_encode(b"M"), "TQ==");
        let bytes = [0u8, 1, 2, 254, 255, 128, 42];
        assert_eq!(base64_decode(&base64_encode(&bytes)), bytes);
    }

    #[test]
    fn tts_frames_wraps_start_audio_stop() {
        let ctx: Arc<str> = Arc::from("ctx-1");
        let frames = tts_frames(vec![1, -1], 24_000, ctx.clone());
        assert!(matches!(frames[0], Frame::TtsStarted { .. }));
        match &frames[1] {
            Frame::TtsAudio { audio, context_id } => {
                assert_eq!(audio.pcm, vec![1, -1]);
                assert_eq!(audio.sample_rate, 24_000);
                assert_eq!(context_id.as_deref(), Some("ctx-1"));
            }
            other => panic!("expected TtsAudio, got {}", other.name()),
        }
        assert!(matches!(frames[2], Frame::TtsStopped { .. }));
    }

    #[test]
    fn tts_frames_empty_pcm_still_frames_start_stop() {
        let ctx: Arc<str> = Arc::from("ctx-2");
        let frames = tts_frames(vec![], 24_000, ctx);
        assert_eq!(frames.len(), 2);
        assert!(matches!(frames[0], Frame::TtsStarted { .. }));
        assert!(matches!(frames[1], Frame::TtsStopped { .. }));
    }
}
