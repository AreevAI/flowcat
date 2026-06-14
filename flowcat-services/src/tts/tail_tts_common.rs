// SPDX-License-Identifier: Apache-2.0
//
//! TTS-tail shared helpers (pure functions only).
//!
//! **Group-H–local.** This file is `#[path]`-included into each Group-H provider
//! module (`fish`, `lmnt`, `neuphonic`, `smallest`, `kokoro`, `piper`, `xtts`,
//! `aws_polly`, `google`, `nvidia`) so the wire/codec seams below are written once
//! and unit-tested once, without adding a `mod`/`pub use` line to `tts/mod.rs` or a
//! Cargo dependency (the Group-H features enable only `reqwest`/`tokio` — plus
//! `hmac`/`sha2` for Polly and `tonic` for the gRPC pair — so we hand-roll the few
//! codec primitives; `base64`/`tokio-tungstenite` are *not* on these features).
//!
//! Because it is included into several modules, each provider compiles its own copy
//! (the classic "header" pattern); `#[allow(dead_code)]` keeps clippy quiet when a
//! given provider only uses a subset.

#![allow(dead_code)]

use std::sync::Arc;

use flowcat_core::error::FlowcatError;
use flowcat_core::processor::frame::{AudioFrame, Frame};

/// Decode little-endian i16 PCM bytes into samples (drops a trailing odd byte).
pub fn pcm_s16le(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// If `bytes` begins with a RIFF/WAVE header, return the slice that starts at the
/// `data` chunk's payload; otherwise return `bytes` unchanged. Used by the HTTP
/// providers that return a one-shot WAV body (piper-http, smallest REST, polly-wav)
/// when raw PCM is wanted. Pure — no allocation.
pub fn strip_wav_header(bytes: &[u8]) -> &[u8] {
    // "RIFF"<size:4>"WAVE" then a sequence of <id:4><size:4><payload> chunks.
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return bytes;
    }
    let mut i = 12usize;
    while i + 8 <= bytes.len() {
        let id = &bytes[i..i + 4];
        let size =
            u32::from_le_bytes([bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]]) as usize;
        let payload_start = i + 8;
        if id == b"data" {
            let end = payload_start.saturating_add(size).min(bytes.len());
            return &bytes[payload_start..end];
        }
        // Chunks are word-aligned: a padding byte follows an odd size.
        i = payload_start + size + (size & 1);
    }
    bytes
}

/// Wrap raw little-endian i16 PCM bytes at `rate` Hz into a single
/// [`Frame::TtsAudio`] tagged with `context_id`. Returns `None` for empty audio.
pub fn pcm_audio_frame(bytes: &[u8], rate: u32, context_id: &Arc<str>) -> Option<Frame> {
    let pcm = pcm_s16le(bytes);
    if pcm.is_empty() {
        return None;
    }
    Some(Frame::TtsAudio {
        audio: Arc::new(AudioFrame::mono(pcm, rate)),
        context_id: Some(context_id.clone()),
    })
}

/// Frame a one-shot synthesis (`TtsStarted` → [one `TtsAudio`] → `TtsStopped`).
/// The single helper every HTTP/local provider's `run_tts` funnels through, so the
/// framing is identical across the tail.
pub fn one_shot_frames(pcm_bytes: &[u8], rate: u32, context_id: Arc<str>) -> Vec<Frame> {
    let mut out = vec![Frame::TtsStarted {
        context_id: Some(context_id.clone()),
    }];
    if let Some(audio) = pcm_audio_frame(pcm_bytes, rate, &context_id) {
        out.push(audio);
    }
    out.push(Frame::TtsStopped {
        context_id: Some(context_id),
    });
    out
}

/// Decode standard-alphabet base64 (`A–Z a–z 0–9 + /`, `=` padding). A tiny
/// self-contained decoder so the base64 SSE/JSON-audio providers
/// (neuphonic/smallest-REST/google-REST) don't need the `base64` crate (which is
/// not on the Group-H feature set). Whitespace is ignored. Returns a protocol error
/// on an invalid character.
pub fn b64_decode(input: &str) -> Result<Vec<u8>, FlowcatError> {
    fn val(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let mut out = Vec::with_capacity(input.len() / 4 * 3);
    let mut acc: u32 = 0;
    let mut bits = 0u32;
    for &c in input.as_bytes() {
        if c == b'=' || c == b'\n' || c == b'\r' || c == b' ' || c == b'\t' {
            continue;
        }
        let v = val(c).ok_or_else(|| FlowcatError::Protocol(format!("base64: bad char {c:#x}")))?;
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Ok(out)
}

// --- Minimal protobuf wire-format encoders (the gRPC pair: google / nvidia) ---
//
// Hand-rolled because the Group-H gRPC features enable `tonic` only (no
// `tonic-build`/`prost` codegen — that would need a build.rs + a build-dependency,
// which Group H must not add to Cargo.toml). These build the *request message bytes*
// so the request shape is unit-tested; the live transport is the (tonic) seam.

/// Append a base-128 varint (protobuf `varint`).
pub fn pb_varint(out: &mut Vec<u8>, mut v: u64) {
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

/// Append a protobuf field tag (`field_number << 3 | wire_type`).
pub fn pb_tag(out: &mut Vec<u8>, field: u32, wire_type: u32) {
    pb_varint(out, ((field << 3) | wire_type) as u64);
}

/// Append a length-delimited field (wire type 2): a string/bytes/embedded message.
pub fn pb_len_delim(out: &mut Vec<u8>, field: u32, bytes: &[u8]) {
    pb_tag(out, field, 2);
    pb_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

/// Append a varint scalar field (wire type 0): int32/enum/bool.
pub fn pb_varint_field(out: &mut Vec<u8>, field: u32, value: u64) {
    pb_tag(out, field, 0);
    pb_varint(out, value);
}

/// Wrap a protobuf message body in the **gRPC length-prefixed frame** (1 byte
/// compressed-flag = 0, then a 4-byte big-endian length, then the message). This is
/// the body a unary gRPC POST carries — the request seam for the gRPC live path.
pub fn grpc_frame(message: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(5 + message.len());
    out.push(0); // not compressed
    out.extend_from_slice(&(message.len() as u32).to_be_bytes());
    out.extend_from_slice(message);
    out
}

/// Split an HTTP **SSE** body into the data payload of each `data:` event. Multiple
/// `data:` lines within one event are concatenated (per the SSE spec); events are
/// separated by a blank line. Comment lines (`:`) and other fields are ignored.
/// Pure — the seam for the SSE providers (neuphonic-http).
pub fn sse_data_events(body: &str) -> Vec<String> {
    let mut events = Vec::new();
    let mut cur = String::new();
    let mut have = false;
    for raw in body.split('\n') {
        let line = raw.strip_suffix('\r').unwrap_or(raw);
        if line.is_empty() {
            if have {
                events.push(std::mem::take(&mut cur));
                have = false;
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix("data:") {
            if have {
                cur.push('\n');
            }
            cur.push_str(rest.strip_prefix(' ').unwrap_or(rest));
            have = true;
        }
        // Non-data fields / comments are ignored.
    }
    if have {
        events.push(cur);
    }
    events
}

#[cfg(test)]
mod tail_common_tests {
    use super::*;

    #[test]
    fn pcm_s16le_drops_trailing_odd_byte() {
        assert_eq!(pcm_s16le(&[1, 0, 2, 0, 99]), vec![1, 2]);
    }

    #[test]
    fn strip_wav_header_finds_data_chunk() {
        // RIFF<4>WAVE  fmt <16-byte chunk>  data<size=4>  payload(4 bytes)
        let mut w = Vec::new();
        w.extend_from_slice(b"RIFF");
        w.extend_from_slice(&[0, 0, 0, 0]);
        w.extend_from_slice(b"WAVE");
        w.extend_from_slice(b"fmt ");
        w.extend_from_slice(&16u32.to_le_bytes());
        w.extend_from_slice(&[0u8; 16]);
        w.extend_from_slice(b"data");
        w.extend_from_slice(&4u32.to_le_bytes());
        w.extend_from_slice(&[1, 0, 2, 0]);
        assert_eq!(strip_wav_header(&w), &[1, 0, 2, 0]);
    }

    #[test]
    fn strip_wav_header_passes_through_raw_pcm() {
        let raw = [9u8, 0, 8, 0];
        assert_eq!(strip_wav_header(&raw), &raw);
    }

    #[test]
    fn one_shot_frames_brackets_audio() {
        let ctx: Arc<str> = Arc::from("ctx-1");
        // Two LE samples: 1 and -1.
        let frames = one_shot_frames(&[1, 0, 255, 255], 24_000, ctx);
        assert!(matches!(frames[0], Frame::TtsStarted { .. }));
        match &frames[1] {
            Frame::TtsAudio { audio, .. } => assert_eq!(audio.pcm, vec![1, -1]),
            _ => panic!("expected TtsAudio"),
        }
        assert!(matches!(frames[2], Frame::TtsStopped { .. }));
    }

    #[test]
    fn one_shot_frames_skips_empty_audio() {
        let ctx: Arc<str> = Arc::from("ctx-1");
        let frames = one_shot_frames(&[], 24_000, ctx);
        assert_eq!(frames.len(), 2); // started + stopped, no audio
    }

    #[test]
    fn b64_decode_roundtrips_known_vectors() {
        assert_eq!(b64_decode("").unwrap(), b"");
        assert_eq!(b64_decode("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(b64_decode("aGVsbG8gd29ybGQ=").unwrap(), b"hello world");
        // No-padding form decodes identically (padding is ignored).
        assert_eq!(b64_decode("aGVsbG8").unwrap(), b"hello");
        // Whitespace inside the stream is skipped.
        assert_eq!(b64_decode("aGVs\nbG8=").unwrap(), b"hello");
    }

    #[test]
    fn b64_decode_rejects_bad_char() {
        assert!(b64_decode("aGVsbG8*").is_err());
    }

    #[test]
    fn sse_data_events_splits_and_concatenates() {
        let body = "data: one\n\ndata: two-a\ndata: two-b\n\n: comment\ndata:three\n\n";
        let evs = sse_data_events(body);
        assert_eq!(evs, vec!["one", "two-a\ntwo-b", "three"]);
    }

    #[test]
    fn pb_varint_encodes_known_values() {
        let mut o = Vec::new();
        pb_varint(&mut o, 1);
        assert_eq!(o, vec![0x01]);
        let mut o = Vec::new();
        pb_varint(&mut o, 300);
        assert_eq!(o, vec![0xac, 0x02]); // canonical protobuf example
    }

    #[test]
    fn pb_len_delim_field_layout() {
        // field 1, string "abc" → tag 0x0a, len 3, "abc".
        let mut o = Vec::new();
        pb_len_delim(&mut o, 1, b"abc");
        assert_eq!(o, vec![0x0a, 0x03, b'a', b'b', b'c']);
    }

    #[test]
    fn pb_varint_field_layout() {
        // field 4, value 16000 → tag 0x20, varint(16000).
        let mut o = Vec::new();
        pb_varint_field(&mut o, 4, 16_000);
        assert_eq!(o, vec![0x20, 0x80, 0x7d]);
    }

    #[test]
    fn grpc_frame_prefixes_length() {
        let frame = grpc_frame(&[1, 2, 3]);
        assert_eq!(frame, vec![0, 0, 0, 0, 3, 1, 2, 3]);
    }
}
