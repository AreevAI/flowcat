// SPDX-License-Identifier: Apache-2.0
//
//! Shared segmented-HTTP STT helpers.
//!
//! The transport seam reused by the REST/segmented STT providers
//! (elevenlabs, sarvam, mistral). Like the WebSocket helper it is **not** a `mod` in
//! `stt/mod.rs` — each provider pulls it in with
//! `#[path = "rest_stt_common.rs"] mod rest;`, so a single-feature build compiles
//! its own private copy with no cross-provider dependency.
//!
//! These providers transcribe a whole *segment* of audio at once (the
//! Whisper-family / `SegmentedSTTService` shape): they buffer PCM across
//! `run_stt` calls and, once a segment's worth has accumulated, POST it as a WAV
//! file and decode the JSON response. This file provides:
//!
//! - [`SegmentBuffer`] — accumulates PCM and tells the caller when a segment is
//!   ready to flush (a size threshold, since the [`SttService`] trait carries no
//!   explicit end-of-turn signal).
//! - [`wav_pcm16_mono`] — wrap raw little-endian PCM in a 44-byte RIFF/WAVE
//!   header (dependency-free; the providers' upload endpoints expect an audio
//!   *file*, not raw samples).
//! - [`multipart_body`] — build a `multipart/form-data` body by hand (the crate's
//!   `reqwest` is built without the `multipart` feature, by design).
//!
//! **Security.** The HTTP response is untrusted JSON — the providers' decoders
//! read it with `serde_json` and tolerate any shape (missing/!string `text` →
//! empty transcript → no frame), never panicking. Hosts are fixed per provider;
//! the API key travels only in a header, never in the URL.
#![allow(dead_code)]

use flowcat_core::processor::frame::AudioFrame;

/// Accumulates PCM across `run_stt` chunks and flags when a segment is ready.
pub struct SegmentBuffer {
    pcm: Vec<i16>,
    sample_rate: u32,
    /// Flush once at least this many samples have accumulated.
    flush_after_samples: usize,
}

impl SegmentBuffer {
    /// A buffer that flushes after roughly `secs` seconds of audio at
    /// `sample_rate` (the segment length POSTed in one request).
    pub fn new(sample_rate: u32, secs: f32) -> Self {
        let flush_after_samples = ((sample_rate as f32) * secs).max(1.0) as usize;
        Self {
            pcm: Vec::new(),
            sample_rate,
            flush_after_samples,
        }
    }

    /// Append a chunk's samples (tracks the chunk's own sample rate so the WAV
    /// header is correct even if the input rate differs from the constructed one).
    pub fn push(&mut self, audio: &AudioFrame) {
        if self.pcm.is_empty() {
            self.sample_rate = audio.sample_rate;
        }
        self.pcm.extend_from_slice(&audio.pcm);
    }

    /// Whether a full segment has accumulated.
    pub fn is_ready(&self) -> bool {
        self.pcm.len() >= self.flush_after_samples
    }

    /// Whether there is any buffered audio at all.
    pub fn is_empty(&self) -> bool {
        self.pcm.is_empty()
    }

    /// Take the buffered segment as a WAV file (drains the buffer). `None` if empty
    /// **or (near-)silent** — segmented Whisper-family ASR HALLUCINATES text from
    /// silence (e.g. "thank you" / "bye"), which would spuriously drive the
    /// conversation (even hang up the call). Only a segment with real speech energy
    /// is POSTed.
    pub fn take_wav(&mut self) -> Option<Vec<u8>> {
        if self.pcm.is_empty() || is_silent(&self.pcm) {
            self.pcm.clear();
            return None;
        }
        let rate = self.sample_rate;
        let pcm = std::mem::take(&mut self.pcm);
        Some(wav_pcm16_mono(&pcm, rate))
    }
}

/// RMS-energy threshold below which a PCM16 segment is treated as silence/quiet
/// background and dropped (not transcribed). Speech RMS is in the thousands; a quiet
/// room is well under 250. Tunable.
const SILENCE_RMS: f64 = 250.0;

/// Whether a PCM16 segment is (near-)silent by RMS energy.
fn is_silent(pcm: &[i16]) -> bool {
    if pcm.is_empty() {
        return true;
    }
    let sum_sq: f64 = pcm.iter().map(|&s| (s as f64) * (s as f64)).sum();
    (sum_sq / pcm.len() as f64).sqrt() < SILENCE_RMS
}

/// Wrap raw little-endian mono PCM16 in a canonical 44-byte RIFF/WAVE header.
/// Dependency-free (no `hound`): the providers' upload endpoints want an audio
/// file, and a minimal valid WAV is the most portable container.
pub fn wav_pcm16_mono(pcm: &[i16], sample_rate: u32) -> Vec<u8> {
    let num_channels: u16 = 1;
    let bits_per_sample: u16 = 16;
    let byte_rate = sample_rate * num_channels as u32 * (bits_per_sample / 8) as u32;
    let block_align = num_channels * (bits_per_sample / 8);
    let data_len = (pcm.len() * 2) as u32;
    let riff_len = 36 + data_len;

    let mut out = Vec::with_capacity(44 + pcm.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_len.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    // fmt subchunk
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // audio format = PCM
    out.extend_from_slice(&num_channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());
    // data subchunk
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for s in pcm {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// One `multipart/form-data` part: either a simple text field or a file (with a
/// filename + content-type and a binary body).
pub enum Part {
    /// A plain `name=value` form field.
    Text { name: String, value: String },
    /// A file upload field.
    File {
        name: String,
        filename: String,
        content_type: String,
        body: Vec<u8>,
    },
}

impl Part {
    /// A text form field.
    pub fn text(name: impl Into<String>, value: impl Into<String>) -> Self {
        Part::Text {
            name: name.into(),
            value: value.into(),
        }
    }

    /// A file upload field.
    pub fn file(
        name: impl Into<String>,
        filename: impl Into<String>,
        content_type: impl Into<String>,
        body: Vec<u8>,
    ) -> Self {
        Part::File {
            name: name.into(),
            filename: filename.into(),
            content_type: content_type.into(),
            body,
        }
    }
}

/// Build a `multipart/form-data` request body and return
/// `(content_type_header, body_bytes)`. Hand-rolled because the crate's `reqwest`
/// is intentionally built without the `multipart` feature.
pub fn multipart_body(boundary: &str, parts: &[Part]) -> (String, Vec<u8>) {
    let mut body = Vec::new();
    for part in parts {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        match part {
            Part::Text { name, value } => {
                body.extend_from_slice(
                    format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
                );
                body.extend_from_slice(value.as_bytes());
                body.extend_from_slice(b"\r\n");
            }
            Part::File {
                name,
                filename,
                content_type,
                body: file_body,
            } => {
                body.extend_from_slice(
                    format!(
                        "Content-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
                    )
                    .as_bytes(),
                );
                body.extend_from_slice(format!("Content-Type: {content_type}\r\n\r\n").as_bytes());
                body.extend_from_slice(file_body);
                body.extend_from_slice(b"\r\n");
            }
        }
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

#[cfg(test)]
mod rest_common_tests {
    use super::*;

    #[test]
    fn wav_header_is_valid_riff_pcm16() {
        let wav = wav_pcm16_mono(&[1, -2, 256], 16_000);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(&wav[36..40], b"data");
        // 44-byte header + 3 samples * 2 bytes.
        assert_eq!(wav.len(), 44 + 6);
        // sample rate at offset 24 (LE u32).
        assert_eq!(
            u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]),
            16_000
        );
        // first sample (LE i16) round-trips.
        assert_eq!(i16::from_le_bytes([wav[44], wav[45]]), 1);
    }

    #[test]
    fn multipart_includes_fields_file_and_terminator() {
        let parts = vec![
            Part::text("model_id", "scribe_v2"),
            Part::file("file", "audio.wav", "audio/x-wav", vec![0xDE, 0xAD]),
        ];
        let (ct, body) = multipart_body("BOUNDARY", &parts);
        assert!(ct.starts_with("multipart/form-data; boundary=BOUNDARY"));
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("name=\"model_id\""));
        assert!(text.contains("scribe_v2"));
        assert!(text.contains("filename=\"audio.wav\""));
        assert!(text.contains("Content-Type: audio/x-wav"));
        assert!(text.contains("--BOUNDARY--"));
    }

    #[test]
    fn segment_buffer_flushes_after_threshold() {
        let mut buf = SegmentBuffer::new(16_000, 0.001); // 16 samples threshold
        assert!(buf.is_empty());
        // Non-silent samples (above the RMS gate) so the segment is actually flushed.
        buf.push(&AudioFrame::mono(vec![5000i16; 8], 16_000));
        assert!(!buf.is_ready());
        buf.push(&AudioFrame::mono(vec![5000i16; 8], 16_000));
        assert!(buf.is_ready());
        let wav = buf.take_wav().expect("wav");
        assert_eq!(&wav[0..4], b"RIFF");
        // draining resets the buffer.
        assert!(buf.is_empty());
        assert!(buf.take_wav().is_none());
    }

    #[test]
    fn segment_buffer_drops_silence() {
        // (Near-)silent segments are NOT transcribed — segmented Whisper-family ASR
        // hallucinates text from silence, which would spuriously drive the call.
        let mut buf = SegmentBuffer::new(16_000, 0.001);
        buf.push(&AudioFrame::mono(vec![0i16; 32], 16_000));
        assert!(buf.is_ready());
        assert!(
            buf.take_wav().is_none(),
            "silence must be dropped, not POSTed"
        );
        assert!(buf.is_empty(), "dropping silence drains the buffer");
    }
}
