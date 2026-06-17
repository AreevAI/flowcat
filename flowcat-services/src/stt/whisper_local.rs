// SPDX-License-Identifier: Apache-2.0
//
//! **Local Whisper** STT (whisper.cpp via `whisper-rs`).
//!
//! A **(D)istinct** local model (PROVIDERS.md §2/§5): runs whisper.cpp
//! in-process via the `whisper-rs` bindings — **no network, no API key**. Audio is
//! buffered as the call streams; whisper transcribes a finalized segment when the
//! buffer crosses a duration threshold. Behind the `stt-whisper-local` feature.
//!
//! **Toolchain note:** `whisper-rs` bundles whisper.cpp, so building this feature
//! needs **`cmake` + a C/C++ toolchain**. The Rust here compiles only when the C
//! build succeeds (see PROVIDERS.md §5). The pure resample/segment seam
//! ([`pcm_to_whisper_f32`], [`SegmentBuffer`]) is unit-tested without the model.
//!
//! ## Design
//!
//! whisper.cpp is a **batch** transcriber (no streaming partials): it takes a whole
//! 16 kHz f32 mono buffer and returns text segments. The live STT contract here
//! buffers incoming chunks ([`SegmentBuffer`]) and, once enough audio has
//! accumulated ([`WhisperLocalStt::segment_secs`]), drains the buffer and runs
//! inference on a **blocking task** (whisper is CPU-bound — never block the async
//! runtime). Each run emits one final [`Frame::Transcription`]. The model is loaded
//! once in [`SttService::start`] and shared (`Arc`) into the blocking task.

use std::sync::Arc;

use async_trait::async_trait;

use flowcat_core::error::{FlowcatError, Result};
use flowcat_core::processor::frame::{AudioFrame, Frame, StartParams};
use flowcat_core::service::SttService;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// whisper.cpp expects **16 kHz** mono f32 audio. Buffers at other rates are
/// linearly resampled to this rate by [`pcm_to_whisper_f32`].
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// Default amount of buffered audio (seconds) before a segment is transcribed.
const DEFAULT_SEGMENT_SECS: f32 = 4.0;

/// Convert an i16-PCM [`AudioFrame`] to the 16 kHz mono `f32` whisper.cpp wants.
///
/// - i16 → f32 in `[-1.0, 1.0]` (divide by 32768).
/// - If the frame is not already 16 kHz, linearly resample to 16 kHz (good enough
///   for ASR; whisper is robust to it).
///
/// **Pure** — the seam the fixture tests drive without a model file.
pub fn pcm_to_whisper_f32(audio: &AudioFrame) -> Vec<f32> {
    // The live path is always mono; we treat `pcm` as a mono stream regardless of
    // `num_channels` (the AudioFrame contract).
    let src: Vec<f32> = audio.pcm.iter().map(|s| *s as f32 / 32_768.0).collect();
    if audio.sample_rate == WHISPER_SAMPLE_RATE || src.is_empty() {
        return src;
    }
    resample_linear(&src, audio.sample_rate, WHISPER_SAMPLE_RATE)
}

/// Linear-interpolation resampler `src_rate → dst_rate`. Pure + allocation-bounded.
fn resample_linear(src: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || src.len() < 2 {
        return src.to_vec();
    }
    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = ((src.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        // Position in source space.
        let pos = i as f64 / ratio;
        let idx = pos.floor() as usize;
        let frac = (pos - idx as f64) as f32;
        let a = src.get(idx).copied().unwrap_or(0.0);
        let b = src.get(idx + 1).copied().unwrap_or(a);
        out.push(a + (b - a) * frac);
    }
    out
}

/// Accumulates resampled 16 kHz f32 audio until a segment's worth is ready.
/// Pure (no model) so the buffering boundary is unit-tested.
#[derive(Default)]
pub struct SegmentBuffer {
    samples: Vec<f32>,
}

impl SegmentBuffer {
    /// A fresh, empty buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append already-resampled 16 kHz f32 samples.
    pub fn push(&mut self, samples: &[f32]) {
        self.samples.extend_from_slice(samples);
    }

    /// Whether at least `threshold_samples` of audio is buffered.
    pub fn is_ready(&self, threshold_samples: usize) -> bool {
        self.samples.len() >= threshold_samples
    }

    /// Buffered sample count.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Take the buffered audio, leaving the buffer empty.
    pub fn drain(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.samples)
    }
}

/// Local whisper.cpp STT service. The model is loaded once on [`SttService::start`].
pub struct WhisperLocalStt {
    model_path: String,
    /// Whisper inference language (e.g. `Some("en")`); `None` ⇒ auto-detect.
    language: Option<String>,
    /// How many seconds of audio to buffer before transcribing a segment.
    segment_secs: f32,
    /// Loaded model, shared into the blocking inference task.
    ctx: Option<Arc<WhisperContext>>,
    buffer: SegmentBuffer,
    muted: bool,
}

impl WhisperLocalStt {
    /// Construct bound to a GGML/GGUF model file path (auto-detect language,
    /// 4-second segments).
    pub fn new(model_path: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
            language: None,
            segment_secs: DEFAULT_SEGMENT_SECS,
            ctx: None,
            buffer: SegmentBuffer::new(),
            muted: false,
        }
    }

    /// Pin the transcription language (default: auto-detect).
    pub fn language(mut self, lang: impl Into<String>) -> Self {
        self.language = Some(lang.into());
        self
    }

    /// Override the per-segment buffer duration (default 4 s).
    pub fn segment_secs(mut self, secs: f32) -> Self {
        self.segment_secs = secs.max(0.1);
        self
    }

    /// Samples that make up one segment at 16 kHz.
    fn threshold_samples(&self) -> usize {
        (self.segment_secs * WHISPER_SAMPLE_RATE as f32) as usize
    }

    /// Run whisper.cpp over one finalized segment, on a blocking task, and decode
    /// the segment text into a single final [`Frame::Transcription`]. Returns an
    /// empty vec when whisper produced no text.
    async fn transcribe_segment(
        ctx: Arc<WhisperContext>,
        language: Option<String>,
        samples: Vec<f32>,
    ) -> Result<Vec<Frame>> {
        if samples.is_empty() {
            return Ok(vec![]);
        }
        // whisper is CPU-bound. The `stt-whisper-local` feature is deliberately
        // tokio-free (`dep:tokio` is not pulled for this feature — §5), so we
        // cannot use `spawn_blocking`. We offload to a **scoped OS thread** to keep
        // the work off the async task's stack and join it here; this `await` does
        // block the calling task for the inference, which is acceptable for a local
        // batch model (the call is doing nothing else meanwhile).
        let text = std::thread::scope(|s| {
            s.spawn(|| -> Result<String> { run_inference(&ctx, language.as_deref(), &samples) })
                .join()
                .map_err(|_| {
                    FlowcatError::Other("whisper_local: inference thread panicked".into())
                })?
        })?;

        Ok(decode_segment_text(&text))
    }
}

/// Run whisper.cpp over `samples`, returning the concatenated trimmed segment text.
/// Blocking (CPU-bound) — called from a worker thread by [`transcribe_segment`].
fn run_inference(ctx: &WhisperContext, language: Option<&str>, samples: &[f32]) -> Result<String> {
    let mut state = ctx
        .create_state()
        .map_err(|e| FlowcatError::Other(format!("whisper_local create_state: {e}")))?;
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_n_threads(num_threads() as i32);
    params.set_translate(false);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    if let Some(lang) = language {
        params.set_language(Some(lang));
    }
    state
        .full(params, samples)
        .map_err(|e| FlowcatError::Other(format!("whisper_local inference: {e}")))?;
    let n = state.full_n_segments();
    let mut out = String::new();
    for i in 0..n {
        if let Some(seg) = state.get_segment(i) {
            if let Ok(s) = seg.to_str_lossy() {
                out.push_str(s.as_ref());
            }
        }
    }
    Ok(out.trim().to_string())
}

/// Turn whisper's finalized segment text into transcription frames. **Pure** — the
/// decode seam the fixture tests drive. whisper.cpp annotates non-speech audio with
/// bracketed/parenthesised markers (`[BLANK_AUDIO]`, `[Music]`, `(silence)`, etc.);
/// a segment that is only such markers (or empty/punctuation) yields **nothing** so
/// silence never reaches the LLM as a "user turn".
pub fn decode_segment_text(text: &str) -> Vec<Frame> {
    let spoken = strip_non_speech(text);
    // Require at least one real word character — drops empty / whitespace /
    // punctuation-only / pure-annotation segments.
    if !spoken.chars().any(|c| c.is_alphanumeric()) {
        return vec![];
    }
    vec![Frame::Transcription {
        text: spoken,
        user_id: Arc::from("user"),
        language: None,
        final_: true,
    }]
}

/// Strip whisper.cpp's non-speech annotations — `[...]` and `(...)` spans like
/// `[BLANK_AUDIO]` / `[Music]` / `(silence)` — and collapse surrounding whitespace,
/// leaving only the spoken words. **Pure.**
fn strip_non_speech(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut square = 0i32;
    let mut round = 0i32;
    for c in text.chars() {
        match c {
            '[' => square += 1,
            ']' => square = (square - 1).max(0),
            '(' => round += 1,
            ')' => round = (round - 1).max(0),
            _ if square > 0 || round > 0 => {}
            _ => out.push(c),
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Thread count for whisper inference, with a safe fallback.
fn num_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .max(1)
}

#[async_trait]
impl SttService for WhisperLocalStt {
    fn name(&self) -> &str {
        "whisper_local"
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        // Load the GGML/GGUF model once (CPU; GPU off by default). Offloaded to a
        // scoped OS thread (the feature is tokio-free — see `transcribe_segment`).
        let path = self.model_path.clone();
        let ctx = std::thread::scope(|s| {
            s.spawn(|| WhisperContext::new_with_params(&path, WhisperContextParameters::default()))
                .join()
                .map_err(|_| FlowcatError::Other("whisper_local: load thread panicked".into()))?
                .map_err(|e| FlowcatError::Other(format!("whisper_local load model: {e}")))
        })?;
        self.ctx = Some(Arc::new(ctx));
        Ok(())
    }

    async fn run_stt(&mut self, audio: Arc<AudioFrame>) -> Result<Vec<Frame>> {
        if self.muted {
            return Ok(vec![]);
        }
        let ctx = self
            .ctx
            .as_ref()
            .ok_or_else(|| FlowcatError::Other("whisper_local: run_stt before start".into()))?
            .clone();
        // Resample + buffer this chunk.
        let resampled = pcm_to_whisper_f32(&audio);
        self.buffer.push(&resampled);
        if !self.buffer.is_ready(self.threshold_samples()) {
            return Ok(vec![]);
        }
        // A segment's worth has accumulated — transcribe it.
        let samples = self.buffer.drain();
        Self::transcribe_segment(ctx, self.language.clone(), samples).await
    }

    async fn set_muted(&mut self, muted: bool) {
        self.muted = muted;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcm_to_f32_scales_and_keeps_rate() {
        let af = AudioFrame::mono(vec![0, 16_384, -16_384, 32_767], 16_000);
        let f = pcm_to_whisper_f32(&af);
        assert_eq!(f.len(), 4);
        assert!((f[0] - 0.0).abs() < 1e-6);
        assert!((f[1] - 0.5).abs() < 1e-3);
        assert!((f[2] + 0.5).abs() < 1e-3);
        assert!(f[3] > 0.99 && f[3] <= 1.0);
    }

    #[test]
    fn pcm_to_f32_resamples_to_16k() {
        // 8 kHz in → 16 kHz out roughly doubles the sample count.
        let af = AudioFrame::mono(vec![0i16; 80], 8_000);
        let f = pcm_to_whisper_f32(&af);
        assert!(
            (159..=161).contains(&f.len()),
            "expected ~160 samples, got {}",
            f.len()
        );
    }

    #[test]
    fn resample_is_identity_at_same_rate() {
        let src = vec![0.1, 0.2, 0.3, 0.4];
        assert_eq!(resample_linear(&src, 16_000, 16_000), src);
    }

    #[test]
    fn segment_buffer_accumulates_and_drains() {
        let mut buf = SegmentBuffer::new();
        assert!(buf.is_empty());
        buf.push(&[0.0; 100]);
        assert!(!buf.is_ready(200));
        buf.push(&[0.0; 150]);
        assert!(buf.is_ready(200));
        assert_eq!(buf.len(), 250);
        let drained = buf.drain();
        assert_eq!(drained.len(), 250);
        assert!(buf.is_empty());
    }

    #[test]
    fn decode_empty_segment_yields_nothing() {
        assert!(decode_segment_text("").is_empty());
        assert!(decode_segment_text("   \n ").is_empty());
    }

    #[test]
    fn decode_drops_non_speech_annotations() {
        // whisper.cpp tags silence/noise; these must never become a user turn.
        assert!(decode_segment_text("[BLANK_AUDIO]").is_empty());
        assert!(decode_segment_text("[Music]").is_empty());
        assert!(decode_segment_text("(silence)").is_empty());
        assert!(decode_segment_text(" [Music] . ").is_empty());
        // Real speech with a stray annotation keeps just the words.
        let frames = decode_segment_text("[Music] book a dentist");
        assert!(matches!(&frames[..],
            [Frame::Transcription { text, .. }] if text == "book a dentist"));
    }

    #[test]
    fn decode_segment_text_makes_a_final_transcription() {
        let frames = decode_segment_text("  book a dentist  ");
        assert_eq!(frames.len(), 1);
        match &frames[0] {
            Frame::Transcription { text, final_, .. } => {
                assert_eq!(text, "book a dentist");
                assert!(final_);
            }
            other => panic!("expected final Transcription, got {}", other.name()),
        }
    }

    /// Live smoke (requires a GGML model file at `WHISPER_MODEL_PATH`): load the
    /// model, run inference on a beat of silence, confirm no panic. Run with:
    /// `WHISPER_MODEL_PATH=/path/ggml-base.en.bin cargo test -p flowcat-services \
    ///   --features stt-whisper-local -- --ignored whisper_local_live`
    #[tokio::test]
    #[ignore = "requires WHISPER_MODEL_PATH (a GGML model file) + cmake-built whisper.cpp"]
    async fn whisper_local_live_loads_and_transcribes() {
        let path = std::env::var("WHISPER_MODEL_PATH").expect("WHISPER_MODEL_PATH");
        let mut stt = WhisperLocalStt::new(path).language("en").segment_secs(0.2);
        stt.start(&StartParams::default())
            .await
            .expect("load model");
        // 1 s of silence at 16 kHz → crosses the 0.2 s threshold.
        let silence = Arc::new(AudioFrame::mono(vec![0i16; 16_000], 16_000));
        let _ = stt.run_stt(silence).await.expect("run_stt");
    }
}
