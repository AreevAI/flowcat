// SPDX-License-Identifier: Apache-2.0
//
//! Wake-word gating (pure logic, no network).
//!
//! Ports pipecat's wake filters (`processors/filters/wake_check_filter.py`,
//! `wake_notifier_filter.py`) to the flowcat frame stream:
//!
//! - [`WakeCheckFilter`] — buffers per-user transcription text and only lets a
//!   [`Frame::Transcription`] through once a wake phrase has been seen; a
//!   keepalive window keeps the gate open for follow-up turns.
//! - [`WakeNotifierProcessor`] — fires a [`WakeNotifier`] callback when a matching
//!   frame passes a predicate, without otherwise altering the stream (used to wake
//!   another component, e.g. arm the brain).
//!
//! Detection is plain substring/word matching over transcripts — no audio model,
//! no network. (Wake detection straight off the audio is a VAD/turn concern.)

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use crate::error::Result;
use crate::processor::frame::{Frame, FrameKind};
use crate::processor::{Envelope, FrameProcessor, Link};

/// A monotonic clock seam so the keepalive window is testable without real time.
pub trait WakeClock: Send {
    /// Current time as a [`Duration`] since some fixed, monotonic epoch.
    fn now(&self) -> Duration;
}

/// The default wall-clock implementation (`Instant`-backed).
pub struct SystemWakeClock {
    base: std::time::Instant,
}

impl Default for SystemWakeClock {
    fn default() -> Self {
        Self {
            base: std::time::Instant::now(),
        }
    }
}

impl WakeClock for SystemWakeClock {
    fn now(&self) -> Duration {
        self.base.elapsed()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WakeState {
    Idle,
    Awake,
}

struct ParticipantState {
    state: WakeState,
    wake_timer: Duration,
    accumulator: String,
}

impl ParticipantState {
    fn new() -> Self {
        Self {
            state: WakeState::Idle,
            wake_timer: Duration::ZERO,
            accumulator: String::new(),
        }
    }
}

/// Gates [`Frame::Transcription`] frames behind a spoken wake phrase. Until a wake
/// phrase appears in a user's accumulated transcript, that user's transcriptions
/// are withheld; once seen, the gate opens and a keepalive window keeps it open for
/// follow-up turns (mirrors pipecat `WakeCheckFilter`). All other frames pass
/// through unchanged.
///
/// Matching is case-insensitive whitespace-tolerant: each wake phrase's words must
/// appear in order separated only by whitespace (pipecat builds the same
/// `\b…\s*…\b` regex). On a match, the accumulator is trimmed to start at the wake
/// phrase so the gated transcript carries the command, not the wake word's prefix.
pub struct WakeCheckFilter {
    name: &'static str,
    wake_phrases: Vec<Vec<String>>,
    keepalive: Duration,
    clock: Box<dyn WakeClock>,
    participants: HashMap<String, ParticipantState>,
}

impl WakeCheckFilter {
    /// A filter that wakes on any of `wake_phrases`, with a `keepalive` window
    /// (seconds) after each accepted transcript during which the gate stays open.
    pub fn new(wake_phrases: Vec<String>, keepalive: Duration) -> Self {
        Self::with_clock(
            wake_phrases,
            keepalive,
            Box::new(SystemWakeClock::default()),
        )
    }

    /// As [`WakeCheckFilter::new`] but with an injected clock (for tests).
    pub fn with_clock(
        wake_phrases: Vec<String>,
        keepalive: Duration,
        clock: Box<dyn WakeClock>,
    ) -> Self {
        let phrases = wake_phrases
            .into_iter()
            .map(|p| p.split_whitespace().map(|w| w.to_lowercase()).collect())
            .collect();
        Self {
            name: "wake-check",
            wake_phrases: phrases,
            keepalive,
            clock,
            participants: HashMap::new(),
        }
    }
}

/// Find the byte offset in `haystack` where a wake phrase begins, matched on
/// **whole-word** boundaries (so "computer" does not match "computers"), tolerant
/// of any whitespace between the phrase's words. Mirrors pipecat's
/// `\bword\s*word\b` regex. A free function so it borrows only the phrase list,
/// not all of `self` (the per-participant accumulator is borrowed mutably at the
/// call site).
fn match_wake(wake_phrases: &[Vec<String>], haystack: &str) -> Option<usize> {
    let lower = haystack.to_lowercase();
    let words: Vec<(usize, &str)> = word_offsets(&lower);
    for phrase in wake_phrases {
        if phrase.is_empty() {
            continue;
        }
        // Slide over the token list looking for the phrase's words in order, each
        // matched as a whole token (whole-word boundary).
        'start: for start in 0..words.len() {
            if words.len() - start < phrase.len() {
                break;
            }
            for (i, pw) in phrase.iter().enumerate() {
                if words[start + i].1 != pw.as_str() {
                    continue 'start;
                }
            }
            return Some(words[start].0);
        }
    }
    None
}

/// Tokenize into `(byte_offset, word)` pairs split on whitespace.
fn word_offsets(s: &str) -> Vec<(usize, &str)> {
    let mut out = Vec::new();
    let mut idx = 0;
    for word in s.split_whitespace() {
        // Find the word starting at/after idx (split_whitespace collapses runs).
        if let Some(rel) = s[idx..].find(word) {
            let at = idx + rel;
            out.push((at, word));
            idx = at + word.len();
        }
    }
    out
}

#[async_trait]
impl FrameProcessor for WakeCheckFilter {
    fn name(&self) -> &str {
        self.name
    }

    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        let Frame::Transcription {
            ref text,
            ref user_id,
            ..
        } = env.frame
        else {
            // Non-transcription frames always pass through.
            link.push(env.meta, env.frame, env.direction).await;
            return Ok(());
        };

        let now = self.clock.now();
        let keepalive = self.keepalive;
        let user_key = user_id.to_string();
        // Reborrow split: take the matcher input (phrases) by shared ref before
        // mutably borrowing the participant entry.
        let wake_phrases = &self.wake_phrases;
        let p = self
            .participants
            .entry(user_key)
            .or_insert_with(ParticipantState::new);

        // Already awake within the keepalive window → pass straight through.
        if p.state == WakeState::Awake {
            if now.saturating_sub(p.wake_timer) < keepalive {
                p.wake_timer = now;
                link.push(env.meta, env.frame, env.direction).await;
                return Ok(());
            }
            p.state = WakeState::Idle;
        }

        // Separate fragments with a space so two transcripts never fuse into a
        // single token (e.g. "…it" + "hey…" must not become "ithey").
        if !p.accumulator.is_empty() && !p.accumulator.ends_with(char::is_whitespace) {
            p.accumulator.push(' ');
        }
        p.accumulator.push_str(text);
        if let Some(start) = match_wake(wake_phrases, &p.accumulator) {
            p.state = WakeState::Awake;
            p.wake_timer = now;
            // Trim to the wake phrase so the command (not the wake prefix) flows on.
            let gated = p.accumulator[start..].to_string();
            p.accumulator.clear();
            let (uid, lang, final_) = match env.frame {
                Frame::Transcription {
                    user_id,
                    language,
                    final_,
                    ..
                } => (user_id, language, final_),
                _ => unreachable!(),
            };
            link.push(
                env.meta,
                Frame::Transcription {
                    text: gated,
                    user_id: uid,
                    language: lang,
                    final_,
                },
                env.direction,
            )
            .await;
        }
        // Not yet woken → withhold this transcription.
        Ok(())
    }
}

/// A callback fired when a wake condition is met. Cheaply cloneable.
pub type WakeNotifier = Arc<dyn Fn() + Send + Sync>;

/// A predicate over a frame deciding whether to fire the notifier.
pub type WakePredicate = Arc<dyn Fn(&Frame) -> bool + Send + Sync>;

/// Fires a [`WakeNotifier`] when a frame of one of the watched [`FrameKind`]s
/// satisfies a predicate, then forwards the frame unchanged. Port of pipecat
/// `WakeNotifierFilter` — used to wake another component (e.g. arm the brain on
/// the first user-stopped-speaking) without consuming the frame.
pub struct WakeNotifierProcessor {
    name: &'static str,
    kinds: Vec<FrameKind>,
    predicate: WakePredicate,
    notifier: WakeNotifier,
}

impl WakeNotifierProcessor {
    /// Watch `kinds`; when one matches `predicate`, call `notifier`.
    pub fn new(kinds: Vec<FrameKind>, predicate: WakePredicate, notifier: WakeNotifier) -> Self {
        Self {
            name: "wake-notifier",
            kinds,
            predicate,
            notifier,
        }
    }
}

#[async_trait]
impl FrameProcessor for WakeNotifierProcessor {
    fn name(&self) -> &str {
        self.name
    }

    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        if self.kinds.contains(&env.frame.kind()) && (self.predicate)(&env.frame) {
            (self.notifier)();
        }
        link.push(env.meta, env.frame, env.direction).await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::test_harness::drive;
    use crate::processor::frame::Direction;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A test clock the test advances manually.
    #[derive(Clone, Default)]
    struct FakeClock(Arc<AtomicU64>);
    impl FakeClock {
        fn advance(&self, ms: u64) {
            self.0.fetch_add(ms, Ordering::SeqCst);
        }
    }
    impl WakeClock for FakeClock {
        fn now(&self) -> Duration {
            Duration::from_millis(self.0.load(Ordering::SeqCst))
        }
    }

    fn transcription(text: &str, user: &str) -> Frame {
        Frame::Transcription {
            text: text.into(),
            user_id: Arc::from(user),
            language: None,
            final_: true,
        }
    }

    #[tokio::test]
    async fn gates_until_wake_phrase_then_opens() {
        let filter = WakeCheckFilter::new(vec!["hey assistant".into()], Duration::from_secs(3));
        let out = drive(
            Box::new(filter),
            vec![
                transcription("what time is it", "u1"), // withheld (no wake)
                transcription("hey assistant set a timer", "u1"), // wakes + passes
                transcription("for ten minutes", "u1"), // keepalive → passes
            ],
            Direction::Downstream,
        )
        .await;
        let texts: Vec<String> = out
            .into_iter()
            .filter_map(|f| match f {
                Frame::Transcription { text, .. } => Some(text),
                _ => None,
            })
            .collect();
        // First is withheld; the woken transcript is trimmed to the wake phrase;
        // the follow-up rides the keepalive.
        assert_eq!(texts, vec!["hey assistant set a timer", "for ten minutes"]);
    }

    #[tokio::test]
    async fn wake_only_fires_on_the_phrase() {
        let filter = WakeCheckFilter::new(vec!["computer".into()], Duration::from_secs(3));
        let out = drive(
            Box::new(filter),
            vec![
                transcription("hello world", "u1"),
                transcription("the computers are noisy", "u1"),
            ],
            Direction::Downstream,
        )
        .await;
        // "computers" is not the standalone word "computer" → nothing passes.
        let count = out
            .iter()
            .filter(|f| matches!(f, Frame::Transcription { .. }))
            .count();
        assert_eq!(count, 0, "fired on a non-wake word: {out:?}");
    }

    #[tokio::test]
    async fn keepalive_expires_and_regates() {
        let clock = FakeClock::default();
        let filter = WakeCheckFilter::with_clock(
            vec!["wake".into()],
            Duration::from_millis(1000),
            Box::new(clock.clone()),
        );
        // We need to drive frames with time advancing between them, so run the
        // processor directly rather than via the batch harness.
        use crate::processor::runtime::{channel, NORMAL_CHAN_CAP};
        use crate::processor::{Envelope, Link};
        use std::sync::atomic::AtomicI64;

        let (cap_tx, mut cap_rx) = channel(Arc::from("cap"), NORMAL_CHAN_CAP);
        let link = Link {
            next: Some(cap_tx.clone()),
            prev: Some(cap_tx),
            name: Arc::from("wake-check"),
            clock: crate::processor::Clock::new(),
            observer: None,
            enable_metrics: false,
            enable_usage_metrics: false,
            ttfb_start: Arc::new(AtomicI64::new(0)),
            processing_start: Arc::new(AtomicI64::new(0)),
        };
        let mut filter: Box<dyn FrameProcessor> = Box::new(filter);

        let drain = |rx: &mut crate::processor::runtime::ProcessorRx| -> Vec<Frame> {
            let mut v = Vec::new();
            while let Ok(e) = rx.normal.try_recv() {
                v.push(e.frame);
            }
            v
        };

        // Wake.
        filter
            .process_frame(
                Envelope::new(transcription("wake up", "u"), Direction::Downstream),
                &link,
            )
            .await
            .unwrap();
        assert_eq!(drain(&mut cap_rx).len(), 1);

        // Within keepalive → passes.
        clock.advance(500);
        filter
            .process_frame(
                Envelope::new(transcription("still here", "u"), Direction::Downstream),
                &link,
            )
            .await
            .unwrap();
        assert_eq!(drain(&mut cap_rx).len(), 1);

        // After keepalive expires (relative to the last accepted frame) → re-gated.
        clock.advance(2000);
        filter
            .process_frame(
                Envelope::new(transcription("ignored now", "u"), Direction::Downstream),
                &link,
            )
            .await
            .unwrap();
        assert_eq!(
            drain(&mut cap_rx).len(),
            0,
            "should re-gate after keepalive"
        );
    }

    #[tokio::test]
    async fn notifier_fires_on_matching_kind_and_predicate() {
        let hits = Arc::new(AtomicU64::new(0));
        let h2 = hits.clone();
        let proc = WakeNotifierProcessor::new(
            vec![FrameKind::Transcription],
            Arc::new(|f| matches!(f, Frame::Transcription { text, .. } if text.contains("now"))),
            Arc::new(move || {
                h2.fetch_add(1, Ordering::SeqCst);
            }),
        );
        let out = drive(
            Box::new(proc),
            vec![
                transcription("later", "u"),
                transcription("do it now", "u"),
                Frame::Text("now".into()), // wrong kind → must not fire
            ],
            Direction::Downstream,
        )
        .await;
        // Fires exactly once (the matching transcription); all frames forwarded.
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        assert_eq!(out.len(), 3, "notifier must forward every frame");
    }
}
