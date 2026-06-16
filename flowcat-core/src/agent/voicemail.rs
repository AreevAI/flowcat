// SPDX-License-Identifier: Apache-2.0
//
//! Voicemail / answering-machine detection (pure logic, no network).
//!
//! pipecat's `VoicemailDetector` is a *parallel LLM classifier* (it asks a text
//! LLM "CONVERSATION or VOICEMAIL?"). That needs a network LLM, so it is out of
//! this pure-logic scope. The flowcat detector is the **heuristic** half:
//! a [`FrameProcessor`] that classifies the far end from the existing frame
//! stream alone, with two signals (either fires a decision):
//!
//! 1. **Phrase heuristic** — accumulate the far-end transcript
//!    ([`Frame::Transcription`]/[`Frame::InterimTranscription`]) and match it
//!    against a curated set of voicemail-greeting phrases (ported from pipecat's
//!    `VoicemailDetector.DEFAULT_SYSTEM_PROMPT` VOICEMAIL list — "please leave a
//!    message", "you've reached", "not available", …).
//! 2. **Long-speech VAD heuristic** — if the *first* turn is one long
//!    uninterrupted stretch of speech (a [`Frame::VadUserStartedSpeaking`] with no
//!    stop for longer than a threshold, or a single turn whose transcript exceeds
//!    a word count), it is far more likely a recorded greeting than a human's
//!    "Hello?" — mirrors pipecat's `long_speech_timeout` early-trigger.
//!
//! A decision is emitted **once** as a [`Frame::Custom`] carrying a
//! [`VoicemailDecision`] (no `Frame` variant is promoted at this layer); the
//! downstream brain downcasts it. The detector then
//! goes inert (it never re-decides) and forwards every frame unchanged.

use std::any::Any;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use crate::error::Result;
use crate::processor::frame::{CustomFrame, Frame, FrameClass};
use crate::processor::{Envelope, FrameProcessor, Link};

/// The classification the detector reaches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Classification {
    /// A live human answered.
    Conversation,
    /// The call reached a voicemail / answering machine / carrier recording.
    Voicemail,
}

/// Why the detector decided as it did (for tracing / brain logic).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionReason {
    /// A voicemail greeting phrase was matched in the transcript.
    GreetingPhrase,
    /// The first turn was an unusually long uninterrupted stretch of speech.
    LongFirstTurn,
    /// No voicemail signal was seen by the time the user stopped speaking.
    HumanResponse,
}

/// The decision frame the detector emits (carried in [`Frame::Custom`]). The
/// brain downcasts this via [`CustomFrame::as_any`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoicemailDecision {
    /// The classification reached.
    pub classification: Classification,
    /// Why (which heuristic fired).
    pub reason: DecisionReason,
}

impl CustomFrame for VoicemailDecision {
    fn frame_class(&self) -> FrameClass {
        // A control-plane signal for the brain — ordered, survives nothing
        // special; Control is the closest match.
        FrameClass::Control
    }

    fn name(&self) -> &'static str {
        "VoicemailDecision"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Tunables for [`VoicemailDetector`].
#[derive(Debug, Clone)]
pub struct VoicemailParams {
    /// First-turn speech longer than this is treated as a recorded greeting.
    pub long_first_turn: Duration,
    /// A first-turn transcript with at least this many words is treated as a
    /// greeting even before the turn ends (a belt-and-braces word-count signal).
    pub long_first_turn_words: usize,
}

impl Default for VoicemailParams {
    fn default() -> Self {
        Self {
            // A human's first utterance ("Hello?", "Who's this?") is short; a
            // recorded greeting runs for seconds.
            long_first_turn: Duration::from_secs(5),
            long_first_turn_words: 12,
        }
    }
}

/// The curated voicemail-greeting phrases (lowercased), ported from pipecat's
/// `VoicemailDetector.DEFAULT_SYSTEM_PROMPT` VOICEMAIL bullet list. A match in the
/// accumulated transcript classifies the call as voicemail.
const VOICEMAIL_PHRASES: &[&str] = &[
    "please leave a message",
    "leave a message",
    "leave your name and number",
    "leave your message after",
    "after the tone",
    "after the beep",
    "you've reached",
    "you have reached",
    "the person you are trying to reach",
    "is not available",
    "not available right now",
    "i'm not available",
    "is unavailable",
    "the number you have dialed",
    "the number you dialed",
    "is not in service",
    "the mailbox",
    "mailbox is full",
    "has not been set up",
    "voicemail",
    "voice mail",
    "all circuits are busy",
    "record your message",
    "i'll get back to you",
    "call me back",
    "our office is currently closed",
    "please record",
    "at the tone",
];

/// Heuristic voicemail detector (see module docs). Emits a single
/// [`VoicemailDecision`] then forwards everything unchanged.
pub struct VoicemailDetector {
    name: &'static str,
    params: VoicemailParams,
    transcript: String,
    /// Set on the first `VadUserStartedSpeaking`; cleared on stop.
    turn_started_secs: Option<f32>,
    first_turn: bool,
    decided: bool,
}

impl VoicemailDetector {
    /// A detector with the given params.
    pub fn new(params: VoicemailParams) -> Self {
        Self {
            name: "voicemail-detector",
            params,
            transcript: String::new(),
            turn_started_secs: None,
            first_turn: true,
            decided: false,
        }
    }

    /// A detector with default params.
    pub fn with_defaults() -> Self {
        Self::new(VoicemailParams::default())
    }

    /// Pure classifier over an accumulated transcript: matches against the
    /// voicemail phrase set (exposed for unit testing without the frame loop).
    pub fn classify_transcript(transcript: &str) -> Option<DecisionReason> {
        let lower = transcript.to_lowercase();
        if VOICEMAIL_PHRASES.iter().any(|p| lower.contains(p)) {
            Some(DecisionReason::GreetingPhrase)
        } else {
            None
        }
    }

    /// Build the decision frame and mark the detector inert.
    fn decide(&mut self, classification: Classification, reason: DecisionReason) -> Frame {
        self.decided = true;
        Frame::Custom(Arc::new(VoicemailDecision {
            classification,
            reason,
        }))
    }
}

#[async_trait]
impl FrameProcessor for VoicemailDetector {
    fn name(&self) -> &str {
        self.name
    }

    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        if self.decided {
            // Inert: forward unchanged.
            link.push(env.meta, env.frame, env.direction).await;
            return Ok(());
        }

        let mut decision: Option<Frame> = None;

        match &env.frame {
            Frame::Transcription { text, .. } | Frame::InterimTranscription { text, .. } => {
                if !self.transcript.is_empty() {
                    self.transcript.push(' ');
                }
                self.transcript.push_str(text);
                if let Some(reason) = Self::classify_transcript(&self.transcript) {
                    decision = Some(self.decide(Classification::Voicemail, reason));
                } else if self.first_turn
                    && self.transcript.split_whitespace().count()
                        >= self.params.long_first_turn_words
                {
                    // A long first-turn transcript with no human cue → greeting.
                    decision =
                        Some(self.decide(Classification::Voicemail, DecisionReason::LongFirstTurn));
                }
            }
            Frame::VadUserStartedSpeaking { start_secs } if self.turn_started_secs.is_none() => {
                self.turn_started_secs = Some(*start_secs);
            }
            Frame::VadUserStoppedSpeaking { stop_secs } => {
                // Evaluate the just-finished turn's duration.
                if let Some(start) = self.turn_started_secs.take() {
                    let dur = Duration::from_secs_f32((*stop_secs - start).max(0.0));
                    if self.first_turn && dur >= self.params.long_first_turn {
                        decision = Some(
                            self.decide(Classification::Voicemail, DecisionReason::LongFirstTurn),
                        );
                    }
                }
                // If the first turn ended without any voicemail signal, the far end
                // behaved like a human (short turn) → classify conversation.
                if decision.is_none() && self.first_turn {
                    decision = Some(
                        self.decide(Classification::Conversation, DecisionReason::HumanResponse),
                    );
                }
                self.first_turn = false;
            }
            _ => {}
        }

        // Forward the original frame first (the decision is sideband signalling).
        let direction = env.direction;
        link.push(env.meta, env.frame, direction).await;
        if let Some(d) = decision {
            link.push_down(d).await;
        }
        Ok(())
    }
}

/// Downcast helper: pull a [`VoicemailDecision`] out of a [`Frame::Custom`].
pub fn as_voicemail_decision(frame: &Frame) -> Option<&VoicemailDecision> {
    match frame {
        Frame::Custom(c) => c.as_any().downcast_ref::<VoicemailDecision>(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::test_harness::drive;
    use crate::processor::frame::Direction;

    fn transcription(text: &str) -> Frame {
        Frame::Transcription {
            text: text.into(),
            user_id: Arc::from("far-end"),
            language: None,
            final_: true,
        }
    }

    #[test]
    fn classify_transcript_matches_voicemail_phrases() {
        assert_eq!(
            VoicemailDetector::classify_transcript(
                "Hi, you've reached Jane, please leave a message"
            ),
            Some(DecisionReason::GreetingPhrase)
        );
        assert_eq!(
            VoicemailDetector::classify_transcript("The number you have dialed is not in service"),
            Some(DecisionReason::GreetingPhrase)
        );
        assert_eq!(
            VoicemailDetector::classify_transcript("Hello? Who is this?"),
            None
        );
    }

    #[tokio::test]
    async fn detects_voicemail_from_greeting_phrase() {
        let out = drive(
            Box::new(VoicemailDetector::with_defaults()),
            vec![transcription(
                "Hi, you've reached the Smiths. Please leave a message after the beep.",
            )],
            Direction::Downstream,
        )
        .await;
        let decision = out
            .iter()
            .find_map(as_voicemail_decision)
            .expect("no decision emitted");
        assert_eq!(decision.classification, Classification::Voicemail);
        assert_eq!(decision.reason, DecisionReason::GreetingPhrase);
    }

    #[tokio::test]
    async fn classifies_human_on_short_first_turn() {
        // VAD edges with a short turn (0.3s) and a non-voicemail transcript.
        let out = drive(
            Box::new(VoicemailDetector::with_defaults()),
            vec![
                Frame::VadUserStartedSpeaking { start_secs: 0.0 },
                transcription("Hello?"),
                Frame::VadUserStoppedSpeaking { stop_secs: 0.3 },
            ],
            Direction::Downstream,
        )
        .await;
        let decision = out
            .iter()
            .find_map(as_voicemail_decision)
            .expect("no decision emitted");
        assert_eq!(decision.classification, Classification::Conversation);
        assert_eq!(decision.reason, DecisionReason::HumanResponse);
    }

    #[tokio::test]
    async fn detects_voicemail_from_long_first_turn_vad() {
        // A long uninterrupted first turn (6s) with no human cue → voicemail.
        let out = drive(
            Box::new(VoicemailDetector::with_defaults()),
            vec![
                Frame::VadUserStartedSpeaking { start_secs: 0.0 },
                Frame::VadUserStoppedSpeaking { stop_secs: 6.0 },
            ],
            Direction::Downstream,
        )
        .await;
        let decision = out
            .iter()
            .find_map(as_voicemail_decision)
            .expect("no decision emitted");
        assert_eq!(decision.classification, Classification::Voicemail);
        assert_eq!(decision.reason, DecisionReason::LongFirstTurn);
    }

    #[tokio::test]
    async fn decides_only_once_then_forwards() {
        let out = drive(
            Box::new(VoicemailDetector::with_defaults()),
            vec![
                transcription("please leave a message"), // decides voicemail
                transcription("you've reached"),         // must NOT decide again
            ],
            Direction::Downstream,
        )
        .await;
        let decisions = out.iter().filter_map(as_voicemail_decision).count();
        assert_eq!(decisions, 1, "detector must decide exactly once");
        // Both original transcription frames still forwarded.
        let transcripts = out
            .iter()
            .filter(|f| matches!(f, Frame::Transcription { .. }))
            .count();
        assert_eq!(transcripts, 2);
    }
}
