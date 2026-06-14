// SPDX-License-Identifier: Apache-2.0
//
//! Transcript collection.
//!
//! Accumulates the turn-by-turn transcript from the realtime model's
//! input/output transcription events and renders it for upload as a transcript
//! artifact (see DESIGN.md "Audio path" / finalize).

use serde::{Deserialize, Serialize};

/// Who produced a transcript line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Speaker {
    /// The human caller (input transcription).
    User,
    /// The bot / model (output transcription).
    Bot,
}

/// A single transcript line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TranscriptLine {
    /// Who spoke.
    pub speaker: Speaker,
    /// The (possibly incrementally-assembled) text of the line.
    pub text: String,
}

/// Accumulates transcript lines over the life of a call.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Transcript {
    /// The ordered transcript lines.
    pub lines: Vec<TranscriptLine>,
}

impl Transcript {
    /// Create an empty transcript.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append (or merge into) a user line from an input-transcription delta.
    ///
    /// The realtime model streams transcription incrementally; consecutive
    /// deltas from the *same* speaker are appended into the current line so a
    /// rendered turn reads as one utterance. A speaker switch starts a new line.
    pub fn push_user(&mut self, text: &str) {
        self.push(Speaker::User, text);
    }

    /// Append (or merge into) a bot line from an output-transcription delta.
    pub fn push_bot(&mut self, text: &str) {
        self.push(Speaker::Bot, text);
    }

    /// Shared delta-merge: extend the trailing line if the speaker matches,
    /// otherwise begin a fresh line. Empty deltas are dropped.
    fn push(&mut self, speaker: Speaker, text: &str) {
        if text.is_empty() {
            return;
        }
        match self.lines.last_mut() {
            Some(last) if last.speaker == speaker => last.text.push_str(text),
            _ => self.lines.push(TranscriptLine {
                speaker,
                text: text.to_owned(),
            }),
        }
    }

    /// Number of user turns (merged user lines) — reported as
    /// `usage_metrics.user_turns` for the Run-detail "User Turns" metric.
    pub fn user_turns(&self) -> u64 {
        self.lines
            .iter()
            .filter(|l| l.speaker == Speaker::User)
            .count() as u64
    }

    /// Number of bot turns (merged bot lines) — reported as
    /// `usage_metrics.bot_turns`. (Transition markers ride on a bot line, so this
    /// is the count of bot utterances including any folded transition note.)
    pub fn bot_turns(&self) -> u64 {
        self.lines
            .iter()
            .filter(|l| l.speaker == Speaker::Bot)
            .count() as u64
    }

    /// Render the transcript as a JSON array of `{role, text}` turns.
    ///
    /// `role` is the lowercase speaker (`"user"`/`"bot"`). Never panics —
    /// serialization of this owned shape cannot fail, but on the off chance it
    /// did we fall back to an empty array so a finalize can still proceed.
    pub fn render(&self) -> Vec<u8> {
        let turns: Vec<Turn> = self
            .lines
            .iter()
            .map(|l| Turn {
                role: match l.speaker {
                    Speaker::User => "user",
                    Speaker::Bot => "bot",
                },
                text: &l.text,
            })
            .collect();
        serde_json::to_vec(&turns).unwrap_or_else(|_| b"[]".to_vec())
    }
}

/// A rendered transcript turn: `{role, text}` (the JSON artifact shape).
#[derive(Debug, Serialize)]
struct Turn<'a> {
    /// Lowercase speaker role: `"user"` or `"bot"`.
    role: &'a str,
    /// The full text of the turn.
    text: &'a str,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn empty_transcript_renders_empty_json_array() {
        let t = Transcript::new();
        assert_eq!(t.render(), b"[]".to_vec());
    }

    #[test]
    fn consecutive_same_speaker_deltas_merge_into_one_line() {
        let mut t = Transcript::new();
        t.push_user("hel");
        t.push_user("lo ");
        t.push_user("there");
        assert_eq!(t.lines.len(), 1);
        assert_eq!(t.lines[0].speaker, Speaker::User);
        assert_eq!(t.lines[0].text, "hello there");
    }

    #[test]
    fn speaker_switch_starts_a_new_line() {
        let mut t = Transcript::new();
        t.push_user("hi");
        t.push_bot("hello, how can I help?");
        t.push_user("a question");
        assert_eq!(t.lines.len(), 3);
        assert_eq!(t.lines[0].speaker, Speaker::User);
        assert_eq!(t.lines[1].speaker, Speaker::Bot);
        assert_eq!(t.lines[2].speaker, Speaker::User);
    }

    #[test]
    fn empty_deltas_are_dropped() {
        let mut t = Transcript::new();
        t.push_user("");
        t.push_bot("");
        assert!(t.lines.is_empty());
        // An empty delta between real deltas must not split a line.
        t.push_user("ab");
        t.push_user("");
        t.push_user("cd");
        assert_eq!(t.lines.len(), 1);
        assert_eq!(t.lines[0].text, "abcd");
    }

    #[test]
    fn turn_counts_tally_lines_per_speaker() {
        let mut t = Transcript::new();
        t.push_bot("hello, how can I help?");
        t.push_user("book an appointment");
        t.push_bot("sure, what day?");
        t.push_user("tuesday"); // merged deltas count as ONE turn
        t.push_user(" afternoon");
        assert_eq!(t.user_turns(), 2, "two user lines");
        assert_eq!(t.bot_turns(), 2, "two bot lines");
        let empty = Transcript::new();
        assert_eq!(empty.user_turns(), 0);
        assert_eq!(empty.bot_turns(), 0);
    }

    #[test]
    fn render_is_role_text_json_array() {
        let mut t = Transcript::new();
        t.push_user("hi");
        t.push_bot("hello");
        let json: Value = serde_json::from_slice(&t.render()).unwrap();
        assert_eq!(
            json,
            serde_json::json!([
                { "role": "user", "text": "hi" },
                { "role": "bot", "text": "hello" },
            ])
        );
    }
}
