// SPDX-License-Identifier: Apache-2.0
//
//! IVR-navigator state machine (pure logic, no network).
//!
//! pipecat's `IVRNavigator` drives menu traversal with an *LLM* (it asks a model
//! which key to press). That needs a network LLM, so it is out of this pure-logic
//! scope. The flowcat navigator is the **deterministic** half: a scripted
//! state machine that matches each incoming transcript ([`Frame::Transcription`])
//! against a configured target path and emits the matching action — DTMF
//! ([`Frame::OutputDtmf`]) or a spoken reply ([`Frame::TtsSpeak`]) — to navigate
//! the menu, advancing step by step until a terminal state.
//!
//! A [`Script`] is an ordered list of [`IvrStep`]s. Each step has prompt
//! *triggers* (substrings to look for in the menu's transcript) and an
//! [`IvrAction`] to take when triggered. After acting on a step the navigator
//! advances to the next; when it runs off the end it reports
//! [`IvrStatus::Completed`]. If a configured number of unmatched prompts pass with
//! no progress it reports [`IvrStatus::Stuck`]. Status changes are emitted as a
//! [`Frame::Custom`] carrying an [`IvrStatusChanged`] (no `Frame` variant is
//! promoted at this layer).

use std::any::Any;
use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Result;
use crate::processor::frame::{CustomFrame, Frame, FrameClass, KeypadEntry};
use crate::processor::{Envelope, FrameProcessor, Link};

/// The action a matched [`IvrStep`] takes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IvrAction {
    /// Press these keypad digits in order (DTMF).
    Dtmf(Vec<KeypadEntry>),
    /// Speak this verbal response (for "Say yes or no"-style prompts).
    Say(String),
    /// No output — the step only marks that this prompt was reached (e.g. an
    /// acknowledgement before the next menu).
    Wait,
}

/// One step in an IVR [`Script`]: when any `triggers` substring appears in the
/// menu transcript, take `action` and advance.
#[derive(Debug, Clone)]
pub struct IvrStep {
    /// Case-insensitive substrings; any one present in the prompt fires the step.
    pub triggers: Vec<String>,
    /// What to do when this step's prompt is reached.
    pub action: IvrAction,
}

impl IvrStep {
    /// A step that presses `digits` when any of `triggers` is heard.
    pub fn press(triggers: &[&str], digits: Vec<KeypadEntry>) -> Self {
        Self {
            triggers: triggers.iter().map(|s| s.to_lowercase()).collect(),
            action: IvrAction::Dtmf(digits),
        }
    }

    /// A step that speaks `reply` when any of `triggers` is heard.
    pub fn say(triggers: &[&str], reply: impl Into<String>) -> Self {
        Self {
            triggers: triggers.iter().map(|s| s.to_lowercase()).collect(),
            action: IvrAction::Say(reply.into()),
        }
    }

    fn matches(&self, prompt_lower: &str) -> bool {
        self.triggers.iter().any(|t| prompt_lower.contains(t))
    }
}

/// An ordered IVR traversal plan.
#[derive(Debug, Clone, Default)]
pub struct Script {
    steps: Vec<IvrStep>,
    /// How many consecutive unmatched prompts before declaring [`IvrStatus::Stuck`].
    stuck_after: usize,
}

impl Script {
    /// A new script from `steps`, declaring "stuck" after `stuck_after`
    /// consecutive prompts that match no remaining step.
    pub fn new(steps: Vec<IvrStep>, stuck_after: usize) -> Self {
        Self { steps, stuck_after }
    }
}

/// IVR navigation status (mirrors pipecat `IVRStatus`, minus the LLM-only `Wait`
/// which is handled inline as a no-op step).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IvrStatus {
    /// Navigation finished — every step in the path was taken.
    Completed,
    /// No remaining step matched after `stuck_after` prompts.
    Stuck,
}

/// Status-change signal emitted as a [`Frame::Custom`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IvrStatusChanged {
    /// The new status.
    pub status: IvrStatus,
}

impl CustomFrame for IvrStatusChanged {
    fn frame_class(&self) -> FrameClass {
        FrameClass::Control
    }
    fn name(&self) -> &'static str {
        "IvrStatusChanged"
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A scripted IVR navigator (see module docs). Stateless across calls but holds
/// per-call traversal position.
pub struct IvrNavigator {
    name: &'static str,
    script: Script,
    cursor: usize,
    unmatched_streak: usize,
    finished: bool,
}

impl IvrNavigator {
    /// A navigator that follows `script`.
    pub fn new(script: Script) -> Self {
        Self {
            name: "ivr-navigator",
            script,
            cursor: 0,
            unmatched_streak: 0,
            finished: false,
        }
    }

    /// Whether the navigator has reached a terminal state.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Turn an [`IvrAction`] into the output frame it emits, if any.
    fn action_frame(action: &IvrAction) -> Option<Frame> {
        match action {
            IvrAction::Dtmf(digits) => Some(Frame::OutputDtmf(digits.clone())),
            IvrAction::Say(reply) => Some(Frame::TtsSpeak {
                text: reply.clone(),
                append_to_context: Some(false),
            }),
            IvrAction::Wait => None,
        }
    }
}

#[async_trait]
impl FrameProcessor for IvrNavigator {
    fn name(&self) -> &str {
        self.name
    }

    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        let Frame::Transcription { ref text, .. } = env.frame else {
            link.push(env.meta, env.frame, env.direction).await;
            return Ok(());
        };

        // Forward the prompt transcript itself, then act on it.
        let direction = env.direction;
        let prompt_lower = text.to_lowercase();
        link.push(env.meta, env.frame, direction).await;

        if self.finished {
            return Ok(());
        }

        // Does the current step match this prompt? (We only ever try the step at
        // the cursor; menus are visited in order.)
        if let Some(step) = self.script.steps.get(self.cursor) {
            if step.matches(&prompt_lower) {
                self.unmatched_streak = 0;
                if let Some(frame) = Self::action_frame(&step.action) {
                    link.push_down(frame).await;
                }
                self.cursor += 1;
                if self.cursor >= self.script.steps.len() {
                    self.finished = true;
                    link.push_down(Frame::Custom(Arc::new(IvrStatusChanged {
                        status: IvrStatus::Completed,
                    })))
                    .await;
                }
                return Ok(());
            }
        }

        // No match for the current step on this prompt.
        self.unmatched_streak += 1;
        if self.script.stuck_after > 0 && self.unmatched_streak >= self.script.stuck_after {
            self.finished = true;
            link.push_down(Frame::Custom(Arc::new(IvrStatusChanged {
                status: IvrStatus::Stuck,
            })))
            .await;
        }
        Ok(())
    }
}

/// Downcast helper: pull an [`IvrStatusChanged`] out of a [`Frame::Custom`].
pub fn as_ivr_status(frame: &Frame) -> Option<&IvrStatusChanged> {
    match frame {
        Frame::Custom(c) => c.as_any().downcast_ref::<IvrStatusChanged>(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::test_harness::drive;
    use crate::processor::frame::Direction;

    fn prompt(text: &str) -> Frame {
        Frame::Transcription {
            text: text.into(),
            user_id: Arc::from("ivr"),
            language: None,
            final_: true,
        }
    }

    fn dtmf_of(frame: &Frame) -> Option<Vec<KeypadEntry>> {
        match frame {
            Frame::OutputDtmf(d) => Some(d.clone()),
            _ => None,
        }
    }

    #[tokio::test]
    async fn navigates_a_menu_path_and_completes() {
        // Goal: reach billing → press 2, then enter account "1" via DTMF, then
        // answer a verbal confirmation.
        let script = Script::new(
            vec![
                IvrStep::press(
                    &["press 2 for billing", "for billing press 2"],
                    vec![KeypadEntry::Two],
                ),
                IvrStep::press(&["enter your account number"], vec![KeypadEntry::One]),
                IvrStep::say(&["say yes or no", "is this correct"], "Yes."),
            ],
            3,
        );
        let out = drive(
            Box::new(IvrNavigator::new(script)),
            vec![
                prompt("Thank you for calling. Press 1 for sales, press 2 for billing."),
                prompt("Please enter your account number followed by pound."),
                prompt("I heard account one. Is this correct? Say yes or no."),
            ],
            Direction::Downstream,
        )
        .await;

        // First step pressed 2.
        let dtmfs: Vec<Vec<KeypadEntry>> = out.iter().filter_map(dtmf_of).collect();
        assert_eq!(dtmfs, vec![vec![KeypadEntry::Two], vec![KeypadEntry::One]]);
        // Third step spoke "Yes."
        let said = out
            .iter()
            .any(|f| matches!(f, Frame::TtsSpeak { text, .. } if text == "Yes."));
        assert!(said, "verbal step did not speak: {out:?}");
        // Completed status emitted.
        let status = out.iter().find_map(as_ivr_status).expect("no status");
        assert_eq!(status.status, IvrStatus::Completed);
    }

    #[tokio::test]
    async fn waits_through_irrelevant_prompts_then_matches() {
        let script = Script::new(
            vec![IvrStep::press(
                &["press 0 for an agent"],
                vec![KeypadEntry::Zero],
            )],
            5,
        );
        let out = drive(
            Box::new(IvrNavigator::new(script)),
            vec![
                prompt("Your call is important to us."), // no match → wait
                prompt("Please continue to hold."),      // no match → wait
                prompt("Or press 0 for an agent."),      // match → press 0
            ],
            Direction::Downstream,
        )
        .await;
        let dtmfs: Vec<Vec<KeypadEntry>> = out.iter().filter_map(dtmf_of).collect();
        assert_eq!(dtmfs, vec![vec![KeypadEntry::Zero]]);
        let status = out.iter().find_map(as_ivr_status).expect("no status");
        assert_eq!(status.status, IvrStatus::Completed);
    }

    #[tokio::test]
    async fn reports_stuck_after_unmatched_streak() {
        let script = Script::new(
            vec![IvrStep::press(
                &["press 9 for spanish"],
                vec![KeypadEntry::Nine],
            )],
            2, // stuck after 2 consecutive unmatched prompts
        );
        let out = drive(
            Box::new(IvrNavigator::new(script)),
            vec![
                prompt("Welcome to the help line."), // unmatched 1
                prompt("All agents are busy."),      // unmatched 2 → stuck
                prompt("press 9 for spanish"),       // already finished → ignored
            ],
            Direction::Downstream,
        )
        .await;
        let status = out.iter().find_map(as_ivr_status).expect("no status");
        assert_eq!(status.status, IvrStatus::Stuck);
        // No DTMF after stuck.
        assert!(out.iter().filter_map(dtmf_of).next().is_none());
    }

    #[tokio::test]
    async fn non_transcription_frames_pass_through() {
        let script = Script::new(vec![IvrStep::press(&["x"], vec![KeypadEntry::One])], 3);
        let out = drive(
            Box::new(IvrNavigator::new(script)),
            vec![Frame::Text("hi".into())],
            Direction::Downstream,
        )
        .await;
        assert!(matches!(out.first(), Some(Frame::Text(t)) if t == "hi"));
    }
}
