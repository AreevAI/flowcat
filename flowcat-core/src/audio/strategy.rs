// SPDX-License-Identifier: Apache-2.0
//
//! Turn start/stop/mute + interruption strategies (pure framework logic).
//!
//! These mirror pipecat's `pipecat/turns/user_start`, `user_stop`, `user_mute`
//! and the interruption-strategy family. They are **pure, synchronous decision
//! functions** with no heavy dep, no I/O, and no `async`: each consumes the
//! lifecycle frames a [`VadProcessor`](crate::audio::vad)/`TurnProcessor` already
//! emits ([`Frame::VadUserStartedSpeaking`], [`Frame::Transcription`],
//! [`Frame::BotStartedSpeaking`], …) and returns a decision. The owning processor
//! (the turn controller in the cascaded path) turns those decisions into
//! [`Frame::UserStartedSpeaking`]/[`Frame::UserStoppedSpeaking`]/[`Frame::SttMute`]
//! /[`Frame::Interruption`] frames.
//!
//! Keeping them pure (vs. baked into the processor) means they are exhaustively
//! unit-testable and composable — exactly pipecat's split of "strategy" vs
//! "processor". The `*_secs`/timeout-driven branches are modeled as **explicit
//! tick inputs** ([`TurnStopStrategy::on_silence_tick`]) instead of `asyncio`
//! timers, so the logic is deterministic and clock-free; the processor feeds
//! wall-clock ticks.

use crate::processor::frame::Frame;

/// What a turn-*start* strategy decides for the frame it just saw. Mirrors
/// pipecat's `ProcessFrameResult` (`turns/types.py`): `Triggered` ⇒ the user turn
/// has started (emit `UserStartedSpeaking` + interrupt the bot); `Continue` ⇒
/// keep evaluating later strategies / frames; `ResetAggregation` ⇒ a partial
/// transcript fell below the word floor, drop the accumulated text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnStartDecision {
    /// The user turn has started.
    Triggered,
    /// No decision yet — keep going.
    Continue,
    /// Discard the in-progress user aggregation (a sub-threshold interim).
    ResetAggregation,
}

/// A user-turn **start** strategy: decides when the user has begun a turn.
/// Mirrors pipecat `BaseUserTurnStartStrategy`. Pure + synchronous.
pub trait TurnStartStrategy: Send {
    /// Inspect one frame and decide. Stateful strategies mutate `self`.
    fn on_frame(&mut self, frame: &Frame) -> TurnStartDecision;
    /// Reset to the initial state at the start of a new turn.
    fn reset(&mut self);
}

/// Start the turn the instant VAD says the user started speaking. Mirrors
/// `VADUserTurnStartStrategy`. The simplest, lowest-latency strategy — the
/// default for the cascaded path.
#[derive(Debug, Default, Clone)]
pub struct VadTurnStart;

impl TurnStartStrategy for VadTurnStart {
    fn on_frame(&mut self, frame: &Frame) -> TurnStartDecision {
        match frame {
            Frame::VadUserStartedSpeaking { .. } => TurnStartDecision::Triggered,
            _ => TurnStartDecision::Continue,
        }
    }
    fn reset(&mut self) {}
}

/// Start the turn once the user has spoken at least `min_words` (counted from
/// transcription frames). While the bot is speaking the full `min_words` floor
/// applies (avoid spurious barge-in on a single back-channel word); once the bot
/// is silent a **single** word triggers. Mirrors `MinWordsUserTurnStartStrategy`.
#[derive(Debug, Clone)]
pub struct MinWordsTurnStart {
    min_words: usize,
    use_interim: bool,
    bot_speaking: bool,
}

impl MinWordsTurnStart {
    /// `min_words` while the bot is speaking; `use_interim` counts interim
    /// transcripts too (earlier, noisier detection).
    pub fn new(min_words: usize, use_interim: bool) -> Self {
        Self {
            min_words,
            use_interim,
            bot_speaking: false,
        }
    }

    fn handle_transcription(&self, text: &str) -> TurnStartDecision {
        // When the bot is silent a single word is enough to start a turn.
        let floor = if self.bot_speaking {
            self.min_words.max(1)
        } else {
            1
        };
        let words = text.split_whitespace().count();
        if words >= floor {
            TurnStartDecision::Triggered
        } else {
            TurnStartDecision::ResetAggregation
        }
    }
}

impl TurnStartStrategy for MinWordsTurnStart {
    fn on_frame(&mut self, frame: &Frame) -> TurnStartDecision {
        match frame {
            Frame::BotStartedSpeaking => {
                self.bot_speaking = true;
                TurnStartDecision::Continue
            }
            Frame::BotStoppedSpeaking => {
                self.bot_speaking = false;
                TurnStartDecision::Continue
            }
            Frame::Transcription { text, .. } => self.handle_transcription(text),
            Frame::InterimTranscription { text, .. } if self.use_interim => {
                self.handle_transcription(text)
            }
            _ => TurnStartDecision::Continue,
        }
    }
    fn reset(&mut self) {
        self.bot_speaking = false;
    }
}

/// A turn start driven entirely by an out-of-band signal (e.g. a client
/// "push-to-talk" message) — VAD/transcription frames never trigger it. Mirrors
/// `ExternalUserTurnStartStrategy`. The processor calls [`Self::trigger`] when the
/// external signal arrives.
#[derive(Debug, Default, Clone)]
pub struct ExternalTurnStart {
    triggered: bool,
}

impl ExternalTurnStart {
    /// Arm the external trigger; the next `on_frame` reports `Triggered` once.
    pub fn trigger(&mut self) {
        self.triggered = true;
    }
}

impl TurnStartStrategy for ExternalTurnStart {
    fn on_frame(&mut self, _frame: &Frame) -> TurnStartDecision {
        if self.triggered {
            self.triggered = false;
            TurnStartDecision::Triggered
        } else {
            TurnStartDecision::Continue
        }
    }
    fn reset(&mut self) {
        self.triggered = false;
    }
}

/// What a turn-*stop* strategy decides. `Triggered` ⇒ the user turn is over (emit
/// `UserStoppedSpeaking`, run the LLM); `Waiting` ⇒ all conditions not yet met.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnStopDecision {
    /// The user turn has ended.
    Triggered,
    /// Conditions not met yet — keep waiting (more audio / a transcript / a tick).
    Waiting,
}

/// A user-turn **stop** strategy: decides when the user has finished a turn.
/// Mirrors pipecat `BaseUserTurnStopStrategy`. The wall-clock timers pipecat runs
/// as `asyncio` tasks are modeled here as explicit
/// [`on_silence_tick`](TurnStopStrategy::on_silence_tick) calls the processor
/// drives, so the logic is deterministic and clock-free.
pub trait TurnStopStrategy: Send {
    /// Inspect one frame and (maybe) decide the turn ended.
    fn on_frame(&mut self, frame: &Frame) -> TurnStopDecision;
    /// Advance `elapsed_ms` of post-VAD-stop silence; may push past a timeout.
    fn on_silence_tick(&mut self, elapsed_ms: f32) -> TurnStopDecision;
    /// Reset for the next turn.
    fn reset(&mut self);
}

/// End the turn after the user stops speaking (VAD) **and** the turn analyzer says
/// the turn is semantically complete **and** a transcript has arrived. A finalized
/// transcript triggers immediately; otherwise the configured `stt_timeout_ms` of
/// post-stop silence is the fallback. Mirrors `TurnAnalyzerUserTurnStopStrategy`
/// (the timeout-driven path, reduced to its decision core).
#[derive(Debug, Clone)]
pub struct TurnAnalyzerStop {
    /// Post-VAD-stop silence (ms) after which a non-finalized transcript still
    /// triggers (the STT-latency safety net).
    stt_timeout_ms: f32,
    vad_user_speaking: bool,
    /// Turn analyzer reported the turn complete (set by feeding turn predictions).
    turn_complete: bool,
    have_text: bool,
    transcript_finalized: bool,
    timeout_expired: bool,
}

impl TurnAnalyzerStop {
    /// `stt_timeout_ms`: post-stop wait for a final transcript before triggering.
    pub fn new(stt_timeout_ms: f32) -> Self {
        Self {
            stt_timeout_ms,
            vad_user_speaking: false,
            turn_complete: false,
            have_text: false,
            transcript_finalized: false,
            timeout_expired: false,
        }
    }

    /// Feed an end-of-turn prediction from the turn analyzer (the processor calls
    /// this when its `TurnAnalyzer` returns a complete/incomplete result).
    pub fn set_turn_complete(&mut self, complete: bool) -> TurnStopDecision {
        self.turn_complete = complete;
        self.evaluate()
    }

    fn evaluate(&self) -> TurnStopDecision {
        if !self.have_text || !self.turn_complete || self.vad_user_speaking {
            return TurnStopDecision::Waiting;
        }
        // Finalized transcript ⇒ trigger now; else wait for the silence timeout.
        if self.transcript_finalized || self.timeout_expired {
            TurnStopDecision::Triggered
        } else {
            TurnStopDecision::Waiting
        }
    }
}

impl TurnStopStrategy for TurnAnalyzerStop {
    fn on_frame(&mut self, frame: &Frame) -> TurnStopDecision {
        match frame {
            Frame::VadUserStartedSpeaking { .. } => {
                self.vad_user_speaking = true;
                self.turn_complete = false;
                self.transcript_finalized = false;
                self.timeout_expired = false;
                TurnStopDecision::Waiting
            }
            Frame::VadUserStoppedSpeaking { .. } => {
                self.vad_user_speaking = false;
                self.evaluate()
            }
            Frame::Transcription { text, final_, .. } => {
                if !text.is_empty() {
                    self.have_text = true;
                }
                if *final_ {
                    self.transcript_finalized = true;
                }
                self.evaluate()
            }
            _ => TurnStopDecision::Waiting,
        }
    }

    fn on_silence_tick(&mut self, elapsed_ms: f32) -> TurnStopDecision {
        if !self.vad_user_speaking && elapsed_ms >= self.stt_timeout_ms {
            self.timeout_expired = true;
        }
        self.evaluate()
    }

    fn reset(&mut self) {
        self.vad_user_speaking = false;
        self.turn_complete = false;
        self.have_text = false;
        self.transcript_finalized = false;
        self.timeout_expired = false;
    }
}

/// End the turn purely on a silence timeout after the user stops speaking, once at
/// least one transcript exists. Two independent waits (the policy floor
/// `user_speech_timeout_ms` and the STT safety-net `stt_timeout_ms`) must both
/// elapse — a finalized transcript short-circuits the STT wait. Mirrors
/// `SpeechTimeoutUserTurnStopStrategy`, reduced to its decision core.
#[derive(Debug, Clone)]
pub struct SpeechTimeoutStop {
    user_speech_timeout_ms: f32,
    stt_timeout_ms: f32,
    vad_user_speaking: bool,
    have_text: bool,
    transcript_finalized: bool,
    user_speech_done: bool,
    stt_done: bool,
}

impl SpeechTimeoutStop {
    /// `user_speech_timeout_ms`: floor window the user may resume in.
    /// `stt_timeout_ms`: STT-latency safety net (short-circuited on finalize).
    pub fn new(user_speech_timeout_ms: f32, stt_timeout_ms: f32) -> Self {
        Self {
            user_speech_timeout_ms,
            stt_timeout_ms,
            vad_user_speaking: false,
            have_text: false,
            transcript_finalized: false,
            user_speech_done: false,
            stt_done: false,
        }
    }

    fn evaluate(&self) -> TurnStopDecision {
        if self.vad_user_speaking || !self.have_text {
            return TurnStopDecision::Waiting;
        }
        if self.user_speech_done && self.stt_done {
            TurnStopDecision::Triggered
        } else {
            TurnStopDecision::Waiting
        }
    }
}

impl TurnStopStrategy for SpeechTimeoutStop {
    fn on_frame(&mut self, frame: &Frame) -> TurnStopDecision {
        match frame {
            Frame::VadUserStartedSpeaking { .. } => {
                self.vad_user_speaking = true;
                self.transcript_finalized = false;
                self.user_speech_done = false;
                self.stt_done = false;
                TurnStopDecision::Waiting
            }
            Frame::VadUserStoppedSpeaking { stop_secs } => {
                self.vad_user_speaking = false;
                // If VAD stop_secs already covered the STT wait, mark it done.
                if self.transcript_finalized || stop_secs * 1000.0 >= self.stt_timeout_ms {
                    self.stt_done = true;
                }
                self.evaluate()
            }
            Frame::Transcription { text, final_, .. } => {
                if !text.is_empty() {
                    self.have_text = true;
                }
                if *final_ {
                    self.transcript_finalized = true;
                    self.stt_done = true;
                }
                self.evaluate()
            }
            _ => TurnStopDecision::Waiting,
        }
    }

    fn on_silence_tick(&mut self, elapsed_ms: f32) -> TurnStopDecision {
        if !self.vad_user_speaking {
            if elapsed_ms >= self.user_speech_timeout_ms {
                self.user_speech_done = true;
            }
            if elapsed_ms >= self.stt_timeout_ms {
                self.stt_done = true;
            }
        }
        self.evaluate()
    }

    fn reset(&mut self) {
        self.vad_user_speaking = false;
        self.have_text = false;
        self.transcript_finalized = false;
        self.user_speech_done = false;
        self.stt_done = false;
    }
}

/// A user-**mute** strategy: decides whether inbound user audio/transcription
/// should be suppressed given the current system state. Mirrors pipecat
/// `BaseUserMuteStrategy`. Pure + synchronous; the processor emits
/// [`Frame::SttMute`] when [`MuteStrategy::muted`] flips.
pub trait MuteStrategy: Send {
    /// Update state from one frame, then return whether the user is muted.
    fn on_frame(&mut self, frame: &Frame) -> bool;
    /// Current muted state (without consuming a frame).
    fn muted(&self) -> bool;
}

/// Always mute. Mirrors `AlwaysUserMuteStrategy` (bot fully owns the floor).
#[derive(Debug, Default, Clone)]
pub struct AlwaysMute;

impl MuteStrategy for AlwaysMute {
    fn on_frame(&mut self, _frame: &Frame) -> bool {
        true
    }
    fn muted(&self) -> bool {
        true
    }
}

/// Mute from the start of the call until the bot finishes its **first** speaking
/// turn; then never again. Mirrors `MuteUntilFirstBotCompleteUserMuteStrategy`.
#[derive(Debug, Default, Clone)]
pub struct MuteUntilFirstBotComplete {
    first_speech_handled: bool,
}

impl MuteStrategy for MuteUntilFirstBotComplete {
    fn on_frame(&mut self, frame: &Frame) -> bool {
        if let Frame::BotStoppedSpeaking = frame {
            self.first_speech_handled = true;
        }
        self.muted()
    }
    fn muted(&self) -> bool {
        !self.first_speech_handled
    }
}

/// Mute the user **only during** the bot's first speaking turn (early user input
/// before the bot speaks is allowed). Mirrors `FirstSpeechUserMuteStrategy`.
#[derive(Debug, Default, Clone)]
pub struct FirstSpeechMute {
    bot_speaking: bool,
    first_speech_handled: bool,
}

impl MuteStrategy for FirstSpeechMute {
    fn on_frame(&mut self, frame: &Frame) -> bool {
        match frame {
            Frame::BotStartedSpeaking => self.bot_speaking = true,
            Frame::BotStoppedSpeaking => {
                self.bot_speaking = false;
                self.first_speech_handled = true;
            }
            _ => {}
        }
        self.muted()
    }
    fn muted(&self) -> bool {
        self.bot_speaking && !self.first_speech_handled
    }
}

/// Mute the user while **any** function call is in flight; unmute when every call
/// has resolved (result) or cancelled. Mirrors `FunctionCallUserMuteStrategy`.
#[derive(Debug, Default, Clone)]
pub struct FunctionCallMute {
    in_progress: std::collections::HashSet<String>,
}

impl MuteStrategy for FunctionCallMute {
    fn on_frame(&mut self, frame: &Frame) -> bool {
        match frame {
            Frame::FunctionCallsStarted(calls) => {
                for c in calls {
                    self.in_progress.insert(c.tool_call_id.clone());
                }
            }
            Frame::FunctionCallResult(r) => {
                self.in_progress.remove(&r.tool_call_id);
            }
            Frame::FunctionCallCancel { tool_call_id, .. } => {
                self.in_progress.remove(tool_call_id);
            }
            _ => {}
        }
        self.muted()
    }
    fn muted(&self) -> bool {
        !self.in_progress.is_empty()
    }
}

/// Whether a detected user turn during bot speech should actually interrupt the
/// bot. Mirrors pipecat's interruption-strategy family: a turn-start
/// (`UserStartedSpeaking`) is *proposed* by a [`TurnStartStrategy`], and an
/// interruption strategy is the final gate deciding whether to broadcast a
/// [`Frame::Interruption`]. Pure + synchronous.
pub trait InterruptionStrategy: Send {
    /// Should the accumulated user speech-so-far interrupt the bot? Called when a
    /// turn-start is proposed while the bot is speaking.
    fn should_interrupt(&self) -> bool;
    /// Feed a transcription frame so word/char-count strategies accumulate.
    fn append_transcription(&mut self, frame: &Frame);
    /// Feed an audio frame's volume so amplitude strategies accumulate.
    fn append_audio_volume(&mut self, _volume: f32) {}
    /// Reset accumulated state after a (non-)interruption decision.
    fn reset(&mut self);
}

/// Always allow interruptions (the default; barge-in on any detected turn-start).
/// Mirrors pipecat's default of "no interruption strategy = always interrupt".
#[derive(Debug, Default, Clone)]
pub struct AlwaysInterrupt;

impl InterruptionStrategy for AlwaysInterrupt {
    fn should_interrupt(&self) -> bool {
        true
    }
    fn append_transcription(&mut self, _frame: &Frame) {}
    fn reset(&mut self) {}
}

/// Interrupt the bot only once the user has spoken at least `min_words` (counted
/// across the accumulated transcriptions of the in-progress turn). Mirrors
/// pipecat's `MinWordsInterruptionStrategy` — filters spurious one-word barge-ins.
#[derive(Debug, Clone)]
pub struct MinWordsInterrupt {
    min_words: usize,
    accumulated_words: usize,
}

impl MinWordsInterrupt {
    /// Require `min_words` accumulated before an interruption is allowed.
    pub fn new(min_words: usize) -> Self {
        Self {
            min_words,
            accumulated_words: 0,
        }
    }
}

impl InterruptionStrategy for MinWordsInterrupt {
    fn should_interrupt(&self) -> bool {
        self.accumulated_words >= self.min_words
    }
    fn append_transcription(&mut self, frame: &Frame) {
        match frame {
            Frame::Transcription { text, .. } | Frame::InterimTranscription { text, .. } => {
                self.accumulated_words += text.split_whitespace().count();
            }
            _ => {}
        }
    }
    fn reset(&mut self) {
        self.accumulated_words = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processor::frame::{FunctionCall, FunctionCallResult};
    use std::sync::Arc;

    fn transcription(text: &str, final_: bool) -> Frame {
        Frame::Transcription {
            text: text.into(),
            user_id: Arc::from("u"),
            language: None,
            final_,
        }
    }
    fn interim(text: &str) -> Frame {
        Frame::InterimTranscription {
            text: text.into(),
            user_id: Arc::from("u"),
            language: None,
        }
    }

    // ---- start strategies ----

    #[test]
    fn vad_start_triggers_on_vad_started() {
        let mut s = VadTurnStart;
        assert_eq!(
            s.on_frame(&Frame::Text("noise".into())),
            TurnStartDecision::Continue
        );
        assert_eq!(
            s.on_frame(&Frame::VadUserStartedSpeaking { start_secs: 0.2 }),
            TurnStartDecision::Triggered
        );
    }

    #[test]
    fn min_words_start_single_word_when_bot_silent() {
        let mut s = MinWordsTurnStart::new(3, true);
        // Bot silent: 1 word is enough.
        assert_eq!(
            s.on_frame(&transcription("hello", true)),
            TurnStartDecision::Triggered
        );
    }

    #[test]
    fn min_words_start_needs_min_while_bot_speaking() {
        let mut s = MinWordsTurnStart::new(3, true);
        s.on_frame(&Frame::BotStartedSpeaking);
        // One word while bot speaks ⇒ reset (sub-threshold).
        assert_eq!(
            s.on_frame(&transcription("hi", false)),
            TurnStartDecision::ResetAggregation
        );
        // Three words ⇒ trigger (barge-in).
        assert_eq!(
            s.on_frame(&transcription("hold on please", false)),
            TurnStartDecision::Triggered
        );
        // After bot stops, a single word triggers again.
        s.on_frame(&Frame::BotStoppedSpeaking);
        assert_eq!(
            s.on_frame(&transcription("yes", false)),
            TurnStartDecision::Triggered
        );
    }

    #[test]
    fn min_words_start_ignores_interim_when_disabled() {
        let mut s = MinWordsTurnStart::new(2, false);
        assert_eq!(
            s.on_frame(&interim("one two three")),
            TurnStartDecision::Continue
        );
    }

    #[test]
    fn external_start_only_triggers_once_armed() {
        let mut s = ExternalTurnStart::default();
        assert_eq!(
            s.on_frame(&Frame::VadUserStartedSpeaking { start_secs: 0.2 }),
            TurnStartDecision::Continue,
            "VAD must not trigger an external strategy"
        );
        s.trigger();
        assert_eq!(
            s.on_frame(&Frame::Text("x".into())),
            TurnStartDecision::Triggered
        );
        // Fires once.
        assert_eq!(
            s.on_frame(&Frame::Text("x".into())),
            TurnStartDecision::Continue
        );
    }

    // ---- stop strategies ----

    #[test]
    fn turn_analyzer_stop_finalized_transcript_triggers_immediately() {
        let mut s = TurnAnalyzerStop::new(1000.0);
        assert_eq!(
            s.on_frame(&Frame::VadUserStoppedSpeaking { stop_secs: 0.2 }),
            TurnStopDecision::Waiting
        );
        // Turn analyzer says complete, but no text yet.
        assert_eq!(s.set_turn_complete(true), TurnStopDecision::Waiting);
        // Finalized transcript ⇒ trigger.
        assert_eq!(
            s.on_frame(&transcription("done", true)),
            TurnStopDecision::Triggered
        );
    }

    #[test]
    fn turn_analyzer_stop_waits_for_timeout_without_finalize() {
        let mut s = TurnAnalyzerStop::new(800.0);
        s.on_frame(&Frame::VadUserStoppedSpeaking { stop_secs: 0.2 });
        s.set_turn_complete(true);
        // Non-final transcript: still waiting.
        assert_eq!(
            s.on_frame(&transcription("partial", false)),
            TurnStopDecision::Waiting
        );
        // Below timeout: still waiting.
        assert_eq!(s.on_silence_tick(500.0), TurnStopDecision::Waiting);
        // Past timeout: trigger.
        assert_eq!(s.on_silence_tick(800.0), TurnStopDecision::Triggered);
    }

    #[test]
    fn turn_analyzer_stop_blocks_while_user_speaking() {
        let mut s = TurnAnalyzerStop::new(0.0);
        s.on_frame(&Frame::VadUserStartedSpeaking { start_secs: 0.2 });
        s.set_turn_complete(true);
        // User is speaking again — never trigger even with text + complete + timeout.
        assert_eq!(
            s.on_frame(&transcription("more", true)),
            TurnStopDecision::Waiting
        );
    }

    #[test]
    fn speech_timeout_stop_needs_both_waits_and_text() {
        let mut s = SpeechTimeoutStop::new(600.0, 1200.0);
        s.on_frame(&Frame::VadUserStoppedSpeaking { stop_secs: 0.2 });
        s.on_frame(&transcription("ok", false));
        // Only the user-speech wait elapsed.
        assert_eq!(s.on_silence_tick(600.0), TurnStopDecision::Waiting);
        // Both waits elapsed ⇒ trigger.
        assert_eq!(s.on_silence_tick(1200.0), TurnStopDecision::Triggered);
    }

    #[test]
    fn speech_timeout_stop_finalize_shortcircuits_stt_wait() {
        let mut s = SpeechTimeoutStop::new(600.0, 5000.0);
        s.on_frame(&Frame::VadUserStoppedSpeaking { stop_secs: 0.2 });
        // Finalized transcript marks stt_done immediately.
        s.on_frame(&transcription("done", true));
        // Only the user-speech floor remains.
        assert_eq!(s.on_silence_tick(600.0), TurnStopDecision::Triggered);
    }

    #[test]
    fn speech_timeout_stop_no_text_never_triggers() {
        let mut s = SpeechTimeoutStop::new(0.0, 0.0);
        s.on_frame(&Frame::VadUserStoppedSpeaking { stop_secs: 0.5 });
        // Both timeouts are zero, but with no transcript it never fires.
        assert_eq!(s.on_silence_tick(1000.0), TurnStopDecision::Waiting);
    }

    // ---- mute strategies ----

    #[test]
    fn always_mute_is_always_muted() {
        let mut s = AlwaysMute;
        assert!(s.on_frame(&Frame::BotStoppedSpeaking));
        assert!(s.muted());
    }

    #[test]
    fn mute_until_first_bot_complete() {
        let mut s = MuteUntilFirstBotComplete::default();
        assert!(s.muted(), "muted from the very start");
        assert!(
            s.on_frame(&Frame::BotStartedSpeaking),
            "still muted while speaking"
        );
        assert!(
            !s.on_frame(&Frame::BotStoppedSpeaking),
            "unmuted after first complete"
        );
        // Subsequent bot turns no longer mute.
        assert!(!s.on_frame(&Frame::BotStartedSpeaking));
    }

    #[test]
    fn first_speech_mute_only_during_first_bot_turn() {
        let mut s = FirstSpeechMute::default();
        assert!(!s.muted(), "not muted before the bot speaks");
        assert!(
            s.on_frame(&Frame::BotStartedSpeaking),
            "muted during first speech"
        );
        assert!(!s.on_frame(&Frame::BotStoppedSpeaking), "unmuted after");
        // A later bot turn no longer mutes.
        assert!(!s.on_frame(&Frame::BotStartedSpeaking));
    }

    #[test]
    fn function_call_mute_tracks_in_flight_calls() {
        let mut s = FunctionCallMute::default();
        let call = FunctionCall {
            function_name: "lookup".into(),
            tool_call_id: "c1".into(),
            arguments: serde_json::json!({}),
        };
        assert!(s.on_frame(&Frame::FunctionCallsStarted(vec![call])));
        assert!(s.muted());
        // Resolving the call unmutes.
        assert!(!s.on_frame(&Frame::FunctionCallResult(FunctionCallResult {
            function_name: "lookup".into(),
            tool_call_id: "c1".into(),
            result: serde_json::json!({"ok": true}),
        })));
    }

    #[test]
    fn function_call_mute_cancel_also_unmutes() {
        let mut s = FunctionCallMute::default();
        let call = FunctionCall {
            function_name: "lookup".into(),
            tool_call_id: "c2".into(),
            arguments: serde_json::json!({}),
        };
        s.on_frame(&Frame::FunctionCallsStarted(vec![call]));
        assert!(!s.on_frame(&Frame::FunctionCallCancel {
            function_name: "lookup".into(),
            tool_call_id: "c2".into(),
        }));
    }

    // ---- interruption strategies ----

    #[test]
    fn always_interrupt_allows_barge_in() {
        let s = AlwaysInterrupt;
        assert!(s.should_interrupt());
    }

    #[test]
    fn min_words_interrupt_gates_on_word_count() {
        let mut s = MinWordsInterrupt::new(3);
        assert!(!s.should_interrupt());
        s.append_transcription(&interim("hold"));
        assert!(!s.should_interrupt(), "1 word < 3");
        s.append_transcription(&transcription("on please", false));
        assert!(s.should_interrupt(), "3 words >= 3");
        s.reset();
        assert!(!s.should_interrupt());
    }
}
