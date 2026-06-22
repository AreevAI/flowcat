// SPDX-License-Identifier: Apache-2.0
//
//! ContextRelay — reusable context compaction for realtime (S2S) sessions.
//!
//! Real-time speech-to-speech models attend the **entire conversation on every
//! turn**, and audio tokens cost far more than text (≈25 tok/s of audio vs ≈3 tok/s
//! of the same speech as text). On a long call the audio context is re-attended —
//! and re-billed — every turn, so cost grows roughly with the square of the turn
//! count, and providers cap session length outright.
//!
//! [`ContextRelayProcessor`] keeps a compact **text digest** of the conversation
//! (a rolling summary of older turns + the most recent turns kept verbatim) and,
//! when a configurable budget is crossed, re-bases the live session onto that digest
//! by emitting a [`Reprompt`](super::s2s) — the same frame a graph transition uses,
//! which the realtime service turns into `update_system`. On a stateful provider
//! (e.g. a session that reopens on `update_system`) this drops the expensive audio
//! history and seeds the fresh session with the cheap text digest. The mechanism is
//! provider-agnostic: it rides only [`Frame::Transcription`], the `UsageReport` /
//! [`Reprompt`](super::s2s) custom frames, and the existing `update_system` seam,
//! all of which every realtime backend supports.
//!
//! ## Off by default
//!
//! The processor is inserted into the S2S chain only when a [`ContextRelayConfig`]
//! is supplied to the builder. With no config the pipeline is unchanged.
//!
//! ## When it fires
//!
//! Compaction is evaluated at the **end of a bot turn** — the `UsageReport` boundary
//! (`response.done`), which also carries the provider's per-turn `input_tokens`
//! (the live context size). That moment is the safe idle gap: the model has finished
//! responding and no user utterance is pending, so re-basing the session does not
//! discard an in-flight turn. (Firing at the *next user* turn instead would tear the
//! session down right as the user finished speaking, before the model could answer —
//! losing that turn.)
//!
//! Three triggers are evaluated at that boundary: the per-turn token **budget**
//! (`max_context_tokens`), the **session age** (`max_session_secs` — re-base ahead of
//! a provider's hard session cap, e.g. Gemini Live's ~15-min audio limit), and a
//! **turn-count** fallback. The session-age trigger bypasses the anti-thrash floor so
//! a hard cap is never missed. This composes with — and does not replace — a
//! provider's *native* session resumption for transient drops, which the realtime
//! client handles orthogonally to compaction.
//!
//! The digest also enriches **every** transition reprompt, so a graph transition that
//! re-establishes the session carries the conversation forward as text instead of
//! starting blank.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use tokio::time::Instant;

use crate::error::Result;
use crate::processor::frame::{Frame, StartParams};
use crate::processor::{Envelope, FrameProcessor, Link, ProcessorSetup};
use crate::transcript::{Speaker, Transcript, TranscriptLine};
use crate::types::ToolDecl;

use super::s2s::{Reprompt, UsageReport};

/// Compresses older conversation turns into a summary, off the hot path. The actual
/// summarization model lives behind this trait so `flowcat-core` pulls no provider
/// dependency; an embedder (or a `flowcat-services` provider) supplies the impl.
///
/// Mirrors the cascaded path's `ContextSummarizer`
/// ([`super::ContextSummarizer`]) but with the digest model ContextRelay needs:
/// a rolling summary that folds a `prior` summary with the newly-aged `older` turns,
/// keeping the most recent turns verbatim (rather than collapsing all history).
#[async_trait]
pub trait ContextCompactor: Send + Sync {
    /// Summarize `older` turns, folding any `prior` summary, into a single
    /// replacement summary string. `None` leaves the digest unchanged (the digest
    /// then keeps carrying the turns verbatim until a later compaction succeeds).
    async fn compact(&self, older: &[TranscriptLine], prior: Option<&str>) -> Option<String>;
}

/// A no-op [`ContextCompactor`] that never summarizes, so the digest carries the
/// recent turns **verbatim**. Useful when no summarizer LLM is wired — the audio→text
/// re-base still drops the expensive audio history — or to exercise the relay live
/// without a second provider. Caveat: with no summary the verbatim tail grows over a
/// very long call; wire a real summarizing [`ContextCompactor`] for unbounded-length
/// calls.
pub struct VerbatimCompactor;

#[async_trait]
impl ContextCompactor for VerbatimCompactor {
    async fn compact(&self, _older: &[TranscriptLine], _prior: Option<&str>) -> Option<String> {
        None
    }
}

/// The running compacted representation of a conversation: a rolling `summary` of
/// the turns up to [`summarized_through`](Self::summarized_through), with everything
/// after it carried verbatim. Rendered into a delimited block appended to the base
/// system prompt when the session is re-based.
#[derive(Debug, Default, Clone)]
pub struct ContextDigest {
    /// Rolling summary of the earlier turns (`None` until the first compaction
    /// completes). Cumulative — each compaction folds the prior summary in.
    summary: Option<String>,
    /// Number of leading turns the [`summary`](Self::summary) already accounts for;
    /// turns at and after this index are rendered verbatim.
    summarized_through: usize,
}

impl ContextDigest {
    /// Render the digest as a delimited prompt block to append to a base system
    /// prompt, given the full observed `turns`. Returns `""` when there is nothing
    /// to carry (no summary yet and no turns past the summarized prefix).
    pub fn render_block(&self, turns: &[TranscriptLine]) -> String {
        let recent = &turns[self.summarized_through.min(turns.len())..];
        if self.summary.is_none() && recent.is_empty() {
            return String::new();
        }
        let mut out =
            String::from("\n\n--- Conversation so far (preserved across a session refresh) ---\n");
        if let Some(s) = &self.summary {
            out.push_str("Summary of earlier conversation: ");
            out.push_str(s);
            out.push('\n');
        }
        for line in recent {
            out.push_str(match line.speaker {
                Speaker::User => "user: ",
                Speaker::Bot => "assistant: ",
            });
            out.push_str(&line.text);
            out.push('\n');
        }
        out.push_str("--- end of preserved conversation ---\n");
        out
    }
}

/// Policy + engine for [`ContextRelayProcessor`]: when to compact and how much to
/// keep verbatim. The triggers are evaluated at each bot-turn boundary; the relay
/// re-bases the session when either the token budget or the turn-count fallback is
/// crossed (subject to [`min_turns_between`](Self::min_turns_between)).
#[derive(Clone)]
pub struct ContextRelayConfig {
    /// Re-base when the provider's per-turn `input_tokens` exceeds this. `None`
    /// disables the token trigger (e.g. a provider that reports no usage).
    pub max_context_tokens: Option<u64>,
    /// Fallback trigger: re-base after this many bot turns since the last compaction.
    /// `None` disables it. Useful when usage is reported without `input_tokens`.
    pub trigger_after_turns: Option<usize>,
    /// Keep this many most-recent turns verbatim; older turns are summarized.
    pub keep_recent_turns: usize,
    /// Minimum bot turns between two compactions — an anti-thrash floor that also
    /// bounds how often a stateful provider reopens its session. The session-age
    /// trigger bypasses it (a hard cap must not be missed); budget + turn triggers
    /// respect it.
    pub min_turns_between: usize,
    /// Re-base when the live session has been open at least this many seconds, ahead
    /// of the provider's **hard session cap** (e.g. Gemini Live's ~15-min audio
    /// limit, which forcibly drops the call regardless of token count). Evaluated at
    /// bot-turn boundaries; the clock resets on each re-base (the reopen starts a
    /// fresh session). `None` disables the time trigger.
    pub max_session_secs: Option<u64>,
    /// The summarization engine (off the hot path).
    pub compactor: Arc<dyn ContextCompactor>,
}

impl ContextRelayConfig {
    /// A config with sensible defaults over `compactor`: a 4k-token budget, keep the
    /// last 4 turns verbatim, and at least 2 turns between compactions. Tune the
    /// public fields as needed.
    pub fn new(compactor: Arc<dyn ContextCompactor>) -> Self {
        Self {
            max_context_tokens: Some(4_000),
            trigger_after_turns: None,
            keep_recent_turns: 4,
            min_turns_between: 2,
            max_session_secs: None,
            compactor,
        }
    }
}

/// Why a [`ContextRelayProcessor`] re-based the session — surfaced in tracing so an
/// operator can see whether compaction was driven by the token budget, the session
/// age (a hard provider cap), or the turn-count fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompactionReason {
    /// The provider's per-turn `input_tokens` crossed `max_context_tokens`.
    Budget,
    /// The live session has been open longer than `max_session_secs` — re-base ahead
    /// of the provider's hard session cap.
    SessionAge,
    /// `trigger_after_turns` bot turns elapsed since the last compaction.
    TurnCount,
}

/// The relay processor (PROCESSOR-DESIGN §10 extension processor). Inserted between
/// the realtime service and the brain so it observes the model's transcripts +
/// usage flowing downstream and the brain's reprompts flowing upstream.
///
/// It taps every final [`Frame::Transcription`] into its own running [`Transcript`]
/// (deterministic and self-contained — it does not depend on the downstream
/// transcript tap), reads `input_tokens` off each `UsageReport`, enriches transition
/// reprompts with the digest, and on a budget/turn trigger emits a compaction
/// reprompt that re-bases the session onto the digest.
pub struct ContextRelayProcessor {
    config: ContextRelayConfig,
    /// The current digest-free node prompt (seeded from the connect setup, updated by
    /// each brain transition reprompt). Kept separate from the emitted "effective"
    /// prompt (`base + digest`) so digests never accumulate onto each other.
    base_prompt: String,
    /// The current node tool set (seeded at construction, updated on transitions).
    base_tools: Vec<ToolDecl>,
    /// Our own turn-by-turn accumulation, built from observed final transcriptions.
    transcript: Transcript,
    /// The rolling digest, shared with the off-hot-path compaction task.
    digest: Arc<Mutex<ContextDigest>>,
    /// One compaction cycle (reopen + summarize) in flight at a time.
    in_flight: Arc<AtomicBool>,
    /// Bot turns observed since the last compaction (the trigger clock).
    turns_since_compaction: usize,
    /// When the current underlying realtime session opened — the pipeline start, then
    /// reset on each re-base (an `update_system` reopen starts a fresh session). Drives
    /// the session-age trigger. `None` until [`start`](FrameProcessor::start) runs.
    session_started: Option<Instant>,
}

impl ContextRelayProcessor {
    /// Build a relay seeded with the call's opening `base_prompt` + `base_tools`
    /// (the same values the assembler puts in the initial realtime setup, since the
    /// first prompt never arrives as a transition reprompt).
    pub fn new(config: ContextRelayConfig, base_prompt: String, base_tools: Vec<ToolDecl>) -> Self {
        Self {
            config,
            base_prompt,
            base_tools,
            transcript: Transcript::new(),
            digest: Arc::new(Mutex::new(ContextDigest::default())),
            in_flight: Arc::new(AtomicBool::new(false)),
            turns_since_compaction: 0,
            session_started: None,
        }
    }

    /// The prompt to send on a re-base: the digest-free base + the current digest
    /// block. Idempotent in the base (calling it twice never doubles the base).
    fn effective_prompt(&self) -> String {
        let block = self
            .digest
            .lock()
            .unwrap()
            .render_block(&self.transcript.lines);
        format!("{}{}", self.base_prompt, block)
    }

    /// The trigger (if any) that fires a re-base at this bot-turn boundary, given the
    /// provider's per-turn `input_tokens` and the live session's `elapsed_secs`.
    fn should_compact(
        &self,
        input_tokens: Option<u64>,
        elapsed_secs: u64,
    ) -> Option<CompactionReason> {
        if self.in_flight.load(Ordering::SeqCst) || self.transcript.lines.is_empty() {
            return None;
        }
        // Session-age (hard-cap) trigger bypasses the anti-thrash floor — missing it
        // risks the provider forcibly dropping the call.
        if matches!(self.config.max_session_secs, Some(max) if elapsed_secs >= max) {
            return Some(CompactionReason::SessionAge);
        }
        // Budget + turn-count triggers respect the min-turns floor.
        if self.turns_since_compaction < self.config.min_turns_between.max(1) {
            return None;
        }
        if matches!(
            (self.config.max_context_tokens, input_tokens),
            (Some(max), Some(t)) if t > max
        ) {
            return Some(CompactionReason::Budget);
        }
        if matches!(
            self.config.trigger_after_turns,
            Some(n) if self.turns_since_compaction >= n
        ) {
            return Some(CompactionReason::TurnCount);
        }
        None
    }

    /// Re-base the session onto the digest (the reopen) and kick off the off-hot-path
    /// summary that folds the newly-aged turns into the rolling summary.
    async fn fire_compaction(&mut self, link: &Link, reason: CompactionReason) {
        self.in_flight.store(true, Ordering::SeqCst);
        self.turns_since_compaction = 0;
        // The reopen starts a fresh underlying session — restart the age clock.
        self.session_started = Some(Instant::now());
        tracing::info!(
            ?reason,
            turns = self.transcript.lines.len(),
            "context-relay: re-basing realtime session onto text digest"
        );

        // Re-base now, using the digest as it stands (summary from prior compactions
        // + the turns observed since, verbatim). A no-op ToolResult is NOT sent: the
        // model isn't awaiting a tool call here, only a fresh system context.
        let prompt = self.effective_prompt();
        let tools = self.base_tools.clone();
        link.push_up(Frame::Custom(Arc::new(Reprompt { prompt, tools })))
            .await;

        // Fold the turns now older than `keep_recent_turns` into the summary so the
        // NEXT re-base carries fewer verbatim turns. Runs detached; releases the
        // in-flight guard when done.
        let lines = self.transcript.lines.clone();
        let through = self.digest.lock().unwrap().summarized_through;
        let target = lines.len().saturating_sub(self.config.keep_recent_turns);
        if target <= through {
            self.in_flight.store(false, Ordering::SeqCst);
            return;
        }
        let older = lines[..target].to_vec();
        let prior = self.digest.lock().unwrap().summary.clone();
        let compactor = self.config.compactor.clone();
        let digest = self.digest.clone();
        let in_flight = self.in_flight.clone();
        tokio::spawn(async move {
            let summary = compactor.compact(&older, prior.as_deref()).await;
            if let Some(s) = summary {
                let mut d = digest.lock().unwrap();
                d.summary = Some(s);
                d.summarized_through = target;
            }
            in_flight.store(false, Ordering::SeqCst);
        });
    }
}

#[async_trait]
impl FrameProcessor for ContextRelayProcessor {
    fn name(&self) -> &str {
        "ContextRelay"
    }

    /// Capture the session start so the session-age trigger can measure elapsed time.
    async fn start(&mut self, _setup: &ProcessorSetup, _params: &StartParams) -> Result<()> {
        self.session_started = Some(Instant::now());
        Ok(())
    }

    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        match &env.frame {
            // Tap final transcriptions into our own turn log (then forward, so the
            // downstream transcript/recorder still see them).
            Frame::Transcription {
                text,
                user_id,
                final_: true,
                ..
            } => {
                if user_id.as_ref() == "bot" {
                    self.transcript.push_bot(text);
                } else {
                    self.transcript.push_user(text);
                }
                link.push(env.meta, env.frame, env.direction).await;
            }

            // Bot-turn boundary + the live context-size signal: evaluate the trigger.
            Frame::Custom(c) if c.as_any().is::<UsageReport>() => {
                let input_tokens = c
                    .as_any()
                    .downcast_ref::<UsageReport>()
                    .unwrap()
                    .0
                    .input_tokens;
                self.turns_since_compaction += 1;
                let elapsed_secs = self
                    .session_started
                    .map(|t| t.elapsed().as_secs())
                    .unwrap_or(0);
                if let Some(reason) = self.should_compact(input_tokens, elapsed_secs) {
                    self.fire_compaction(link, reason).await;
                }
                // Forward downstream so the recorder still folds usage.
                link.push(env.meta, env.frame, env.direction).await;
            }

            // A brain transition reprompt: adopt it as the new digest-free base, then
            // forward an enriched copy so the transition carries the conversation
            // forward as text. The original (un-enriched) reprompt is consumed.
            Frame::Custom(c) if c.as_any().is::<Reprompt>() => {
                let rp = c.as_any().downcast_ref::<Reprompt>().unwrap();
                self.base_prompt = rp.prompt.clone();
                self.base_tools = rp.tools.clone();
                let prompt = self.effective_prompt();
                let tools = self.base_tools.clone();
                link.push_up(Frame::Custom(Arc::new(Reprompt { prompt, tools })))
                    .await;
            }

            _ => link.push(env.meta, env.frame, env.direction).await,
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A compactor that returns a fixed summary (the no-network test engine).
    struct FixedCompactor(&'static str);
    #[async_trait]
    impl ContextCompactor for FixedCompactor {
        async fn compact(&self, _older: &[TranscriptLine], _prior: Option<&str>) -> Option<String> {
            Some(self.0.to_string())
        }
    }

    fn lines() -> Vec<TranscriptLine> {
        vec![
            TranscriptLine {
                speaker: Speaker::Bot,
                text: "Hello, how can I help?".into(),
            },
            TranscriptLine {
                speaker: Speaker::User,
                text: "I have a billing question".into(),
            },
            TranscriptLine {
                speaker: Speaker::Bot,
                text: "Sure — what's the issue?".into(),
            },
        ]
    }

    #[test]
    fn empty_digest_renders_empty_block() {
        let d = ContextDigest::default();
        assert_eq!(d.render_block(&[]), "");
    }

    #[test]
    fn verbatim_only_block_when_no_summary_yet() {
        // No summary, summarized_through = 0 → every turn rendered verbatim.
        let d = ContextDigest::default();
        let block = d.render_block(&lines());
        assert!(block.contains("--- Conversation so far"));
        assert!(!block.contains("Summary of earlier conversation"));
        assert!(block.contains("assistant: Hello, how can I help?"));
        assert!(block.contains("user: I have a billing question"));
    }

    #[test]
    fn summary_plus_recent_verbatim() {
        // Summary covers the first turn; turns 2..n render verbatim after it.
        let d = ContextDigest {
            summary: Some("caller asked about billing".into()),
            summarized_through: 1,
        };
        let block = d.render_block(&lines());
        assert!(block.contains("Summary of earlier conversation: caller asked about billing"));
        // The summarized leading turn is not repeated verbatim...
        assert!(!block.contains("assistant: Hello, how can I help?"));
        // ...but the turns after the summarized prefix are kept verbatim.
        assert!(block.contains("user: I have a billing question"));
        assert!(block.contains("assistant: Sure — what's the issue?"));
    }

    #[test]
    fn effective_prompt_does_not_accumulate_the_digest() {
        let mut p = ContextRelayProcessor::new(
            ContextRelayConfig::new(Arc::new(FixedCompactor("s"))),
            "BASE PROMPT".into(),
            vec![],
        );
        for l in lines() {
            match l.speaker {
                Speaker::Bot => p.transcript.push_bot(&l.text),
                Speaker::User => p.transcript.push_user(&l.text),
            }
        }
        let once = p.effective_prompt();
        let twice = p.effective_prompt();
        assert_eq!(once, twice, "render is idempotent");
        // Base appears exactly once (the digest block is appended, not folded in).
        assert_eq!(once.matches("BASE PROMPT").count(), 1);
        assert!(once.starts_with("BASE PROMPT"));
        assert!(once.contains("--- Conversation so far"));
    }

    #[test]
    fn trigger_respects_budget_min_turns_and_empty_transcript() {
        let mut cfg = ContextRelayConfig::new(Arc::new(FixedCompactor("s")));
        cfg.max_context_tokens = Some(1_000);
        cfg.min_turns_between = 2;
        let mut p = ContextRelayProcessor::new(cfg, "BASE".into(), vec![]);

        // Empty transcript → never compact even over budget.
        p.turns_since_compaction = 5;
        assert_eq!(p.should_compact(Some(9_999), 0), None);

        // With turns but under the min-turns floor → no compaction.
        p.transcript.push_user("hi");
        p.turns_since_compaction = 1;
        assert_eq!(p.should_compact(Some(9_999), 0), None);

        // Floor met + over budget → compact (budget).
        p.turns_since_compaction = 2;
        assert_eq!(
            p.should_compact(Some(9_999), 0),
            Some(CompactionReason::Budget)
        );

        // Floor met + under budget → no compaction.
        assert_eq!(p.should_compact(Some(10), 0), None);
    }

    #[test]
    fn turn_count_fallback_trigger() {
        let mut cfg = ContextRelayConfig::new(Arc::new(FixedCompactor("s")));
        cfg.max_context_tokens = None; // budget disabled
        cfg.trigger_after_turns = Some(3);
        cfg.min_turns_between = 1;
        let mut p = ContextRelayProcessor::new(cfg, "BASE".into(), vec![]);
        p.transcript.push_user("hi");

        p.turns_since_compaction = 2;
        assert_eq!(p.should_compact(None, 0), None, "below the turn threshold");
        p.turns_since_compaction = 3;
        assert_eq!(
            p.should_compact(None, 0),
            Some(CompactionReason::TurnCount),
            "at the turn threshold, no usage needed"
        );
    }

    #[test]
    fn session_age_trigger_fires_and_bypasses_min_turns() {
        let mut cfg = ContextRelayConfig::new(Arc::new(FixedCompactor("s")));
        cfg.max_context_tokens = None; // budget + turn triggers disabled
        cfg.trigger_after_turns = None;
        cfg.min_turns_between = 10; // a high floor the budget/turn paths can't clear
        cfg.max_session_secs = Some(600);
        let mut p = ContextRelayProcessor::new(cfg, "BASE".into(), vec![]);
        p.transcript.push_user("hi");
        p.turns_since_compaction = 1;

        // Under the age limit → no compaction.
        assert_eq!(p.should_compact(None, 599), None);

        // Past the age limit → compact even though the min-turns floor isn't met
        // (a hard provider cap must not be missed).
        assert_eq!(
            p.should_compact(None, 600),
            Some(CompactionReason::SessionAge)
        );
    }
}
