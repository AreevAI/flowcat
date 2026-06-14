// SPDX-License-Identifier: Apache-2.0
//
//! Text aggregators / output text filters (pure logic, no network).
//!
//! Ports pipecat's text-aggregator + `BaseTextFilter` set
//! (`pipecat/src/pipecat/utils/text/`) to deterministic Rust transforms over
//! the `LlmText`/`Text` frame stream before TTS:
//!
//! - [`SimpleTextAggregator`] — accumulate streamed text and emit it a
//!   *sentence* at a time (sentence-boundary detection with lookahead).
//! - [`SkipTagsAggregator`] — like the simple aggregator, but never split a
//!   sentence inside a configured start/end tag pair.
//! - [`PatternPairAggregator`] — strip/keep/aggregate content between a pair of
//!   delimiters (e.g. `<dtmf>…</dtmf>`), the building block the IVR navigator
//!   reuses to pull commands out of model text.
//! - [`MarkdownTextFilter`] — strip Markdown formatting to plain speakable text.
//!
//! Each aggregator is a plain value type (`aggregate` + `flush`) so it is trivially
//! unit-testable; [`TextAggregatorProcessor`] wraps any [`TextAggregator`] into a
//! [`FrameProcessor`] that re-emits whole sentences as `Text` frames, and
//! [`TextFilterProcessor`] wraps a [`TextFilter`] to clean each `LlmText`/`Text`
//! frame in place.

use async_trait::async_trait;

use crate::error::Result;
use crate::processor::frame::Frame;
use crate::processor::{Envelope, FrameProcessor, Link};

// ===========================================================================
// Aggregators (sentence / tag / pattern) — pure value types.
// ===========================================================================

/// Sentence-ending punctuation (mirrors pipecat `SENTENCE_ENDING_PUNCTUATION`).
const SENTENCE_ENDING_PUNCTUATION: &[char] = &['.', '?', '!', ';', '。', '？', '！', '；'];

/// A streamed-text aggregator: feed chunks, get back completed aggregations
/// (usually whole sentences), and `flush` the tail at end-of-stream. Mirrors
/// pipecat `BaseTextAggregator`.
pub trait TextAggregator: Send {
    /// Feed `text`; return any aggregations completed by it (in order).
    fn aggregate(&mut self, text: &str) -> Vec<String>;
    /// Flush whatever remains buffered (called at end-of-response). `None` if empty.
    fn flush(&mut self) -> Option<String>;
    /// Discard the buffer (on interruption / reset).
    fn reset(&mut self);
}

/// Accumulate text until an end-of-sentence marker, then release the sentence.
/// Port of pipecat `SimpleTextAggregator` with the same "wait for non-whitespace
/// lookahead after terminal punctuation" disambiguation (so `"$29."` does not
/// split until the next word arrives).
#[derive(Debug, Default, Clone)]
pub struct SimpleTextAggregator {
    text: String,
    needs_lookahead: bool,
}

impl SimpleTextAggregator {
    /// A fresh, empty aggregator.
    pub fn new() -> Self {
        Self::default()
    }

    /// The currently-buffered (un-emitted) text, trimmed of edge spaces.
    pub fn buffered(&self) -> &str {
        &self.text
    }

    /// Feed one character, returning a completed sentence if `char` closed one.
    /// Shared lookahead logic reused by [`SkipTagsAggregator`] /
    /// [`PatternPairAggregator`] (pipecat `_check_sentence_with_lookahead`).
    fn check_sentence_with_lookahead(&mut self, ch: char) -> Option<String> {
        if self.needs_lookahead {
            // Already saw terminal punctuation; wait for the first non-whitespace
            // char to confirm the boundary.
            if !ch.is_whitespace() {
                self.needs_lookahead = false;
                if let Some(marker) = match_endofsentence(&self.text) {
                    let result: String = self.text[..marker].to_string();
                    self.text = self.text[marker..].to_string();
                    return Some(result.trim_matches(' ').to_string());
                }
            }
            return None;
        }
        // Did we just append terminal punctuation? If so, defer to lookahead.
        if self
            .text
            .chars()
            .last()
            .is_some_and(|c| SENTENCE_ENDING_PUNCTUATION.contains(&c))
        {
            self.needs_lookahead = true;
        }
        None
    }
}

impl TextAggregator for SimpleTextAggregator {
    fn aggregate(&mut self, text: &str) -> Vec<String> {
        let mut out = Vec::new();
        for ch in text.chars() {
            self.text.push(ch);
            if let Some(sentence) = self.check_sentence_with_lookahead(ch) {
                out.push(sentence);
            }
        }
        out
    }

    fn flush(&mut self) -> Option<String> {
        if self.text.is_empty() {
            return None;
        }
        let result = std::mem::take(&mut self.text);
        self.needs_lookahead = false;
        Some(result.trim_matches(' ').to_string())
    }

    fn reset(&mut self) {
        self.text.clear();
        self.needs_lookahead = false;
    }
}

/// Find the byte offset just past the first end-of-sentence marker in `text`,
/// or `None` if the buffer is not yet a complete sentence.
///
/// A lightweight, dependency-free stand-in for pipecat's NLTK
/// `match_endofsentence`: a sentence ends at the **last run** of terminal
/// punctuation (so `"Mr. Smith."` is one sentence, and `"3.14"` — a digit on
/// each side of the dot — is not treated as a boundary).
fn match_endofsentence(text: &str) -> Option<usize> {
    let chars: Vec<char> = text.chars().collect();
    let mut idx: Option<usize> = None;
    for (i, &c) in chars.iter().enumerate() {
        if SENTENCE_ENDING_PUNCTUATION.contains(&c) {
            // Skip a decimal point between digits ("3.14") — not a boundary.
            if c == '.'
                && i > 0
                && chars[i - 1].is_ascii_digit()
                && chars.get(i + 1).is_some_and(|n| n.is_ascii_digit())
            {
                continue;
            }
            idx = Some(i);
        }
    }
    // Convert the char index of the last terminal punctuation to a byte offset
    // *after* it (and any trailing run of punctuation).
    let last = idx?;
    let mut end = last;
    while end + 1 < chars.len() && SENTENCE_ENDING_PUNCTUATION.contains(&chars[end + 1]) {
        end += 1;
    }
    Some(chars[..=end].iter().map(|c| c.len_utf8()).sum())
}

/// A start/end delimiter pair (pipecat `StartEndTags`).
#[derive(Debug, Clone)]
pub struct StartEndTags {
    /// The opening delimiter (e.g. `"<think>"`).
    pub start: String,
    /// The closing delimiter (e.g. `"</think>"`).
    pub end: String,
}

impl StartEndTags {
    /// A new tag pair.
    pub fn new(start: impl Into<String>, end: impl Into<String>) -> Self {
        Self {
            start: start.into(),
            end: end.into(),
        }
    }
}

/// Like [`SimpleTextAggregator`] but never splits a sentence *inside* a configured
/// tag pair — the tagged region is aggregated as one unit regardless of internal
/// punctuation. Port of pipecat `SkipTagsAggregator`.
#[derive(Debug, Clone)]
pub struct SkipTagsAggregator {
    inner: SimpleTextAggregator,
    tags: Vec<StartEndTags>,
    inside: bool,
}

impl SkipTagsAggregator {
    /// A new aggregator that suspends sentence detection between any of `tags`.
    pub fn new(tags: Vec<StartEndTags>) -> Self {
        Self {
            inner: SimpleTextAggregator::new(),
            tags,
            inside: false,
        }
    }

    /// Recompute tag state from the current buffer (the last unmatched start tag
    /// without its end keeps us "inside").
    fn update_tag_state(&mut self) {
        for tag in &self.tags {
            let starts = self.inner.text.matches(&tag.start).count();
            let ends = self.inner.text.matches(&tag.end).count();
            if starts > ends {
                self.inside = true;
                return;
            }
        }
        self.inside = false;
    }
}

impl TextAggregator for SkipTagsAggregator {
    fn aggregate(&mut self, text: &str) -> Vec<String> {
        let mut out = Vec::new();
        for ch in text.chars() {
            self.inner.text.push(ch);
            self.update_tag_state();
            if self.inside {
                continue;
            }
            if let Some(sentence) = self.inner.check_sentence_with_lookahead(ch) {
                out.push(sentence);
            }
        }
        out
    }

    fn flush(&mut self) -> Option<String> {
        self.inside = false;
        self.inner.flush()
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.inside = false;
    }
}

/// What to do with the region between a matched pattern pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchAction {
    /// Drop the delimiters *and* the content; aggregation continues as if absent.
    Remove,
    /// Drop the delimiters but keep the content in the sentence stream.
    Keep,
    /// Emit the content between the delimiters as its own aggregation.
    Aggregate,
}

/// A registered delimiter pair.
#[derive(Debug, Clone)]
struct Pattern {
    id: String,
    start: String,
    end: String,
    action: MatchAction,
}

/// A completed pattern match: the `id` of the pattern and its inner `content`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatternMatch {
    /// The registered pattern id (e.g. `"dtmf"`), or `"sentence"` for plain text.
    pub id: String,
    /// The text between the delimiters (trimmed), or the sentence text.
    pub content: String,
}

/// Identify and process content between pattern pairs in streaming text. Port of
/// pipecat `PatternPairAggregator` (the IVR navigator's command extractor): plain
/// text is still emitted sentence-by-sentence, while `Aggregate` patterns emit
/// their inner content as a tagged [`PatternMatch`].
#[derive(Debug, Clone, Default)]
pub struct PatternPairAggregator {
    inner: SimpleTextAggregator,
    patterns: Vec<Pattern>,
}

impl PatternPairAggregator {
    /// A new aggregator with no patterns registered.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a delimiter pair. Returns `self` for chaining.
    pub fn add_pattern(
        mut self,
        id: impl Into<String>,
        start: impl Into<String>,
        end: impl Into<String>,
        action: MatchAction,
    ) -> Self {
        self.patterns.push(Pattern {
            id: id.into(),
            start: start.into(),
            end: end.into(),
            action,
        });
        self
    }

    /// True while the buffer holds an unclosed start delimiter (so we must keep
    /// buffering rather than emit a partial sentence).
    fn incomplete_pattern(&self) -> bool {
        self.patterns.iter().any(|p| {
            self.inner.text.matches(&p.start).count() > self.inner.text.matches(&p.end).count()
        })
    }

    /// Try to resolve the first complete pattern in the buffer. Returns the match
    /// (when the action surfaces one) and rewrites the buffer per the action.
    fn process_complete_patterns(&mut self) -> Option<PatternMatch> {
        for p in self.patterns.clone() {
            let Some(s_idx) = self.inner.text.find(&p.start) else {
                continue;
            };
            let after_start = s_idx + p.start.len();
            let Some(rel_end) = self.inner.text[after_start..].find(&p.end) else {
                continue;
            };
            let e_idx = after_start + rel_end;
            let content = self.inner.text[after_start..e_idx].trim().to_string();
            let full_end = e_idx + p.end.len();
            match p.action {
                MatchAction::Remove => {
                    self.inner.text.replace_range(s_idx..full_end, "");
                    // Removed patterns yield no surfaced match; keep scanning.
                    return self.process_complete_patterns();
                }
                MatchAction::Keep => {
                    self.inner
                        .text
                        .replace_range(s_idx..full_end, &content.clone());
                    return None;
                }
                MatchAction::Aggregate => {
                    self.inner.text.replace_range(s_idx..full_end, "");
                    return Some(PatternMatch { id: p.id, content });
                }
            }
        }
        None
    }

    /// Feed `text`, returning ordered [`PatternMatch`]es (sentences carry the id
    /// `"sentence"`; aggregated patterns carry their registered id).
    pub fn aggregate_matches(&mut self, text: &str) -> Vec<PatternMatch> {
        let mut out = Vec::new();
        for ch in text.chars() {
            self.inner.text.push(ch);
            if let Some(m) = self.process_complete_patterns() {
                out.push(m);
                continue;
            }
            if self.incomplete_pattern() {
                continue;
            }
            if let Some(sentence) = self.inner.check_sentence_with_lookahead(ch) {
                out.push(PatternMatch {
                    id: "sentence".to_string(),
                    content: sentence,
                });
            }
        }
        out
    }
}

impl TextAggregator for PatternPairAggregator {
    fn aggregate(&mut self, text: &str) -> Vec<String> {
        self.aggregate_matches(text)
            .into_iter()
            .map(|m| m.content)
            .collect()
    }

    fn flush(&mut self) -> Option<String> {
        self.inner.flush()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

// ===========================================================================
// Filters (per-frame text transforms) — pure value types.
// ===========================================================================

/// A per-chunk text transform applied before TTS. Mirrors pipecat
/// `BaseTextFilter`.
pub trait TextFilter: Send {
    /// Transform `text` (e.g. strip Markdown). Returns the cleaned text.
    fn filter(&mut self, text: &str) -> String;
    /// Reset any cross-chunk state (on interruption).
    fn reset(&mut self) {}
}

/// Configuration for [`MarkdownTextFilter`].
#[derive(Debug, Clone)]
pub struct MarkdownFilterParams {
    /// Whether filtering is applied at all (pass-through when `false`).
    pub enabled: bool,
}

impl Default for MarkdownFilterParams {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// Strip common Markdown formatting so the text is cleanly speakable. A
/// dependency-free subset of pipecat `MarkdownTextFilter` covering the cases that
/// matter for TTS: emphasis markers (`**`/`*`/`_`), inline-code backticks, ATX
/// headings (`#`), list bullets, blockquote markers, link syntax
/// (`[text](url)` → `text`), and bare URL scheme stripping.
#[derive(Debug, Clone, Default)]
pub struct MarkdownTextFilter {
    params: MarkdownFilterParams,
}

impl MarkdownTextFilter {
    /// A new filter with the given params.
    pub fn new(params: MarkdownFilterParams) -> Self {
        Self { params }
    }
}

impl TextFilter for MarkdownTextFilter {
    fn filter(&mut self, text: &str) -> String {
        if !self.params.enabled {
            return text.to_string();
        }
        let mut out = String::with_capacity(text.len());
        for line in text.split_inclusive('\n') {
            // Split trailing newline back out so we can trim line-leading markup.
            let (body, nl) = match line.strip_suffix('\n') {
                Some(b) => (b, "\n"),
                None => (line, ""),
            };
            let trimmed = body.trim_start();
            let lead_ws = &body[..body.len() - trimmed.len()];
            // Strip ATX heading markers, blockquote `>` and list bullets.
            let mut content = trimmed;
            content = content.trim_start_matches('#').trim_start();
            content = content.trim_start_matches('>').trim_start();
            if let Some(rest) = content
                .strip_prefix("- ")
                .or_else(|| content.strip_prefix("* "))
                .or_else(|| content.strip_prefix("+ "))
            {
                content = rest;
            }
            out.push_str(lead_ws);
            out.push_str(&strip_inline_markdown(content));
            out.push_str(nl);
        }
        out
    }
}

/// Strip inline emphasis/code/link Markdown from a single line.
fn strip_inline_markdown(s: &str) -> String {
    // Collapse links `[text](url)` → `text`.
    let mut s = collapse_links(s);
    // Remove emphasis/code markers and bare URL schemes.
    s = s.replace("**", "");
    s = s.replace("__", "");
    s = s.replace('`', "");
    s = s.replace("https://", "").replace("http://", "");
    // Remove lone `*` / `_` used for emphasis (kept simple: drop them all).
    s.retain(|c| c != '*' && c != '_');
    s
}

/// Replace every `[text](url)` with `text`.
fn collapse_links(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'[' {
            if let Some(close) = s[i + 1..].find(']') {
                let text_end = i + 1 + close;
                let rest = &s[text_end + 1..];
                if rest.starts_with('(') {
                    if let Some(paren) = rest.find(')') {
                        out.push_str(&s[i + 1..text_end]);
                        i = text_end + 1 + paren + 1;
                        continue;
                    }
                }
            }
        }
        // Copy this UTF-8 char whole.
        let ch_len = s[i..].chars().next().map(|c| c.len_utf8()).unwrap_or(1);
        out.push_str(&s[i..i + ch_len]);
        i += ch_len;
    }
    out
}

// ===========================================================================
// Processors wrapping the value types into the frame pipeline.
// ===========================================================================

/// A [`FrameProcessor`] that aggregates streamed `LlmText`/`Text` frames into
/// whole-sentence `Text` frames using any [`TextAggregator`]. On `LlmResponseEnd`
/// it flushes the tail; on `Interruption` it resets the buffer. Mirrors where
/// pipecat places a text aggregator before TTS.
pub struct TextAggregatorProcessor<A: TextAggregator> {
    name: &'static str,
    agg: A,
}

impl<A: TextAggregator> TextAggregatorProcessor<A> {
    /// Wrap `agg` as a processor named `name`.
    pub fn new(name: &'static str, agg: A) -> Self {
        Self { name, agg }
    }
}

#[async_trait]
impl<A: TextAggregator + 'static> FrameProcessor for TextAggregatorProcessor<A> {
    fn name(&self) -> &str {
        self.name
    }

    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        match &env.frame {
            Frame::LlmText(t) | Frame::Text(t) => {
                let direction = env.direction;
                for sentence in self.agg.aggregate(t) {
                    if !sentence.is_empty() {
                        link.push(
                            crate::processor::frame::FrameMeta::new(&Frame::Text(sentence.clone())),
                            Frame::Text(sentence),
                            direction,
                        )
                        .await;
                    }
                }
            }
            Frame::LlmResponseEnd => {
                if let Some(tail) = self.agg.flush() {
                    if !tail.is_empty() {
                        link.push_down(Frame::Text(tail)).await;
                    }
                }
                link.push(env.meta, env.frame, env.direction).await;
            }
            Frame::Interruption => {
                self.agg.reset();
                link.push(env.meta, env.frame, env.direction).await;
            }
            _ => {
                link.push(env.meta, env.frame, env.direction).await;
            }
        }
        Ok(())
    }
}

/// A [`FrameProcessor`] that runs a [`TextFilter`] over every `LlmText`/`Text`
/// frame, replacing the frame text in place (dropping frames that filter to
/// empty). Resets the filter on `Interruption`.
pub struct TextFilterProcessor<F: TextFilter> {
    name: &'static str,
    filter: F,
}

impl<F: TextFilter> TextFilterProcessor<F> {
    /// Wrap `filter` as a processor named `name`.
    pub fn new(name: &'static str, filter: F) -> Self {
        Self { name, filter }
    }
}

#[async_trait]
impl<F: TextFilter + 'static> FrameProcessor for TextFilterProcessor<F> {
    fn name(&self) -> &str {
        self.name
    }

    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        match env.frame {
            Frame::LlmText(t) => {
                let cleaned = self.filter.filter(&t);
                if !cleaned.is_empty() {
                    link.push(env.meta, Frame::LlmText(cleaned), env.direction)
                        .await;
                }
            }
            Frame::Text(t) => {
                let cleaned = self.filter.filter(&t);
                if !cleaned.is_empty() {
                    link.push(env.meta, Frame::Text(cleaned), env.direction)
                        .await;
                }
            }
            Frame::Interruption => {
                self.filter.reset();
                link.push(env.meta, Frame::Interruption, env.direction)
                    .await;
            }
            other => {
                link.push(env.meta, other, env.direction).await;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::test_harness::drive;
    use crate::processor::frame::Direction;

    #[test]
    fn simple_aggregator_emits_one_sentence_at_a_time() {
        let mut agg = SimpleTextAggregator::new();
        // Punctuation alone does not emit until non-whitespace lookahead arrives.
        assert!(agg.aggregate("Hello there.").is_empty());
        let out = agg.aggregate(" How are you?");
        assert_eq!(out, vec!["Hello there."]);
        // The second sentence stays buffered until its lookahead.
        let out2 = agg.aggregate(" Fine");
        assert_eq!(out2, vec!["How are you?"]);
        assert_eq!(agg.flush().as_deref(), Some("Fine"));
    }

    #[test]
    fn simple_aggregator_does_not_split_decimals() {
        let mut agg = SimpleTextAggregator::new();
        // "3.14" must not be treated as a sentence boundary.
        let out = agg.aggregate("pi is 3.14 ok");
        assert!(out.is_empty(), "decimal split: {out:?}");
        assert_eq!(agg.flush().as_deref(), Some("pi is 3.14 ok"));
    }

    #[test]
    fn skip_tags_keeps_tagged_region_intact() {
        let mut agg = SkipTagsAggregator::new(vec![StartEndTags::new("<think>", "</think>")]);
        // The period inside the tag must not split.
        let out = agg.aggregate("<think>one. two.</think>");
        assert!(out.is_empty(), "split inside tag: {out:?}");
        // A sentence after the closing tag emits normally.
        let out2 = agg.aggregate(" Done. Next");
        assert_eq!(out2, vec!["<think>one. two.</think> Done."]);
    }

    #[test]
    fn pattern_pair_remove_strips_commands_and_keeps_sentences() {
        let mut agg = PatternPairAggregator::new().add_pattern(
            "dtmf",
            "<dtmf>",
            "</dtmf>",
            MatchAction::Remove,
        );
        let matches = agg.aggregate_matches("Press <dtmf>1</dtmf> now. Then");
        // The command is removed; the surrounding text emits as a sentence.
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].id, "sentence");
        assert_eq!(matches[0].content, "Press  now.");
    }

    #[test]
    fn pattern_pair_aggregate_surfaces_command() {
        let mut agg = PatternPairAggregator::new().add_pattern(
            "dtmf",
            "<dtmf>",
            "</dtmf>",
            MatchAction::Aggregate,
        );
        let matches = agg.aggregate_matches("<dtmf>5</dtmf>");
        assert_eq!(
            matches,
            vec![PatternMatch {
                id: "dtmf".into(),
                content: "5".into()
            }]
        );
    }

    #[test]
    fn markdown_filter_strips_formatting() {
        let mut f = MarkdownTextFilter::default();
        let cleaned = f.filter("**Bold** and `code` and a [link](https://x.io).");
        assert_eq!(cleaned, "Bold and code and a link.");
        let heading = f.filter("# Title");
        assert_eq!(heading, "Title");
        let bullet = f.filter("- item one");
        assert_eq!(bullet, "item one");
    }

    #[test]
    fn markdown_filter_disabled_is_passthrough() {
        let mut f = MarkdownTextFilter::new(MarkdownFilterParams { enabled: false });
        assert_eq!(f.filter("**keep**"), "**keep**");
    }

    #[tokio::test]
    async fn aggregator_processor_emits_text_frames_per_sentence() {
        let proc = TextAggregatorProcessor::new("agg", SimpleTextAggregator::new());
        let out = drive(
            Box::new(proc),
            vec![
                Frame::LlmText("Hi there.".into()),
                Frame::LlmText(" Bye now.".into()),
                Frame::LlmResponseEnd,
            ],
            Direction::Downstream,
        )
        .await;
        let texts: Vec<String> = out
            .into_iter()
            .filter_map(|f| match f {
                Frame::Text(t) => Some(t),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["Hi there.", "Bye now."]);
    }

    #[tokio::test]
    async fn filter_processor_cleans_text_frames() {
        let proc = TextFilterProcessor::new("md", MarkdownTextFilter::default());
        let out = drive(
            Box::new(proc),
            vec![Frame::LlmText("**hello**".into())],
            Direction::Downstream,
        )
        .await;
        assert!(matches!(out.first(), Some(Frame::LlmText(t)) if t == "hello"));
    }
}
