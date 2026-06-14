// SPDX-License-Identifier: Apache-2.0
//
//! Function-call filters (pure logic, no network).
//!
//! Guards over [`Frame::FunctionCallsStarted`] before a tool is dispatched —
//! the gate that keeps a model from invoking tools it should not. Mirrors the
//! intent of pipecat's `FunctionFilter` (a predicate gate) specialised for
//! function-call frames, plus the allow-list / dedup behaviour an agent runtime
//! wants in front of an MCP/tool dispatcher (this is the in-core, network-free
//! half; the actual dispatch lives in `flowcat-services::mcp`).
//!
//! [`FunctionCallFilter`] applies, per call in the batch:
//! 1. **allow-listing** — drop calls whose `function_name` is not permitted;
//! 2. **argument validation** — drop calls whose required argument keys are
//!    missing (a cheap structural check, no JSON-Schema engine);
//! 3. **de-duplication** — drop a repeat of a `(function_name, arguments)` pair
//!    already seen this session (idempotent tool calls).
//!
//! A batch that ends up empty is dropped entirely; otherwise the surviving calls
//! are re-emitted as a single [`Frame::FunctionCallsStarted`]. All non-function
//! frames pass through unchanged.

use std::collections::HashSet;

use async_trait::async_trait;

use crate::error::Result;
use crate::processor::frame::{Frame, FunctionCall};
use crate::processor::{Envelope, FrameProcessor, Link};

/// How a function call is permitted.
#[derive(Debug, Clone)]
pub enum AllowPolicy {
    /// Permit any function name (validation/dedup still apply).
    Any,
    /// Permit only these exact function names.
    Only(HashSet<String>),
}

/// Required-argument-keys rule for one function: the call is dropped unless every
/// listed key is present in its `arguments` object.
#[derive(Debug, Clone, Default)]
pub struct RequiredArgs {
    /// `function_name` → list of required top-level argument keys.
    pub by_function: std::collections::HashMap<String, Vec<String>>,
}

/// Filters [`Frame::FunctionCallsStarted`] batches: allow-list, validate required
/// args, and de-duplicate `(name, args)` pairs within a session. See the module
/// docs for the per-call pipeline.
pub struct FunctionCallFilter {
    name: &'static str,
    allow: AllowPolicy,
    required: RequiredArgs,
    dedup: bool,
    seen: HashSet<String>,
}

impl FunctionCallFilter {
    /// A filter with the given allow policy. Dedup on, no required-arg rules by
    /// default — chain [`Self::require_args`] / [`Self::without_dedup`] to adjust.
    pub fn new(allow: AllowPolicy) -> Self {
        Self {
            name: "fcall-filter",
            allow,
            required: RequiredArgs::default(),
            dedup: true,
            seen: HashSet::new(),
        }
    }

    /// Require `keys` to be present in `function`'s arguments object.
    pub fn require_args(mut self, function: impl Into<String>, keys: Vec<String>) -> Self {
        self.required.by_function.insert(function.into(), keys);
        self
    }

    /// Disable `(name, args)` de-duplication (allow repeated identical calls).
    pub fn without_dedup(mut self) -> Self {
        self.dedup = false;
        self
    }

    /// Decide whether a single call survives the allow-list + arg-validation +
    /// dedup pipeline. Mutates `self.seen` for dedup bookkeeping.
    fn permits(&mut self, call: &FunctionCall) -> bool {
        // 1. Allow-list.
        let allowed = match &self.allow {
            AllowPolicy::Any => true,
            AllowPolicy::Only(set) => set.contains(&call.function_name),
        };
        if !allowed {
            return false;
        }
        // 2. Required-argument structural validation.
        if let Some(keys) = self.required.by_function.get(&call.function_name) {
            let obj = call.arguments.as_object();
            let all_present = keys.iter().all(|k| obj.is_some_and(|o| o.contains_key(k)));
            if !all_present {
                return false;
            }
        }
        // 3. De-duplication of identical (name, args) pairs.
        if self.dedup {
            // Canonical key: name + compact JSON of the arguments.
            let key = format!("{}:{}", call.function_name, call.arguments);
            if !self.seen.insert(key) {
                return false;
            }
        }
        true
    }
}

#[async_trait]
impl FrameProcessor for FunctionCallFilter {
    fn name(&self) -> &str {
        self.name
    }

    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        let Frame::FunctionCallsStarted(ref calls) = env.frame else {
            link.push(env.meta, env.frame, env.direction).await;
            return Ok(());
        };

        let kept: Vec<FunctionCall> = calls.iter().filter(|c| self.permits(c)).cloned().collect();

        // An empty surviving batch is dropped entirely (no tool dispatch).
        if kept.is_empty() {
            return Ok(());
        }
        link.push(env.meta, Frame::FunctionCallsStarted(kept), env.direction)
            .await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::test_harness::drive;
    use crate::processor::frame::Direction;
    use serde_json::json;

    fn call(name: &str, args: serde_json::Value) -> FunctionCall {
        FunctionCall {
            function_name: name.into(),
            tool_call_id: format!("id-{name}"),
            arguments: args,
        }
    }

    fn only(names: &[&str]) -> AllowPolicy {
        AllowPolicy::Only(names.iter().map(|s| s.to_string()).collect())
    }

    #[tokio::test]
    async fn allow_list_drops_unpermitted_calls() {
        let filter = FunctionCallFilter::new(only(&["get_weather"]));
        let out = drive(
            Box::new(filter),
            vec![Frame::FunctionCallsStarted(vec![
                call("get_weather", json!({"city": "SG"})),
                call("delete_account", json!({})), // not allowed → dropped
            ])],
            Direction::Downstream,
        )
        .await;
        let kept = match out.first() {
            Some(Frame::FunctionCallsStarted(c)) => c.clone(),
            other => panic!("expected FunctionCallsStarted, got {other:?}"),
        };
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].function_name, "get_weather");
    }

    #[tokio::test]
    async fn required_args_validation_drops_incomplete_calls() {
        let filter = FunctionCallFilter::new(AllowPolicy::Any)
            .require_args("book", vec!["date".into(), "name".into()]);
        let out = drive(
            Box::new(filter),
            vec![Frame::FunctionCallsStarted(vec![
                call("book", json!({"date": "today"})), // missing "name" → dropped
                call("book", json!({"date": "today", "name": "Sam"})), // ok
            ])],
            Direction::Downstream,
        )
        .await;
        let kept = match out.first() {
            Some(Frame::FunctionCallsStarted(c)) => c.clone(),
            other => panic!("expected FunctionCallsStarted, got {other:?}"),
        };
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].arguments["name"], "Sam");
    }

    #[tokio::test]
    async fn dedup_drops_repeated_identical_calls() {
        let filter = FunctionCallFilter::new(AllowPolicy::Any);
        let out = drive(
            Box::new(filter),
            vec![
                Frame::FunctionCallsStarted(vec![call("ping", json!({"n": 1}))]),
                Frame::FunctionCallsStarted(vec![call("ping", json!({"n": 1}))]), // dup → dropped
                Frame::FunctionCallsStarted(vec![call("ping", json!({"n": 2}))]), // distinct args → ok
            ],
            Direction::Downstream,
        )
        .await;
        let batches: Vec<usize> = out
            .iter()
            .filter_map(|f| match f {
                Frame::FunctionCallsStarted(c) => Some(c.len()),
                _ => None,
            })
            .collect();
        // First batch (1 call) and third batch (1 call) survive; the dup batch is
        // dropped entirely (empty → no frame).
        assert_eq!(batches, vec![1, 1]);
    }

    #[tokio::test]
    async fn empty_surviving_batch_is_dropped() {
        let filter = FunctionCallFilter::new(only(&["allowed"]));
        let out = drive(
            Box::new(filter),
            vec![Frame::FunctionCallsStarted(vec![call(
                "blocked",
                json!({}),
            )])],
            Direction::Downstream,
        )
        .await;
        assert!(
            out.is_empty(),
            "all-blocked batch must emit nothing: {out:?}"
        );
    }

    #[tokio::test]
    async fn non_function_frames_pass_through() {
        let filter = FunctionCallFilter::new(only(&["x"]));
        let out = drive(
            Box::new(filter),
            vec![Frame::Text("hello".into())],
            Direction::Downstream,
        )
        .await;
        assert!(matches!(out.first(), Some(Frame::Text(t)) if t == "hello"));
    }
}
