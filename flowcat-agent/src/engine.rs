// SPDX-License-Identifier: Apache-2.0
//
//! The graph decision engine. Pure logic: given a parsed graph and the run's
//! variables, it tracks the active node, composes the per-node system prompt and
//! the set of callable transitions, interpolates `{{variables}}`, and advances on
//! a transition/tool call. No dependency on the media runtime — a host wraps it
//! behind a `flowcat_core::AgentBrain` (the `DeclarativeBrain` adapter does this).

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;

use crate::graph::{Graph, GraphError};

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error(transparent)]
    Graph(#[from] GraphError),
    #[error("unknown node: {0}")]
    UnknownNode(String),
}

/// How a human transfer is performed (cold / warm). **Operator-pinned** (node
/// config), never LLM-chosen — same closed trust model as `target_phone`. `cold`
/// ends the agent then bridges; `warm` keeps the agent streaming, dials +
/// introduces the human, then bridges (honored only where the host supports it;
/// falls back to cold otherwise).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum TransferMode {
    #[default]
    Cold,
    Warm,
}

/// What the host should do after a tool/transition call.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Action {
    /// Move to a node; the host should re-prompt the LLM with the new context.
    Transition {
        to_node: String,
        /// Optional speech to play during the transition (text or recording id).
        say: Option<String>,
    },
    /// Stay on the current node (e.g. an informational tool result).
    Stay,
    /// End the call/session.
    End,
    /// Escalate to a human. The host records a pending transfer for the run and
    /// closes the media WS so the cold-transfer fallthrough dials `target_phone`.
    /// `target_phone` is **operator-pinned** (read from node config) — never
    /// LLM-chosen — so the set of reachable numbers is a closed list.
    Transfer {
        target_phone: String,
        target_name: Option<String>,
        /// Optional pre-transfer line spoken before the WS closes.
        say: Option<String>,
        /// cold (default) or warm (honored only where the host supports it).
        #[serde(default)]
        mode: TransferMode,
    },
}

/// The prompt + transitions the host needs to configure the LLM for a turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodePrompt {
    pub node_id: String,
    pub system_prompt: String,
    /// Tool/transition names the LLM may call from this node.
    pub transitions: Vec<TransitionSpec>,
    /// Operator-configured human-transfer targets exposed on this node (0 or more).
    /// Each becomes a no-arg LLM function the host registers; the engine never lets
    /// the LLM pick the number — `target_phone` comes straight from node config.
    #[serde(default)]
    pub transfer_targets: Vec<TransferTarget>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionSpec {
    /// Stable name the LLM calls to take this edge.
    pub name: String,
    pub target: String,
    pub description: Option<String>,
}

/// One operator-pinned human-escalation destination attached to a node. The
/// LLM-facing function is no-arg: the model decides *whether* to escalate, never
/// *to where* — the dialed number is fixed by node config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferTarget {
    /// Stable function name the LLM calls (slugified tool name; default `transfer_call`).
    pub name: String,
    /// Operator-pinned E.164 destination. Never `var_interpolate`d — it must be a
    /// literal the conversation can't rewrite.
    pub target_phone: String,
    #[serde(default)]
    pub target_name: Option<String>,
    /// LLM-facing "when to transfer" description (the tool's description field).
    #[serde(default)]
    pub description: Option<String>,
    /// Optional pre-transfer spoken line. Audio-recording variant deferred.
    #[serde(default)]
    pub say: Option<String>,
    /// Escalation style: cold (default) or warm. Operator-pinned (not LLM-chosen);
    /// warm is honored only where the host supports it (otherwise cold).
    #[serde(default)]
    pub mode: TransferMode,
}

/// Declared type of an extracted variable. Drives coercion of the LLM's JSON.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum VarType {
    #[default]
    String,
    Number,
    Boolean,
}

/// One variable a node wants harvested from the conversation. Mirrors the
/// node-spec `extraction_variables[]` entry (`name` / `type` / `prompt`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionVar {
    pub name: String,
    #[serde(rename = "type", default)]
    pub var_type: VarType,
    /// Per-variable hint (node-spec field `prompt`).
    #[serde(default, rename = "prompt")]
    pub hint: Option<String>,
}

/// A node's variable-extraction configuration, when enabled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionSpec {
    /// Overall extraction instruction (`extraction_prompt`).
    #[serde(default)]
    pub prompt: Option<String>,
    pub variables: Vec<ExtractionVar>,
}

/// The active node's identity for a live-transcript node-transition marker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeInfo {
    pub id: String,
    pub name: String,
    pub allow_interrupt: Option<bool>,
}

/// Stateful engine for one run.
pub struct Engine {
    graph: Graph,
    current: String,
    pub vars: Map<String, Value>,
    runtime_options: Value,
    /// Ordered, first-visit-deduped display names of the nodes this run has been
    /// active on — surfaced as `collected_vars.nodes_visited` for observability.
    visited: Vec<String>,
}

impl Engine {
    /// Build from the persisted graph spec, runtime options and seed variables.
    pub fn new(
        graph_spec: &Value,
        runtime_options: Value,
        seed_vars: Map<String, Value>,
    ) -> Result<Self, EngineError> {
        let graph = Graph::parse(graph_spec)?;
        let current = graph.start_id.clone();
        let mut engine = Self {
            graph,
            current,
            vars: seed_vars,
            runtime_options,
            visited: Vec::new(),
        };
        // Auto-advance off the start node to the first real node if there is a
        // single unconditional edge (start nodes carry no prompt of their own).
        engine.advance_through_start();
        engine.record_visit();
        Ok(engine)
    }

    /// Skip a start node ONLY when it carries no opening prompt — i.e. the
    /// synthetic `{type:"start"}` anchor used in hand-written graphs/tests.
    /// Editor-authored `startCall` nodes carry a `data.prompt` (the agent's
    /// opening utterance) and must be preserved as the active node so the first
    /// turn delivers that prompt. Both kinds collapse to `NodeKind::Start` via
    /// `#[serde(alias="startCall")]`, so we disambiguate by content, not by name.
    fn advance_through_start(&mut self) {
        let Some(node) = self.graph.node(&self.current) else {
            return;
        };
        if !matches!(node.kind, crate::graph::NodeKind::Start) {
            return;
        }
        let has_prompt = node
            .data
            .get("prompt")
            .and_then(|v| v.as_str())
            .map(|s| !s.trim().is_empty())
            .unwrap_or(false);
        if has_prompt {
            return;
        }
        if let Some(edge) = self.graph.outgoing(&self.current).first() {
            self.current = edge.target.clone();
        }
    }

    /// Re-seat the engine at a previously persisted node id (resume after a
    /// snapshot / restart). Returns `UnknownNode` if the id is not in the parsed
    /// graph.
    pub fn set_current_node(&mut self, node_id: &str) -> Result<(), EngineError> {
        if self.graph.node(node_id).is_some() {
            self.current = node_id.to_string();
            self.record_visit();
            Ok(())
        } else {
            Err(EngineError::UnknownNode(node_id.to_string()))
        }
    }

    pub fn current_node(&self) -> &str {
        &self.current
    }

    /// The active node's id, display name, and `allow_interrupt` flag — for a
    /// node-transition transcript marker. `allow_interrupt` is read from the node's
    /// `data` (`allow_interrupt`, or the `allowInterrupt` alias); `None` when the
    /// field is absent.
    pub fn current_node_info(&self) -> NodeInfo {
        let allow_interrupt = self.graph.node(&self.current).and_then(|n| {
            n.data
                .get("allow_interrupt")
                .or_else(|| n.data.get("allowInterrupt"))
                .and_then(|v| v.as_bool())
        });
        NodeInfo {
            id: self.current.clone(),
            name: self.graph.node_name(&self.current),
            allow_interrupt,
        }
    }

    /// The ordered, deduped display names of nodes this run has visited.
    pub fn visited_nodes(&self) -> &[String] {
        &self.visited
    }

    /// Record the active node in `visited` (first-visit order, no duplicates).
    fn record_visit(&mut self) {
        let name = self.graph.node_name(&self.current);
        if !self.visited.iter().any(|n| n == &name) {
            self.visited.push(name);
        }
    }

    pub fn is_finished(&self) -> bool {
        self.graph.is_terminal(&self.current)
    }

    /// Compose the prompt and the set of callable transitions for the active node.
    pub fn current_prompt(&self) -> NodePrompt {
        let node = self.graph.node(&self.current);
        let raw_prompt = node
            .and_then(|n| n.data.get("prompt"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let mut system_prompt = var_interpolate(&raw_prompt, &self.vars);

        // Prepend the global directive (persona/tone) when this node opts in.
        // The editor writes `add_global_prompt` per node (alias
        // `inherit_global_directive`); default is to inherit when a global node
        // exists.
        let inherit_global = node
            .and_then(|n| {
                n.data
                    .get("add_global_prompt")
                    .or_else(|| n.data.get("inherit_global_directive"))
            })
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        if inherit_global {
            if let Some(global_raw) = self.graph.global_prompt() {
                let global = var_interpolate(global_raw, &self.vars);
                if !global.trim().is_empty() {
                    system_prompt = if system_prompt.trim().is_empty() {
                        global
                    } else {
                        format!("{global}\n\n{system_prompt}")
                    };
                }
            }
        }

        let transitions = self
            .graph
            .outgoing(&self.current)
            .into_iter()
            .enumerate()
            .map(|(i, e)| {
                // The description the LLM reads to choose this edge is the edge's
                // transition criteria (field `condition`; alias
                // `transition_criteria`), falling back to the label.
                let description = e
                    .data
                    .get("transition_criteria")
                    .or_else(|| e.data.get("condition"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .filter(|s| !s.trim().is_empty())
                    .or_else(|| e.label.clone());
                TransitionSpec {
                    name: e
                        .label
                        .clone()
                        .map(|l| slugify(&l))
                        .unwrap_or_else(|| format!("transition_{i}")),
                    target: e.target.clone(),
                    description,
                }
            })
            .collect();

        // Operator-pinned human-transfer destinations. Read
        // from the node's `data.transfer_targets` array; `target_phone` is taken
        // LITERAL (never interpolated — the conversation must not be able to
        // rewrite the dialed number), while `say`/`description` get the same
        // var-interpolation as prompts. Entries without a target_phone are dropped.
        let transfer_targets: Vec<TransferTarget> = node
            .and_then(|n| n.data.get("transfer_targets"))
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .enumerate()
                    .filter_map(|(i, t)| {
                        let target_phone = t
                            .get("target_phone")
                            .and_then(Value::as_str)
                            .map(str::trim)
                            .filter(|s| !s.is_empty())?
                            .to_string();
                        let name = t
                            .get("name")
                            .and_then(Value::as_str)
                            .map(slugify)
                            .filter(|s| !s.is_empty())
                            .unwrap_or_else(|| {
                                if i == 0 {
                                    "transfer_call".to_string()
                                } else {
                                    format!("transfer_call_{i}")
                                }
                            });
                        let target_name = t
                            .get("target_name")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                            .filter(|s| !s.trim().is_empty());
                        let description = t
                            .get("description")
                            .and_then(Value::as_str)
                            .filter(|s| !s.trim().is_empty())
                            .map(|s| var_interpolate(s, &self.vars));
                        let say = t
                            .get("say")
                            .and_then(Value::as_str)
                            .filter(|s| !s.trim().is_empty())
                            .map(|s| var_interpolate(s, &self.vars));
                        // Operator-pinned escalation style (not interpolated). Only
                        // "warm" opts into warm transfer; anything else is cold.
                        let mode = match t.get("mode").and_then(Value::as_str) {
                            Some("warm") => TransferMode::Warm,
                            _ => TransferMode::Cold,
                        };
                        Some(TransferTarget {
                            name,
                            target_phone,
                            target_name,
                            description,
                            say,
                            mode,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        // When this node exposes human-transfer targets, make the capability
        // explicit in the system prompt. A realtime LLM will otherwise sometimes
        // narrate "I can't transfer you" rather than call the (registered) no-arg
        // function. The function still owns the pinned number; this only tells the
        // model the capability exists and when to use it.
        if !transfer_targets.is_empty() {
            let names = transfer_targets
                .iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>()
                .join("` or `");
            system_prompt.push_str(&format!(
                "\n\n[Human escalation] If the caller asks to speak to a human, a person, \
                 an agent, a manager, or to be transferred: FIRST say one short sentence \
                 telling them you're connecting them to a human now and to please hold, \
                 THEN call the `{names}` function. Never claim you are unable to transfer."
            ));
        }

        NodePrompt {
            node_id: self.current.clone(),
            system_prompt,
            transitions,
            transfer_targets,
        }
    }

    /// Apply a transition the LLM requested by name. Unknown names keep the
    /// engine on the current node (`Stay`).
    pub fn on_transition(&mut self, name: &str) -> Action {
        let target = {
            let specs = self.current_prompt().transitions;
            specs.into_iter().find(|t| t.name == name).map(|t| t.target)
        };
        match target {
            Some(to) => {
                self.current = to.clone();
                self.record_visit();
                if self.is_finished() {
                    Action::End
                } else {
                    Action::Transition {
                        to_node: to,
                        say: None,
                    }
                }
            }
            None => Action::Stay,
        }
    }

    /// Apply a human-transfer the LLM requested by name. Looks
    /// the name up in the active node's operator-configured `transfer_targets` and
    /// returns `Action::Transfer` with the **pinned** number; an unknown name keeps
    /// the engine on the current node (`Stay`). Unlike `on_transition`, this does
    /// NOT advance the active node — the call ends via the media path (the host
    /// records the pending transfer + closes the WS), like `endCall`.
    pub fn on_transfer(&self, name: &str) -> Action {
        match self
            .current_prompt()
            .transfer_targets
            .into_iter()
            .find(|t| t.name == name)
        {
            Some(t) => Action::Transfer {
                target_phone: t.target_phone,
                target_name: t.target_name,
                say: t.say,
                mode: t.mode,
            },
            None => Action::Stay,
        }
    }

    /// The active node's variable-extraction config, if extraction is enabled
    /// and at least one variable is declared. Pure — the host runs the actual
    /// LLM extraction pass and feeds results back via [`Engine::capture`].
    pub fn extraction_spec(&self) -> Option<ExtractionSpec> {
        let node = self.graph.node(&self.current)?;
        let enabled = node
            .data
            .get("extraction_enabled")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if !enabled {
            return None;
        }
        let variables: Vec<ExtractionVar> = node
            .data
            .get("extraction_variables")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|e| serde_json::from_value(e.clone()).ok())
                    .collect()
            })
            .unwrap_or_default();
        if variables.is_empty() {
            return None;
        }
        let prompt = node
            .data
            .get("extraction_prompt")
            .and_then(Value::as_str)
            .map(str::to_string)
            .filter(|s| !s.trim().is_empty());
        Some(ExtractionSpec { prompt, variables })
    }

    /// The tool ids (`tool_uuids`) the active node enables. The host loads the
    /// tool definitions (from the store) and offers them to the LLM; pure here.
    pub fn tool_refs(&self) -> Vec<String> {
        self.string_array_field("tool_uuids")
    }

    /// The knowledge-base document ids (`document_uuids`) the active node attaches.
    /// The host offers a retrieval tool scoped to these documents; pure here.
    pub fn document_refs(&self) -> Vec<String> {
        self.string_array_field("document_uuids")
    }

    /// Read a string-array field from the current node's `data` (empty if absent).
    fn string_array_field(&self, key: &str) -> Vec<String> {
        self.graph
            .node(&self.current)
            .and_then(|n| n.data.get(key))
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Merge extracted variables into the run context.
    pub fn capture(&mut self, vars: Map<String, Value>) {
        for (k, v) in vars {
            self.vars.insert(k, v);
        }
    }

    /// Serialize enough state to resume later (rewind / recovery).
    pub fn snapshot(&self) -> Value {
        serde_json::json!({
            "current": self.current,
            "vars": self.vars,
            "runtime_options": self.runtime_options,
        })
    }
}

/// Interpolate `{{ path | fallback }}` placeholders in a string against the run
/// variables. Supports:
/// - **nested dot-paths** — `{{customer.contact.email}}` (object keys; numeric
///   segments index arrays);
/// - **fallbacks** — `{{name | there}}` and the legacy `{{name | fallback:there}}`;
///   used when the path is missing, null, or an empty string;
/// - **JSON values** — object/array values are serialised as JSON, scalars as
///   their plain string;
/// - literal `\n` → newline.
///
/// A missing path with no fallback renders empty. An unterminated `{{` is left
/// literal. Also resolves **time builtins** against the wall clock:
/// `{{current_time}}` / `{{current_date}}` / `{{current_weekday}}` (UTC) and a
/// `_<IANA TZ>` suffix — `{{current_time_America/New_York}}` — for a local time.
/// `{{greeting_phrase}}` renders an hour-bucketed time-of-day phrase ("Good
/// morning" / "Good afternoon" / "Good evening" / "Hello"); it defaults to
/// Asia/Singapore when bare and honours the same `_<IANA TZ>` suffix.
pub fn var_interpolate(template: &str, vars: &Map<String, Value>) -> String {
    var_interpolate_at(template, vars, chrono::Utc::now())
}

/// Pure core of [`var_interpolate`]: interpolates against an explicit `now`, so
/// the time builtins are deterministically testable. The public wrapper supplies
/// the wall clock — the only impurity lives there.
fn var_interpolate_at(
    template: &str,
    vars: &Map<String, Value>,
    now: chrono::DateTime<chrono::Utc>,
) -> String {
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(open) = rest.find("{{") {
        out.push_str(&rest[..open]);
        let after = &rest[open + 2..];
        match after.find("}}") {
            Some(close) => {
                let inner = &after[..close];
                // Time builtins take precedence over a (non-existent) variable of
                // the same name; everything else resolves against `vars`.
                match time_builtin(inner.trim(), now) {
                    Some(t) => out.push_str(&t),
                    None => out.push_str(&resolve_placeholder(inner, vars)),
                }
                rest = &after[close + 2..];
            }
            None => {
                // No closing braces — emit the literal `{{` and stop scanning.
                out.push_str("{{");
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out.replace("\\n", "\n")
}

/// Resolve a `current_time` / `current_date` / `current_weekday` /
/// `greeting_phrase` placeholder (optionally `_<IANA TZ>`-suffixed) against `now`.
/// Returns `None` for any other placeholder (so it falls through to variable
/// resolution). An unrecognised/bad timezone yields `None` (renders empty rather
/// than a wrong time).
fn time_builtin(inner: &str, now: chrono::DateTime<chrono::Utc>) -> Option<String> {
    use chrono::{Timelike, Utc};
    // `greeting_phrase` → a time-of-day phrase, hour-bucketed. Unlike the `current_*`
    // builtins (UTC when bare), the bare form defaults to Asia/Singapore — the
    // primary market — since a time-of-day greeting only makes sense in a wall-clock
    // local to the caller. A `_<IANA TZ>` suffix overrides that default.
    if let Some(rest) = inner.strip_prefix("greeting_phrase") {
        let hour = if rest.is_empty() {
            now.with_timezone(&chrono_tz::Asia::Singapore).hour()
        } else {
            let tz: chrono_tz::Tz = rest.strip_prefix('_')?.parse().ok()?;
            now.with_timezone(&tz).hour()
        };
        let phrase = match hour {
            5..=11 => "Good morning",
            12..=16 => "Good afternoon",
            17..=21 => "Good evening",
            _ => "Hello", // 22:00–04:59 — neutral; "good night" reads as a farewell.
        };
        return Some(phrase.to_string());
    }
    // (format, tz-suffix-or-empty). Order: longest distinct prefixes; the three
    // kinds don't overlap.
    let (fmt, tz_part) = if let Some(rest) = inner.strip_prefix("current_time") {
        ("%Y-%m-%d %H:%M %Z", rest)
    } else if let Some(rest) = inner.strip_prefix("current_weekday") {
        ("%A", rest)
    } else {
        ("%Y-%m-%d", inner.strip_prefix("current_date")?)
    };
    if tz_part.is_empty() {
        // UTC.
        return Some(now.with_timezone(&Utc).format(fmt).to_string());
    }
    // `_<IANA name>`, e.g. `_America/New_York`. A leading `_` is required; a name
    // we can't parse means "not a time builtin" → fall through (renders empty).
    let tz_name = tz_part.strip_prefix('_')?;
    let tz: chrono_tz::Tz = tz_name.parse().ok()?;
    Some(now.with_timezone(&tz).format(fmt).to_string())
}

/// Recursively interpolate a JSON template: string leaves run through
/// [`var_interpolate`]; object values and array items recurse (object keys are
/// left as-is). Used for HTTP tool request bodies/params.
pub fn var_interpolate_value(template: &Value, vars: &Map<String, Value>) -> Value {
    match template {
        Value::String(s) => Value::String(var_interpolate(s, vars)),
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|i| var_interpolate_value(i, vars))
                .collect(),
        ),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), var_interpolate_value(v, vars)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Resolve one `path | fallback` placeholder body.
fn resolve_placeholder(inner: &str, vars: &Map<String, Value>) -> String {
    let (path, fallback) = match inner.split_once('|') {
        Some((p, f)) => {
            let f = f.trim();
            // Both `{{x | fallback:default}}` (legacy) and `{{x | default}}`.
            let fb = f.strip_prefix("fallback:").map(str::trim).unwrap_or(f);
            (p.trim(), Some(fb))
        }
        None => (inner.trim(), None),
    };
    match get_nested(vars, path) {
        Some(v) if !is_empty_value(v) => value_to_string(v),
        _ => fallback.unwrap_or("").to_string(),
    }
}

/// Traverse a dot-path through objects (by key) and arrays (by index).
fn get_nested<'a>(vars: &'a Map<String, Value>, path: &str) -> Option<&'a Value> {
    let mut parts = path.split('.');
    let mut cur = vars.get(parts.next()?)?;
    for p in parts {
        cur = match cur {
            Value::Object(_) => cur.get(p)?,
            Value::Array(arr) => arr.get(p.parse::<usize>().ok()?)?,
            _ => return None,
        };
    }
    Some(cur)
}

/// A null or empty-string value triggers the fallback.
fn is_empty_value(v: &Value) -> bool {
    matches!(v, Value::Null) || matches!(v, Value::String(s) if s.is_empty())
}

/// Render a resolved value: strings verbatim, objects/arrays as JSON, scalars
/// as their plain representation, null as empty.
fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        Value::Object(_) | Value::Array(_) => v.to_string(),
        other => other.to_string(),
    }
}

fn slugify(label: &str) -> String {
    label
        .chars()
        .map(|c| {
            if c.is_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}

/// Validate a graph spec without instantiating a full run (structural parse
/// only: start node present, no dangling edges).
pub fn validate(graph_spec: &Value) -> Result<(), EngineError> {
    Graph::parse(graph_spec)?;
    Ok(())
}

/// Full validation for an editor's validate endpoint: structural parse
/// (start node present, no dangling edges) PLUS per-node edge-cardinality rules.
/// Returns structured [`crate::graph::ValidationError`]s (empty ⇒ valid) so the
/// editor can highlight offending nodes; a structural parse failure becomes a
/// single graph-level error. Unlike [`validate`]/`Graph::parse` this is not used
/// to gate running a graph.
pub fn validate_detailed(graph_spec: &Value) -> Vec<crate::graph::ValidationError> {
    match Graph::parse(graph_spec) {
        Ok(graph) => graph.cardinality_errors(),
        Err(e) => vec![crate::graph::ValidationError {
            kind: "graph",
            id: None,
            message: e.to_string(),
        }],
    }
}

// Reserved for richer extraction config keyed by node id.
pub type ExtractionRules = HashMap<String, Value>;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample() -> Value {
        json!({
            "nodes": [
                {"id": "s", "type": "start"},
                {"id": "greet", "type": "agent", "data": {"prompt": "Hello {{name}}"}},
                {"id": "done", "type": "end"}
            ],
            "edges": [
                {"id": "e1", "source": "s", "target": "greet"},
                {"id": "e2", "source": "greet", "target": "done", "label": "Finish"}
            ]
        })
    }

    #[test]
    fn starts_on_first_real_node_with_rendered_prompt() {
        let mut seed = Map::new();
        seed.insert("name".into(), json!("Sam"));
        let eng = Engine::new(&sample(), json!({}), seed).unwrap();
        assert_eq!(eng.current_node(), "greet");
        assert_eq!(eng.current_prompt().system_prompt, "Hello Sam");
    }

    #[test]
    fn visited_nodes_accumulate_in_first_visit_order_deduped() {
        // start (no prompt → skipped) → greet ("Greeting") → done ("End Call").
        let spec = json!({
            "nodes": [
                {"id": "s", "type": "start"},
                {"id": "greet", "type": "agent", "data": {"prompt": "hi", "name": "Greeting"}},
                {"id": "done", "type": "end", "data": {"name": "End Call"}}
            ],
            "edges": [
                {"id": "e1", "source": "s", "target": "greet"},
                {"id": "e2", "source": "greet", "target": "done", "label": "Finish"}
            ]
        });
        let mut eng = Engine::new(&spec, json!({}), Map::new()).unwrap();
        // Construction records the active node (display name from `data.name`).
        assert_eq!(eng.visited_nodes(), ["Greeting".to_string()]);
        // Transition to the terminal node appends it. The transition name is the
        // slugified edge label ("Finish" → "finish").
        assert!(matches!(eng.on_transition("finish"), Action::End));
        assert_eq!(
            eng.visited_nodes(),
            ["Greeting".to_string(), "End Call".to_string()]
        );
        // Re-seating onto an already-visited node must NOT duplicate it.
        eng.set_current_node("greet").unwrap();
        assert_eq!(
            eng.visited_nodes(),
            ["Greeting".to_string(), "End Call".to_string()],
            "first-visit dedup"
        );
    }

    #[test]
    fn editor_authored_start_call_stays_active_with_its_prompt() {
        // The catalog's `startCall` carries a prompt — must NOT be skipped.
        let spec = json!({
            "nodes": [
                {"id": "s", "type": "startCall", "data": {"prompt": "Hi {{name}}!"}},
                {"id": "e", "type": "endCall", "data": {"prompt": "Bye."}}
            ],
            "edges": [
                {"id": "x", "source": "s", "target": "e", "label": "done"}
            ]
        });
        let mut seed = Map::new();
        seed.insert("name".into(), json!("Sam"));
        let eng = Engine::new(&spec, json!({}), seed).unwrap();
        assert_eq!(
            eng.current_node(),
            "s",
            "start node with prompt must remain active"
        );
        assert!(
            !eng.is_finished(),
            "session must not be finished on creation"
        );
        assert_eq!(eng.current_prompt().system_prompt, "Hi Sam!");
    }

    #[test]
    fn set_current_node_restores_persisted_state() {
        let mut eng = Engine::new(&sample(), json!({}), Map::new()).unwrap();
        // Fresh engine auto-advances past start (no prompt) to "greet".
        assert_eq!(eng.current_node(), "greet");
        // Simulate a persisted snapshot landing on "done"; the resume must
        // honor it instead of re-running auto-advance.
        eng.set_current_node("done").unwrap();
        assert_eq!(eng.current_node(), "done");
        assert!(eng.is_finished());
        // Unknown node ids are rejected.
        assert!(matches!(
            eng.set_current_node("nope"),
            Err(EngineError::UnknownNode(_))
        ));
    }

    #[test]
    fn global_directive_is_prepended_and_transition_criteria_used() {
        let spec = json!({
            "nodes": [
                {"id": "g", "type": "globalNode", "data": {"prompt": "You are {{brand}} support."}},
                {"id": "s", "type": "startCall", "data": {"prompt": "Hi, how can I help?"}},
                {"id": "book", "type": "agentNode", "data": {"prompt": "Booking flow."}},
                {"id": "done", "type": "endCall", "data": {"prompt": "Bye."}}
            ],
            "edges": [
                {"id": "e1", "source": "s", "target": "book", "label": "Book",
                 "data": {"condition": "Caller wants to book an appointment"}},
                {"id": "e2", "source": "s", "target": "done", "label": "Done",
                 "data": {"add_global_prompt": true}}
            ]
        });
        let mut seed = Map::new();
        seed.insert("brand".into(), json!("Acme"));
        let eng = Engine::new(&spec, json!({}), seed).unwrap();
        let p = eng.current_prompt();
        // Global persona prepended (rendered), then the node prompt.
        assert_eq!(
            p.system_prompt,
            "You are Acme support.\n\nHi, how can I help?"
        );
        // The edge's `condition` becomes the LLM-facing description; name is the
        // slugified label.
        let book = p.transitions.iter().find(|t| t.name == "book").unwrap();
        assert_eq!(
            book.description.as_deref(),
            Some("Caller wants to book an appointment")
        );
        // No condition → falls back to the label.
        let done = p.transitions.iter().find(|t| t.name == "done").unwrap();
        assert_eq!(done.description.as_deref(), Some("Done"));
    }

    #[test]
    fn node_can_opt_out_of_global_directive() {
        let spec = json!({
            "nodes": [
                {"id": "g", "type": "globalNode", "data": {"prompt": "Persona."}},
                {"id": "s", "type": "startCall", "data": {"prompt": "Body.", "add_global_prompt": false}},
                {"id": "done", "type": "endCall", "data": {}}
            ],
            "edges": [{"id": "e", "source": "s", "target": "done", "label": "x"}]
        });
        let eng = Engine::new(&spec, json!({}), Map::new()).unwrap();
        assert_eq!(eng.current_prompt().system_prompt, "Body.");
    }

    #[test]
    fn extraction_spec_reads_node_config() {
        let spec = json!({
            "nodes": [
                {"id": "s", "type": "startCall", "data": {
                    "prompt": "Ask budget.",
                    "extraction_enabled": true,
                    "extraction_prompt": "Capture budget and timeline.",
                    "extraction_variables": [
                        {"name": "budget", "type": "number", "prompt": "monthly budget in USD"},
                        {"name": "wants_demo", "type": "boolean"},
                        {"name": "company"}
                    ]
                }},
                {"id": "done", "type": "endCall", "data": {}}
            ],
            "edges": [{"id": "e", "source": "s", "target": "done", "label": "x"}]
        });
        let eng = Engine::new(&spec, json!({}), Map::new()).unwrap();
        let ex = eng.extraction_spec().expect("extraction enabled");
        assert_eq!(ex.prompt.as_deref(), Some("Capture budget and timeline."));
        assert_eq!(ex.variables.len(), 3);
        assert_eq!(ex.variables[0].name, "budget");
        assert_eq!(ex.variables[0].var_type, VarType::Number);
        assert_eq!(
            ex.variables[0].hint.as_deref(),
            Some("monthly budget in USD")
        );
        assert_eq!(ex.variables[1].var_type, VarType::Boolean);
        // `company` omits type/hint → defaults.
        assert_eq!(ex.variables[2].var_type, VarType::String);
        assert!(ex.variables[2].hint.is_none());
    }

    #[test]
    fn tool_refs_reads_node_tool_uuids() {
        let spec = json!({
            "nodes": [
                {"id": "s", "type": "startCall", "data": {"prompt": "hi", "tool_uuids": ["t-1", "t-2"]}},
                {"id": "e", "type": "endCall", "data": {}}
            ],
            "edges": [{"id": "z", "source": "s", "target": "e"}]
        });
        let eng = Engine::new(&spec, json!({}), Map::new()).unwrap();
        assert_eq!(eng.tool_refs(), vec!["t-1".to_string(), "t-2".to_string()]);
    }

    #[test]
    fn tool_refs_empty_when_absent() {
        let spec = json!({
            "nodes": [
                {"id": "s", "type": "startCall", "data": {"prompt": "hi"}},
                {"id": "e", "type": "endCall", "data": {}}
            ],
            "edges": [{"id": "z", "source": "s", "target": "e"}]
        });
        let eng = Engine::new(&spec, json!({}), Map::new()).unwrap();
        assert!(eng.tool_refs().is_empty());
    }

    #[test]
    fn document_refs_reads_node_document_uuids() {
        let spec = json!({
            "nodes": [
                {"id": "s", "type": "startCall", "data": {"prompt": "hi", "document_uuids": ["d-1", "d-2"]}},
                {"id": "e", "type": "endCall", "data": {}}
            ],
            "edges": [{"id": "z", "source": "s", "target": "e"}]
        });
        let eng = Engine::new(&spec, json!({}), Map::new()).unwrap();
        assert_eq!(
            eng.document_refs(),
            vec!["d-1".to_string(), "d-2".to_string()]
        );
        // Absent → empty.
        let spec2 = json!({
            "nodes": [
                {"id": "s", "type": "startCall", "data": {"prompt": "hi"}},
                {"id": "e", "type": "endCall", "data": {}}
            ],
            "edges": [{"id": "z", "source": "s", "target": "e"}]
        });
        assert!(Engine::new(&spec2, json!({}), Map::new())
            .unwrap()
            .document_refs()
            .is_empty());
    }

    #[test]
    fn extraction_spec_none_when_disabled_or_empty() {
        // disabled
        let spec = json!({
            "nodes": [
                {"id": "s", "type": "startCall", "data": {"prompt": "hi", "extraction_variables": [{"name": "x"}]}},
                {"id": "e", "type": "endCall"}
            ],
            "edges": [{"id": "z", "source": "s", "target": "e"}]
        });
        assert!(Engine::new(&spec, json!({}), Map::new())
            .unwrap()
            .extraction_spec()
            .is_none());
        // enabled but no variables
        let spec2 = json!({
            "nodes": [
                {"id": "s", "type": "startCall", "data": {"prompt": "hi", "extraction_enabled": true}},
                {"id": "e", "type": "endCall"}
            ],
            "edges": [{"id": "z", "source": "s", "target": "e"}]
        });
        assert!(Engine::new(&spec2, json!({}), Map::new())
            .unwrap()
            .extraction_spec()
            .is_none());
    }

    #[test]
    fn var_interpolate_nested_fallback_and_json() {
        let mut vars = Map::new();
        vars.insert("name".into(), json!("Sam"));
        vars.insert(
            "customer".into(),
            json!({ "contact": { "email": "s@x.io" }, "tags": ["vip", "new"] }),
        );
        vars.insert("empty".into(), json!(""));
        vars.insert("count".into(), json!(3));

        // flat + nested dot-path
        assert_eq!(var_interpolate("Hi {{name}}", &vars), "Hi Sam");
        assert_eq!(
            var_interpolate("{{customer.contact.email}}", &vars),
            "s@x.io"
        );
        // array index in a path
        assert_eq!(var_interpolate("{{customer.tags.0}}", &vars), "vip");
        // object/array values render as JSON
        assert_eq!(
            var_interpolate("{{customer.tags}}", &vars),
            r#"["vip","new"]"#
        );
        // scalar
        assert_eq!(var_interpolate("n={{count}}", &vars), "n=3");
        // fallbacks: missing, empty-string, and the legacy `fallback:` form
        assert_eq!(var_interpolate("Hi {{missing | there}}", &vars), "Hi there");
        assert_eq!(var_interpolate("Hi {{empty | there}}", &vars), "Hi there");
        assert_eq!(
            var_interpolate("Hi {{missing | fallback:friend}}", &vars),
            "Hi friend"
        );
        // present value beats the fallback
        assert_eq!(var_interpolate("Hi {{name | there}}", &vars), "Hi Sam");
        // missing with no fallback → empty
        assert_eq!(var_interpolate("[{{nope}}]", &vars), "[]");
        // literal \n becomes a newline
        assert_eq!(var_interpolate("a\\nb", &vars), "a\nb");
        // an unterminated placeholder is left literal
        assert_eq!(var_interpolate("a {{ b", &vars), "a {{ b");
    }

    #[test]
    fn var_interpolate_value_recurses_json_template() {
        let mut vars = Map::new();
        vars.insert("id".into(), json!("c-42"));
        vars.insert("city".into(), json!("Austin"));
        let template = json!({
            "url": "https://api.test/customers/{{id}}",
            "body": { "where": ["{{city}}", "fixed"], "note": "hi {{missing | n/a}}" },
            "keep": 7
        });
        let out = var_interpolate_value(&template, &vars);
        assert_eq!(out["url"], "https://api.test/customers/c-42");
        assert_eq!(out["body"]["where"][0], "Austin");
        assert_eq!(out["body"]["where"][1], "fixed");
        assert_eq!(out["body"]["note"], "hi n/a");
        assert_eq!(out["keep"], 7); // non-string leaves pass through
    }

    #[test]
    fn var_interpolate_resolves_time_builtins() {
        use chrono::{TimeZone, Utc};
        // 2024-01-05 14:30:00 UTC is a Friday.
        let now = Utc.with_ymd_and_hms(2024, 1, 5, 14, 30, 0).unwrap();
        let vars = Map::new();

        assert_eq!(
            var_interpolate_at("{{current_time}}", &vars, now),
            "2024-01-05 14:30 UTC"
        );
        assert_eq!(
            var_interpolate_at("{{current_date}}", &vars, now),
            "2024-01-05"
        );
        assert_eq!(
            var_interpolate_at("{{current_weekday}}", &vars, now),
            "Friday"
        );

        // Local time in a named TZ: 14:30 UTC = 09:30 EST (UTC-5 in January).
        assert_eq!(
            var_interpolate_at("{{current_time_America/New_York}}", &vars, now),
            "2024-01-05 09:30 EST"
        );
        assert_eq!(
            var_interpolate_at("{{current_weekday_America/New_York}}", &vars, now),
            "Friday"
        );

        // Embedded in a sentence + mixed with a normal variable.
        let mut v = Map::new();
        v.insert("name".into(), json!("Sam"));
        assert_eq!(
            var_interpolate_at("Hi {{name}}, it's {{current_weekday}}.", &v, now),
            "Hi Sam, it's Friday."
        );

        // An unknown timezone is not a valid builtin → renders empty (falls
        // through to variable resolution, which finds nothing).
        assert_eq!(
            var_interpolate_at("{{current_time_Not/AZone}}", &vars, now),
            ""
        );
        // A real variable named like a builtin-prefix is unaffected.
        let mut v2 = Map::new();
        v2.insert("current_year".into(), json!("2024"));
        assert_eq!(var_interpolate_at("{{current_year}}", &v2, now), "2024");
    }

    #[test]
    fn var_interpolate_resolves_greeting_phrase() {
        use chrono::{TimeZone, Utc};
        let vars = Map::new();
        // Bare form defaults to Asia/Singapore (UTC+8). Pick UTC instants so the
        // SG wall-clock hour lands squarely in each bucket.
        // 02:30 UTC = 10:30 SGT → morning.
        let morning = Utc.with_ymd_and_hms(2024, 1, 5, 2, 30, 0).unwrap();
        assert_eq!(
            var_interpolate_at("{{greeting_phrase}}", &vars, morning),
            "Good morning"
        );
        // 06:00 UTC = 14:00 SGT → afternoon.
        let afternoon = Utc.with_ymd_and_hms(2024, 1, 5, 6, 0, 0).unwrap();
        assert_eq!(
            var_interpolate_at("{{greeting_phrase}}", &vars, afternoon),
            "Good afternoon"
        );
        // 11:00 UTC = 19:00 SGT → evening.
        let evening = Utc.with_ymd_and_hms(2024, 1, 5, 11, 0, 0).unwrap();
        assert_eq!(
            var_interpolate_at("{{greeting_phrase}}", &vars, evening),
            "Good evening"
        );
        // 18:00 UTC = 02:00 SGT → late night → neutral "Hello".
        let night = Utc.with_ymd_and_hms(2024, 1, 5, 18, 0, 0).unwrap();
        assert_eq!(
            var_interpolate_at("{{greeting_phrase}}", &vars, night),
            "Hello"
        );

        // Bucket boundaries (SGT): 05:00 is morning, 04:59 is night; 12:00 is
        // afternoon; 17:00 is evening; 22:00 is night.
        let five_sgt = Utc.with_ymd_and_hms(2024, 1, 4, 21, 0, 0).unwrap(); // 05:00 SGT next day
        assert_eq!(
            var_interpolate_at("{{greeting_phrase}}", &vars, five_sgt),
            "Good morning"
        );
        let noon_sgt = Utc.with_ymd_and_hms(2024, 1, 5, 4, 0, 0).unwrap(); // 12:00 SGT
        assert_eq!(
            var_interpolate_at("{{greeting_phrase}}", &vars, noon_sgt),
            "Good afternoon"
        );
        let five_pm_sgt = Utc.with_ymd_and_hms(2024, 1, 5, 9, 0, 0).unwrap(); // 17:00 SGT
        assert_eq!(
            var_interpolate_at("{{greeting_phrase}}", &vars, five_pm_sgt),
            "Good evening"
        );
        let ten_pm_sgt = Utc.with_ymd_and_hms(2024, 1, 5, 14, 0, 0).unwrap(); // 22:00 SGT
        assert_eq!(
            var_interpolate_at("{{greeting_phrase}}", &vars, ten_pm_sgt),
            "Hello"
        );

        // `_<IANA TZ>` suffix overrides the default. 14:00 UTC = 09:00 EST → morning.
        let now = Utc.with_ymd_and_hms(2024, 1, 5, 14, 0, 0).unwrap();
        assert_eq!(
            var_interpolate_at("{{greeting_phrase_America/New_York}}", &vars, now),
            "Good morning"
        );
        // Same instant is 22:00 SGT → bare form is "Hello".
        assert_eq!(
            var_interpolate_at("{{greeting_phrase}}", &vars, now),
            "Hello"
        );

        // Embedded in a greeting alongside a normal variable.
        let mut v = Map::new();
        v.insert("first_name".into(), json!("Sam"));
        assert_eq!(
            var_interpolate_at("{{greeting_phrase}}, {{first_name}}!", &v, evening),
            "Good evening, Sam!"
        );

        // A bad timezone suffix is not a valid builtin → renders empty.
        assert_eq!(
            var_interpolate_at("{{greeting_phrase_Not/AZone}}", &vars, now),
            ""
        );
    }

    #[test]
    fn transition_reaches_terminal() {
        let eng_spec = sample();
        let mut eng = Engine::new(&eng_spec, json!({}), Map::new()).unwrap();
        let names: Vec<_> = eng
            .current_prompt()
            .transitions
            .into_iter()
            .map(|t| t.name)
            .collect();
        assert_eq!(names, vec!["finish"]);
        let action = eng.on_transition("finish");
        assert!(matches!(action, Action::End));
        assert!(eng.is_finished());
    }

    #[test]
    fn current_node_info_reports_id_name_and_interrupt() {
        let spec = json!({
            "nodes": [
                {"id": "s", "type": "startCall", "data": {"prompt": "hi", "allow_interrupt": true}},
                {"id": "done", "type": "endCall", "data": {}}
            ],
            "edges": [{"id": "e", "source": "s", "target": "done", "label": "Finish"}]
        });
        let mut eng = Engine::new(&spec, json!({}), Map::new()).unwrap();
        let info = eng.current_node_info();
        assert_eq!(info.id, "s");
        assert!(!info.name.is_empty());
        assert_eq!(
            info.allow_interrupt,
            Some(true),
            "reads allow_interrupt from node data"
        );
        // The accessor follows the engine after a transition; the end node has no flag.
        let tname = eng.current_prompt().transitions[0].name.clone();
        let _ = eng.on_transition(&tname);
        let info2 = eng.current_node_info();
        assert_eq!(info2.id, "done");
        assert_eq!(info2.allow_interrupt, None, "absent flag → None");
    }

    // ── agent-driven transferCall ──────────────────────────

    fn transfer_spec() -> Value {
        json!({
            "nodes": [
                {"id": "s", "type": "startCall", "data": {
                    "prompt": "Hi {{name}}.",
                    "transfer_targets": [
                        {"target_phone": "+918760036560", "target_name": "Support",
                         "description": "Escalate {{name}} to a human", "say": "Hold for {{name}}."}
                    ]
                }},
                {"id": "done", "type": "endCall", "data": {}}
            ],
            "edges": [{"id": "e", "source": "s", "target": "done", "label": "x"}]
        })
    }

    #[test]
    fn transfer_target_exposed_with_pinned_number_and_interpolated_copy() {
        let mut seed = Map::new();
        seed.insert("name".into(), json!("Sam"));
        let eng = Engine::new(&transfer_spec(), json!({}), seed).unwrap();
        let p = eng.current_prompt();
        assert_eq!(p.transfer_targets.len(), 1);
        let t = &p.transfer_targets[0];
        assert_eq!(
            t.name, "transfer_call",
            "default function name when unnamed"
        );
        assert_eq!(t.target_phone, "+918760036560");
        assert_eq!(t.target_name.as_deref(), Some("Support"));
        // say/description ARE var-interpolated (parity with prompts)…
        assert_eq!(t.say.as_deref(), Some("Hold for Sam."));
        assert_eq!(t.description.as_deref(), Some("Escalate Sam to a human"));
        // …and the system prompt gains an explicit escalation directive naming the fn.
        assert!(
            p.system_prompt.contains("[Human escalation]")
                && p.system_prompt.contains("`transfer_call`"),
            "system prompt must tell the model it can transfer: {:?}",
            p.system_prompt
        );
    }

    #[test]
    fn on_transfer_returns_pinned_number_else_stays() {
        let eng = Engine::new(&transfer_spec(), json!({}), Map::new()).unwrap();
        match eng.on_transfer("transfer_call") {
            Action::Transfer {
                target_phone,
                target_name,
                say,
                mode,
            } => {
                assert_eq!(target_phone, "+918760036560");
                assert_eq!(target_name.as_deref(), Some("Support"));
                assert_eq!(say.as_deref(), Some("Hold for .")); // {{name}} missing + no fallback → empty
                assert_eq!(
                    mode,
                    TransferMode::Cold,
                    "default mode is cold when omitted"
                );
            }
            other => panic!("expected Transfer, got {other:?}"),
        }
        // Unknown name → Stay (no transfer); engine does not advance.
        assert!(matches!(eng.on_transfer("nope"), Action::Stay));
        assert_eq!(
            eng.current_node(),
            "s",
            "transfer must not advance the node"
        );
    }

    #[test]
    fn transfer_mode_warm_round_trips_else_defaults_cold() {
        // `mode:"warm"` opts into warm transfer; anything else (or omitted) is cold.
        let spec = json!({
            "nodes": [
                {"id": "s", "type": "startCall", "data": {
                    "prompt": "Hi.",
                    "transfer_targets": [
                        {"name": "warm_xfer", "target_phone": "+918760036560", "mode": "warm"},
                        {"name": "cold_xfer", "target_phone": "+918760036561", "mode": "cold"},
                        {"name": "default_xfer", "target_phone": "+918760036562"}
                    ]
                }},
                {"id": "done", "type": "endCall", "data": {}}
            ],
            "edges": [{"id": "e", "source": "s", "target": "done", "label": "x"}]
        });
        let eng = Engine::new(&spec, json!({}), Map::new()).unwrap();
        let by = |name: &str| {
            eng.current_prompt()
                .transfer_targets
                .into_iter()
                .find(|t| t.name == name)
                .unwrap()
                .mode
        };
        assert_eq!(by("warm_xfer"), TransferMode::Warm);
        assert_eq!(by("cold_xfer"), TransferMode::Cold);
        assert_eq!(
            by("default_xfer"),
            TransferMode::Cold,
            "omitted mode defaults cold"
        );
        // and it threads through on_transfer
        match eng.on_transfer("warm_xfer") {
            Action::Transfer { mode, .. } => assert_eq!(mode, TransferMode::Warm),
            other => panic!("expected warm Transfer, got {other:?}"),
        }
    }

    #[test]
    fn transfer_target_phone_is_never_interpolated() {
        // Even if an operator typo'd a placeholder into target_phone, it stays
        // LITERAL — the conversation can never rewrite the dialed number.
        let spec = json!({
            "nodes": [
                {"id": "s", "type": "startCall", "data": {
                    "prompt": "Hi.",
                    "transfer_targets": [{"target_phone": "{{evil}}"}]
                }},
                {"id": "done", "type": "endCall", "data": {}}
            ],
            "edges": [{"id": "e", "source": "s", "target": "done", "label": "x"}]
        });
        let mut seed = Map::new();
        seed.insert("evil".into(), json!("+19999999999"));
        let eng = Engine::new(&spec, json!({}), seed).unwrap();
        assert_eq!(
            eng.current_prompt().transfer_targets[0].target_phone,
            "{{evil}}"
        );
    }

    #[test]
    fn empty_target_phone_entries_are_dropped() {
        let spec = json!({
            "nodes": [
                {"id": "s", "type": "startCall", "data": {
                    "prompt": "Hi.",
                    "transfer_targets": [
                        {"target_phone": "  "},
                        {"name": "Sales Line", "target_phone": "+918760036560"}
                    ]
                }},
                {"id": "done", "type": "endCall", "data": {}}
            ],
            "edges": [{"id": "e", "source": "s", "target": "done", "label": "x"}]
        });
        let eng = Engine::new(&spec, json!({}), Map::new()).unwrap();
        let targets = eng.current_prompt().transfer_targets;
        assert_eq!(targets.len(), 1, "blank-number entry dropped");
        assert_eq!(targets[0].name, "sales_line", "operator name is slugified");
        assert_eq!(targets[0].target_phone, "+918760036560");
    }
}
