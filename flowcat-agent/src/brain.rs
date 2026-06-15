// SPDX-License-Identifier: Apache-2.0
//
//! [`DeclarativeBrain`] — a `flowcat_core::AgentBrain` backed by the graph [`Engine`].
//!
//! Drives one run's [`Engine`] directly through the `AgentBrain` seam: each graph
//! transition is exposed as a no-arg tool the model can call to take that edge,
//! plus an always-present `endCall` tool to hang up. This lets a config-driven
//! host run a declarative agent without implementing `AgentBrain` by hand.

use flowcat_core::{AgentBrain, BrainAction, ToolDecl};
use serde_json::{json, Map, Value};

use crate::engine::{Action, Engine, EngineError};

/// The tool name the model calls to end the call. Distinct from the engine's
/// transitions — `endCall` is offered on every node so the agent can always hang
/// up, even from a node with no terminal edge.
pub const END_CALL_TOOL: &str = "endCall";

/// A flowcat-core brain that drives one run's declarative [`Engine`].
///
/// Construct it from the opaque `brain_config` a `SessionSource` returns
/// ([`DeclarativeBrain::from_config`]) or directly from a graph spec
/// ([`DeclarativeBrain::new`]).
pub struct DeclarativeBrain {
    engine: Engine,
}

impl DeclarativeBrain {
    /// Build from a `brain_config` JSON object of the shape
    /// `{ "graph_spec": {…}, "runtime_options": {…}, "seed_vars": {…} }` — the
    /// opaque `ResolvedCall::brain_config` a `SessionSource` returns. Only
    /// `graph_spec` is required; the others default to empty.
    pub fn from_config(brain_config: &Value) -> Result<Self, EngineError> {
        let graph_spec = brain_config
            .get("graph_spec")
            .cloned()
            .unwrap_or_else(|| json!({}));
        let runtime_options = brain_config
            .get("runtime_options")
            .cloned()
            .unwrap_or_else(|| json!({}));
        let seed_vars = brain_config
            .get("seed_vars")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_else(Map::new);
        Self::new(&graph_spec, runtime_options, seed_vars)
    }

    /// Build directly from a graph spec, runtime options and seed variables.
    pub fn new(
        graph_spec: &Value,
        runtime_options: Value,
        seed_vars: Map<String, Value>,
    ) -> Result<Self, EngineError> {
        Ok(Self {
            engine: Engine::new(graph_spec, runtime_options, seed_vars)?,
        })
    }

    /// The current node's tool set: one no-arg tool per transition, plus the
    /// always-present `endCall` tool. Factored out so [`AgentBrain::tools`] and
    /// the transition path build an identical set.
    fn current_tools(&self) -> Vec<ToolDecl> {
        let prompt = self.engine.current_prompt();
        let mut tools: Vec<ToolDecl> = prompt
            .transitions
            .into_iter()
            .map(|t| ToolDecl {
                description: t
                    .description
                    .filter(|d| !d.trim().is_empty())
                    .unwrap_or_else(|| format!("Take the '{}' transition.", t.name)),
                name: t.name,
                // Transitions are no-arg: the model only chooses *which* edge.
                params: json!({ "type": "object", "properties": {} }),
            })
            .collect();
        tools.push(end_call_tool());
        tools
    }
}

/// The `endCall` tool declaration: a single optional `disposition` string the
/// model may set to label the outcome (folded into collected vars by the pipeline).
fn end_call_tool() -> ToolDecl {
    ToolDecl {
        name: END_CALL_TOOL.to_string(),
        description: "Hang up ONLY when the caller clearly asks to end (e.g. 'goodbye', \
                      'that's all'). Do NOT end the call otherwise — keep talking. \
                      Optional `disposition` label for the outcome (e.g. 'completed')."
            .to_string(),
        params: json!({
            "type": "object",
            "properties": {
                "disposition": {
                    "type": "string",
                    "description": "Optional short outcome label for the call."
                }
            }
        }),
    }
}

impl AgentBrain for DeclarativeBrain {
    fn system_prompt(&self) -> String {
        self.engine.current_prompt().system_prompt
    }

    fn tools(&self) -> Vec<ToolDecl> {
        self.current_tools()
    }

    fn current_node_id(&self) -> String {
        // The pipeline scopes the node's MCP/HTTP tool lookup to this id.
        self.engine.current_node().to_string()
    }

    fn on_tool_call(&mut self, name: &str, args: &Value) -> BrainAction {
        // `endCall` is handled directly (it is not an engine edge): pull the
        // optional disposition and end the call.
        if name == END_CALL_TOOL {
            let disposition = args
                .get("disposition")
                .and_then(Value::as_str)
                .map(str::to_string);
            return BrainAction::End { disposition };
        }

        // Otherwise it is an engine transition. The engine owns the decision; we
        // translate its `Action` into the flowcat `BrainAction` the pipeline applies.
        match self.engine.on_transition(name) {
            Action::Transition { say, .. } => {
                // The engine advanced — recompute the destination node's prompt +
                // tool set so the model is re-armed for the new state.
                let prompt = self.engine.current_prompt();
                BrainAction::Transition {
                    system_prompt: prompt.system_prompt,
                    tools: self.current_tools(),
                    say,
                }
            }
            // Landing on a terminal node ends the call.
            Action::End => BrainAction::End { disposition: None },
            // Human escalation. flowcat-core has no transfer-capable `BrainAction`
            // (it is the carrier-agnostic runtime), so an engine `Transfer` ends
            // the session with a `transferred` disposition — the safe outcome (no
            // stranded caller); a host that supports dial-out can special-case it.
            Action::Transfer { .. } => BrainAction::End {
                disposition: Some("transferred".to_string()),
            },
            // An unknown transition name keeps the current node.
            Action::Stay => BrainAction::Stay,
        }
    }

    fn is_finished(&self) -> bool {
        self.engine.is_finished()
    }

    fn collected_vars(&self) -> Value {
        // `snapshot()` → `{current, vars, runtime_options}`; take the harvested
        // variables and fold in `nodes_visited` (the ordered list of nodes this
        // run traversed) for observability.
        let mut map = match self.engine.snapshot().get("vars").cloned() {
            Some(Value::Object(m)) => m,
            _ => serde_json::Map::new(),
        };
        map.insert(
            "nodes_visited".to_string(),
            Value::from(self.engine.visited_nodes().to_vec()),
        );
        Value::Object(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A two-node graph: start (with a prompt + one labelled edge) → end.
    fn two_node_graph() -> Value {
        json!({
            "graph_spec": {
                "nodes": [
                    {"id": "s", "type": "startCall", "data": {
                        "prompt": "Hi {{name}}, how can I help?"
                    }},
                    {"id": "done", "type": "endCall", "data": {"prompt": "Bye."}}
                ],
                "edges": [
                    {"id": "e", "source": "s", "target": "done", "label": "Wrap up",
                     "data": {"condition": "The caller is done"}}
                ]
            },
            "runtime_options": {},
            "seed_vars": {"name": "Sam"}
        })
    }

    #[test]
    fn new_seeds_vars_and_renders_prompt() {
        let brain = DeclarativeBrain::from_config(&two_node_graph()).expect("engine builds");
        assert_eq!(brain.system_prompt(), "Hi Sam, how can I help?");
        assert!(!brain.is_finished());
    }

    #[test]
    fn tools_are_transitions_plus_end_call() {
        let brain = DeclarativeBrain::from_config(&two_node_graph()).expect("engine builds");
        let tools = brain.tools();
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        // The slugified edge label + the always-present endCall.
        assert!(
            names.contains(&"wrap_up"),
            "transition tool present: {names:?}"
        );
        assert!(names.contains(&END_CALL_TOOL), "endCall present: {names:?}");
        assert_eq!(tools.len(), 2, "exactly the transition + endCall");

        // Each transition tool is a no-arg object schema.
        let wrap = tools.iter().find(|t| t.name == "wrap_up").unwrap();
        assert_eq!(wrap.params, json!({ "type": "object", "properties": {} }));
        // Its description comes from the edge condition.
        assert_eq!(wrap.description, "The caller is done");

        // The endCall tool exposes an optional string `disposition` property.
        let end = tools.iter().find(|t| t.name == END_CALL_TOOL).unwrap();
        assert_eq!(end.params["type"], "object");
        assert_eq!(end.params["properties"]["disposition"]["type"], "string");
    }

    #[test]
    fn end_call_maps_to_end_with_disposition() {
        let mut brain = DeclarativeBrain::from_config(&two_node_graph()).expect("engine builds");
        match brain.on_tool_call(END_CALL_TOOL, &json!({ "disposition": "completed" })) {
            BrainAction::End { disposition } => {
                assert_eq!(disposition.as_deref(), Some("completed"))
            }
            other => panic!("expected End, got {other:?}"),
        }
        // endCall with no args → End with no disposition.
        match brain.on_tool_call(END_CALL_TOOL, &json!({})) {
            BrainAction::End { disposition } => assert!(disposition.is_none()),
            other => panic!("expected End, got {other:?}"),
        }
    }

    #[test]
    fn transition_to_terminal_node_ends_the_call() {
        // The only edge lands on an `endCall` node, so taking it is terminal →
        // the engine returns `End`, which the brain maps to `BrainAction::End`.
        let mut brain = DeclarativeBrain::from_config(&two_node_graph()).expect("engine builds");
        match brain.on_tool_call("wrap_up", &json!({})) {
            BrainAction::End { disposition } => assert!(disposition.is_none()),
            other => panic!("expected End on terminal transition, got {other:?}"),
        }
        assert!(brain.is_finished());
    }

    #[test]
    fn transition_to_non_terminal_node_re_arms_prompt_and_tools() {
        // start → middle (non-terminal, its own prompt + edge) → end.
        let cfg = json!({
            "graph_spec": {
                "nodes": [
                    {"id": "s", "type": "startCall", "data": {"prompt": "Start."}},
                    {"id": "mid", "type": "agentNode", "data": {"prompt": "Now collecting details."}},
                    {"id": "done", "type": "endCall", "data": {}}
                ],
                "edges": [
                    {"id": "e1", "source": "s", "target": "mid", "label": "Proceed"},
                    {"id": "e2", "source": "mid", "target": "done", "label": "Finish"}
                ]
            }
        });
        let mut brain = DeclarativeBrain::from_config(&cfg).expect("engine builds");
        match brain.on_tool_call("proceed", &json!({})) {
            BrainAction::Transition {
                system_prompt,
                tools,
                ..
            } => {
                assert_eq!(system_prompt, "Now collecting details.");
                let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
                // The destination node's edge (`finish`) + endCall.
                assert!(
                    names.contains(&"finish"),
                    "re-armed with new tools: {names:?}"
                );
                assert!(names.contains(&END_CALL_TOOL));
            }
            other => panic!("expected Transition, got {other:?}"),
        }
        assert!(!brain.is_finished());
    }

    #[test]
    fn unknown_tool_name_stays() {
        let mut brain = DeclarativeBrain::from_config(&two_node_graph()).expect("engine builds");
        assert!(matches!(
            brain.on_tool_call("does_not_exist", &json!({})),
            BrainAction::Stay
        ));
    }

    #[test]
    fn collected_vars_returns_seed_vars_object_with_nodes_visited() {
        let brain = DeclarativeBrain::from_config(&two_node_graph()).expect("engine builds");
        let vars = brain.collected_vars();
        assert_eq!(vars["name"], "Sam");
        // nodes_visited is folded in (at least the active node on construction).
        assert!(
            vars["nodes_visited"].is_array(),
            "nodes_visited folded in: {vars}"
        );
        assert!(
            !vars["nodes_visited"].as_array().unwrap().is_empty(),
            "active node recorded: {vars}"
        );
    }

    #[test]
    fn current_node_id_tracks_the_engine_state() {
        // start → middle → end; the brain's current_node_id follows the engine.
        let cfg = json!({
            "graph_spec": {
                "nodes": [
                    {"id": "s", "type": "startCall", "data": {"prompt": "Start."}},
                    {"id": "mid", "type": "agentNode", "data": {"prompt": "Mid."}},
                    {"id": "done", "type": "endCall", "data": {}}
                ],
                "edges": [
                    {"id": "e1", "source": "s", "target": "mid", "label": "Proceed"},
                    {"id": "e2", "source": "mid", "target": "done", "label": "Finish"}
                ]
            }
        });
        let mut brain = DeclarativeBrain::from_config(&cfg).expect("engine builds");
        assert_eq!(brain.current_node_id(), "s");
        // Take the first edge → the brain now reports the destination node.
        let _ = brain.on_tool_call("proceed", &json!({}));
        assert_eq!(brain.current_node_id(), "mid");
    }

    #[test]
    fn end_call_tool_description_discourages_premature_hangup() {
        // Guards against a permissive description that lets the model hang up
        // mid-conversation (e.g. right after a transition).
        let desc = end_call_tool().description.to_lowercase();
        assert!(
            desc.contains("only"),
            "must restrict when endCall is allowed: {desc}"
        );
        assert!(
            desc.contains("do not end the call"),
            "must explicitly tell the model NOT to end prematurely: {desc}"
        );
    }
}
