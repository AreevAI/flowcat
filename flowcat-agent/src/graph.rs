// SPDX-License-Identifier: Apache-2.0
//
//! Workflow graph model. An agent is authored as a node/edge graph (e.g. in a
//! React Flow editor) and persisted as `graph_spec` JSON. This parses that JSON
//! into a validated directed graph the engine can traverse.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("graph has no start node")]
    NoStart,
    #[error("edge references unknown node: {0}")]
    DanglingEdge(String),
    #[error("malformed graph spec: {0}")]
    Malformed(String),
}

/// Node roles the engine understands. Unknown types are preserved as `Other`
/// so authoring is not blocked on engine support.
///
/// An editor authors node `type` using the catalog names
/// (`startCall`/`agentNode`/`endCall`), while hand-written graphs and the
/// engine's own tests use the bare `start`/`agent`/`end` roles.
/// The aliases accept both so an editor-built graph runs without a translation
/// layer. The catalog's `trigger`/`webhook`/`qa`/`tuner` names are already
/// snake_case and map to their own kinds (so their edge-cardinality rules can be
/// validated); only genuinely unknown types fall through to `Other`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    #[serde(alias = "startCall")]
    Start,
    #[serde(alias = "agentNode")]
    Agent,
    #[serde(alias = "endCall")]
    End,
    /// The single (optional) persona/tone node whose prompt is prepended to
    /// every node that opts in via `add_global_prompt`. Carries no edges, so it
    /// is never traversed — the engine reads its prompt when composing a turn.
    #[serde(alias = "globalNode")]
    Global,
    /// Inbound entry point (no incoming edges). Inert in the engine for now, but
    /// its cardinality is validated.
    Trigger,
    /// Side-effect HTTP node — modelled as isolated (no edges) per the catalog.
    Webhook,
    /// Post-call review/QA node — isolated.
    Qa,
    /// Agent-tuner export node — isolated.
    Tuner,
    #[serde(other)]
    Other,
}

/// Edge-cardinality bounds for a node kind (`None` = unbounded). The single
/// source for the per-node-type rules an editor can surface as a catalog.
struct EdgeBounds {
    min_in: Option<usize>,
    max_in: Option<usize>,
    min_out: Option<usize>,
    max_out: Option<usize>,
}

impl NodeKind {
    fn edge_bounds(&self) -> EdgeBounds {
        // (min_in, max_in, min_out, max_out)
        let (min_in, max_in, min_out, max_out) = match self {
            // Entry points: no incoming, outgoing unbounded.
            NodeKind::Start | NodeKind::Trigger => (None, Some(0), None, None),
            // Agent: at least one incoming; outgoing unbounded.
            NodeKind::Agent => (Some(1), None, None, None),
            // End: at least one incoming; no outgoing.
            NodeKind::End => (Some(1), None, Some(0), Some(0)),
            // Isolated nodes: no edges at all.
            NodeKind::Global | NodeKind::Webhook | NodeKind::Qa | NodeKind::Tuner => {
                (Some(0), Some(0), Some(0), Some(0))
            }
            // Unknown type — don't constrain (authoring isn't blocked).
            NodeKind::Other => (None, None, None, None),
        };
        EdgeBounds {
            min_in,
            max_in,
            min_out,
            max_out,
        }
    }

    /// Human label used in validation messages.
    fn label(&self) -> &'static str {
        match self {
            NodeKind::Start => "Start",
            NodeKind::Agent => "Agent",
            NodeKind::End => "End",
            NodeKind::Global => "Global",
            NodeKind::Trigger => "Trigger",
            NodeKind::Webhook => "Webhook",
            NodeKind::Qa => "QA",
            NodeKind::Tuner => "Tuner",
            NodeKind::Other => "Node",
        }
    }
}

/// A single graph-validation error in the shape the workflow editor consumes:
/// `kind` (`"node"`/`"edge"`/`"graph"`) + the offending `id` lets the editor
/// highlight the node/edge; `message` is the human-readable reason. Graph-level
/// errors (no start node, dangling edge) carry no `id`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValidationError {
    pub kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    #[serde(rename = "type", default = "default_kind")]
    pub kind: NodeKind,
    #[serde(default)]
    pub data: serde_json::Value,
}

fn default_kind() -> NodeKind {
    NodeKind::Other
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub id: String,
    pub source: String,
    pub target: String,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub data: serde_json::Value,
}

/// Parsed, validated workflow graph with adjacency precomputed.
#[derive(Debug, Clone)]
pub struct Graph {
    pub nodes: HashMap<String, Node>,
    pub edges: Vec<Edge>,
    pub start_id: String,
    /// The global-directive node, if the workflow has one.
    global_id: Option<String>,
    adjacency: HashMap<String, Vec<usize>>,
}

#[derive(Debug, Deserialize)]
struct RawGraph {
    #[serde(default)]
    nodes: Vec<Node>,
    #[serde(default)]
    edges: Vec<Edge>,
}

impl Graph {
    /// Parse and validate a `graph_spec` JSON document.
    pub fn parse(spec: &serde_json::Value) -> Result<Self, GraphError> {
        let raw: RawGraph = serde_json::from_value(spec.clone())
            .map_err(|e| GraphError::Malformed(e.to_string()))?;

        let mut nodes = HashMap::new();
        for n in raw.nodes {
            nodes.insert(n.id.clone(), n);
        }

        let start_id = nodes
            .values()
            .find(|n| n.kind == NodeKind::Start)
            .map(|n| n.id.clone())
            .ok_or(GraphError::NoStart)?;

        let global_id = nodes
            .values()
            .find(|n| n.kind == NodeKind::Global)
            .map(|n| n.id.clone());

        let mut adjacency: HashMap<String, Vec<usize>> = HashMap::new();
        for (idx, e) in raw.edges.iter().enumerate() {
            if !nodes.contains_key(&e.source) {
                return Err(GraphError::DanglingEdge(e.source.clone()));
            }
            if !nodes.contains_key(&e.target) {
                return Err(GraphError::DanglingEdge(e.target.clone()));
            }
            adjacency.entry(e.source.clone()).or_default().push(idx);
        }

        Ok(Self {
            nodes,
            edges: raw.edges,
            start_id,
            global_id,
            adjacency,
        })
    }

    /// Per-node-type edge-cardinality violations,
    /// driven by [`NodeKind::edge_bounds`] (the catalog rules: start/trigger take
    /// no incoming; agent needs ≥1 incoming; end needs ≥1 incoming and no
    /// outgoing; global/webhook/qa/tuner are isolated). Returns structured
    /// [`ValidationError`]s (empty ⇒ valid) so the editor can highlight the
    /// offending node. A *validation* concern, separate from `parse` (which stays
    /// lenient so an in-progress draft can still be run).
    pub fn cardinality_errors(&self) -> Vec<ValidationError> {
        let mut in_deg: HashMap<&str, usize> = HashMap::new();
        let mut out_deg: HashMap<&str, usize> = HashMap::new();
        for id in self.nodes.keys() {
            in_deg.insert(id.as_str(), 0);
            out_deg.insert(id.as_str(), 0);
        }
        for e in &self.edges {
            *out_deg.entry(e.source.as_str()).or_insert(0) += 1;
            *in_deg.entry(e.target.as_str()).or_insert(0) += 1;
        }

        let mut errors = Vec::new();
        for node in self.nodes.values() {
            let id = node.id.as_str();
            let i = in_deg.get(id).copied().unwrap_or(0);
            let o = out_deg.get(id).copied().unwrap_or(0);
            let b = node.kind.edge_bounds();
            let label = node.kind.label();

            let mut msgs: Vec<String> = Vec::new();
            if let Some(m) = b.min_in {
                if i < m {
                    msgs.push(format!("must have at least {m} incoming edge(s)"));
                }
            }
            if let Some(m) = b.max_in {
                if i > m {
                    msgs.push(if m == 0 {
                        "cannot have incoming edges".to_string()
                    } else {
                        format!("must have at most {m} incoming edge(s)")
                    });
                }
            }
            if let Some(m) = b.min_out {
                if o < m {
                    msgs.push(format!("must have at least {m} outgoing edge(s)"));
                }
            }
            if let Some(m) = b.max_out {
                if o > m {
                    msgs.push(if m == 0 {
                        "cannot have outgoing edges".to_string()
                    } else {
                        format!("must have at most {m} outgoing edge(s)")
                    });
                }
            }
            for msg in msgs {
                errors.push(ValidationError {
                    kind: "node",
                    id: Some(node.id.clone()),
                    message: format!("{label} node \"{id}\": {msg}"),
                });
            }
        }
        errors
    }

    /// The global node's raw (un-rendered) `prompt`, if the workflow has a
    /// global node carrying one. Callers render it against the run variables.
    pub fn global_prompt(&self) -> Option<&str> {
        let id = self.global_id.as_ref()?;
        self.nodes
            .get(id)
            .and_then(|n| n.data.get("prompt"))
            .and_then(|v| v.as_str())
    }

    /// Outgoing edges from a node, in declaration order.
    pub fn outgoing(&self, node_id: &str) -> Vec<&Edge> {
        self.adjacency
            .get(node_id)
            .map(|idxs| idxs.iter().map(|&i| &self.edges[i]).collect())
            .unwrap_or_default()
    }

    pub fn node(&self, id: &str) -> Option<&Node> {
        self.nodes.get(id)
    }

    /// The node's human display name for observability (`nodes_visited`), read
    /// from the editor node `data` (`name`/`label`/`title`), falling back to the
    /// raw id when the node carries no label.
    pub fn node_name(&self, id: &str) -> String {
        self.node(id)
            .and_then(|n| {
                ["name", "label", "title"]
                    .iter()
                    .find_map(|k| n.data.get(*k).and_then(|v| v.as_str()))
            })
            .map(str::to_string)
            .unwrap_or_else(|| id.to_string())
    }

    pub fn is_terminal(&self, id: &str) -> bool {
        self.node(id)
            .map(|n| n.kind == NodeKind::End)
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_minimal_graph() {
        let spec = json!({
            "nodes": [
                {"id": "s", "type": "start", "data": {}},
                {"id": "a", "type": "agent", "data": {"prompt": "hi"}},
                {"id": "e", "type": "end", "data": {}}
            ],
            "edges": [
                {"id": "e1", "source": "s", "target": "a"},
                {"id": "e2", "source": "a", "target": "e", "label": "done"}
            ]
        });
        let g = Graph::parse(&spec).expect("valid graph");
        assert_eq!(g.start_id, "s");
        assert_eq!(g.outgoing("a").len(), 1);
        assert!(g.is_terminal("e"));
    }

    #[test]
    fn parses_editor_catalog_node_types() {
        // Graph as the React Flow editor persists it (node-spec catalog names).
        let spec = json!({
            "nodes": [
                {"id": "s", "type": "startCall", "data": {}},
                {"id": "a", "type": "agentNode", "data": {"prompt": "hi"}},
                {"id": "e", "type": "endCall", "data": {}}
            ],
            "edges": [
                {"id": "e1", "source": "s", "target": "a"},
                {"id": "e2", "source": "a", "target": "e", "label": "done"}
            ]
        });
        let g = Graph::parse(&spec).expect("editor-authored graph parses");
        assert_eq!(g.start_id, "s");
        assert!(g.is_terminal("e"));
        // `globalNode`/`trigger`/`webhook`/`qa`/`tuner` now resolve to their own
        // kinds (so their cardinality is validated); a genuinely unknown type
        // still stays inert (Other) rather than failing the parse.
        assert_eq!(
            serde_json::from_value::<NodeKind>(json!("globalNode")).unwrap(),
            NodeKind::Global
        );
        assert_eq!(
            serde_json::from_value::<NodeKind>(json!("trigger")).unwrap(),
            NodeKind::Trigger
        );
        assert_eq!(
            serde_json::from_value::<NodeKind>(json!("webhook")).unwrap(),
            NodeKind::Webhook
        );
        assert_eq!(
            serde_json::from_value::<NodeKind>(json!("madeUpType")).unwrap(),
            NodeKind::Other
        );
    }

    #[test]
    fn cardinality_accepts_valid_graph_and_flags_violations() {
        // start → agent → end is well-formed.
        let ok = json!({
            "nodes": [
                {"id": "s", "type": "startCall", "data": {"prompt": "hi"}},
                {"id": "a", "type": "agentNode", "data": {}},
                {"id": "e", "type": "endCall", "data": {}}
            ],
            "edges": [
                {"id": "e1", "source": "s", "target": "a"},
                {"id": "e2", "source": "a", "target": "e"}
            ]
        });
        assert!(Graph::parse(&ok).unwrap().cardinality_errors().is_empty());

        // Violations: edge INTO start, an agent with no incoming, an edge OUT of
        // end, and a global node wired into the graph.
        let bad = json!({
            "nodes": [
                {"id": "g", "type": "globalNode", "data": {}},
                {"id": "s", "type": "startCall", "data": {}},
                {"id": "a", "type": "agentNode", "data": {}},
                {"id": "orphan", "type": "agentNode", "data": {}},
                {"id": "e", "type": "endCall", "data": {}}
            ],
            "edges": [
                {"id": "x1", "source": "a", "target": "s"}, // into start (bad)
                {"id": "x2", "source": "s", "target": "a"},
                {"id": "x3", "source": "a", "target": "e"},
                {"id": "x4", "source": "e", "target": "a"}, // out of end (bad)
                {"id": "x5", "source": "g", "target": "a"}  // global wired (bad)
            ]
        });
        let errs = Graph::parse(&bad).unwrap().cardinality_errors();
        assert!(errs
            .iter()
            .any(|e| e.message.contains("Start") && e.message.contains("incoming")));
        assert!(errs
            .iter()
            .any(|e| e.message.contains("End") && e.message.contains("outgoing")));
        assert!(errs.iter().any(|e| e.message.contains("Global")));
        assert!(errs
            .iter()
            .any(|e| e.message.contains("orphan") && e.message.contains("incoming")));
        // Every node-level error carries the offending node id + kind for editor
        // highlighting (no bare strings).
        assert!(errs.iter().all(|e| e.kind == "node" && e.id.is_some()));
    }

    #[test]
    fn cardinality_enforces_trigger_and_webhook_rules() {
        // trigger = no incoming (like start); webhook/qa/tuner = isolated.
        let bad = json!({
            "nodes": [
                {"id": "s", "type": "startCall", "data": {}},
                {"id": "t", "type": "trigger", "data": {}},
                {"id": "a", "type": "agentNode", "data": {}},
                {"id": "w", "type": "webhook", "data": {}},
                {"id": "e", "type": "endCall", "data": {}}
            ],
            "edges": [
                {"id": "x1", "source": "s", "target": "a"},
                {"id": "x2", "source": "a", "target": "t"}, // INTO trigger (bad)
                {"id": "x3", "source": "a", "target": "w"}, // INTO webhook (bad — isolated)
                {"id": "x4", "source": "a", "target": "e"}
            ]
        });
        let errs = Graph::parse(&bad).unwrap().cardinality_errors();
        assert!(
            errs.iter()
                .any(|e| e.id.as_deref() == Some("t") && e.message.contains("incoming")),
            "trigger with an incoming edge must be flagged: {errs:?}"
        );
        assert!(
            errs.iter()
                .any(|e| e.id.as_deref() == Some("w") && e.message.contains("incoming")),
            "webhook is isolated — an incoming edge must be flagged: {errs:?}"
        );

        // A trigger as a pure entry point (outgoing only) is valid.
        let ok = json!({
            "nodes": [
                {"id": "t", "type": "trigger", "data": {}},
                {"id": "s", "type": "startCall", "data": {}},
                {"id": "a", "type": "agentNode", "data": {}},
                {"id": "e", "type": "endCall", "data": {}}
            ],
            "edges": [
                {"id": "x1", "source": "t", "target": "a"},
                {"id": "x2", "source": "s", "target": "a"},
                {"id": "x3", "source": "a", "target": "e"}
            ]
        });
        assert!(Graph::parse(&ok).unwrap().cardinality_errors().is_empty());
    }

    #[test]
    fn rejects_missing_start() {
        let spec = json!({"nodes": [{"id": "a", "type": "agent"}], "edges": []});
        assert!(matches!(Graph::parse(&spec), Err(GraphError::NoStart)));
    }

    #[test]
    fn rejects_dangling_edge() {
        let spec = json!({
            "nodes": [{"id": "s", "type": "start"}],
            "edges": [{"id": "x", "source": "s", "target": "ghost"}]
        });
        assert!(matches!(
            Graph::parse(&spec),
            Err(GraphError::DanglingEdge(_))
        ));
    }
}
