// SPDX-License-Identifier: Apache-2.0
//
//! The agent/server config schema + loader.
//!
//! One [`ServerConfig`] describes a single-agent deployment: the agent graph, the
//! pipeline [`TopologyConfig`] (realtime speech-to-speech or cascaded STT→LLM→TTS),
//! the [`TransportConfig`], and the HTTP bind. Loaded from a YAML or JSON file.
//!
//! ```yaml
//! agent:
//!   graph: ./booking.json        # path to a graph_spec (or use `graph_inline`)
//!   seed_vars: { brand: Acme }
//! topology:
//!   mode: realtime               # realtime | cascaded
//!   provider: gemini
//!   model: gemini-2.0-flash
//!   # for cascaded:
//!   # stt: { provider: deepgram, model: nova-3 }
//!   # llm: { provider: openai,   model: gpt-4o }
//!   # tts: { provider: cartesia, model: <voice-id> }
//! transport: { kind: webrtc }    # webrtc | ws-plivo | sip | local
//! server: { bind: "0.0.0.0:6210" }
//! ```
//!
//! Provider **API keys are never in the file** — the host injects them into the
//! resolved [`flowcat_services::ProviderSpec`] from its own env/secret store.

use std::path::Path;

use serde::Deserialize;
use serde_json::{Map, Value};

use flowcat_services::ProviderSpec;

/// One single-agent deployment.
#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    /// The agent: its graph + seed variables.
    pub agent: AgentConfig,
    /// The pipeline shape (realtime vs cascaded) + provider selections.
    pub topology: TopologyConfig,
    /// Where calls arrive from (default `webrtc`).
    #[serde(default)]
    pub transport: TransportConfig,
    /// HTTP server settings.
    #[serde(default)]
    pub server: HttpConfig,
}

/// The agent definition: a graph spec (by path or inline) + seed variables.
#[derive(Debug, Clone, Deserialize)]
pub struct AgentConfig {
    /// Path to a `graph_spec` file (JSON or YAML), resolved relative to the config
    /// file's directory. Mutually exclusive with [`AgentConfig::graph_inline`].
    #[serde(default)]
    pub graph: Option<String>,
    /// An inline `graph_spec`. Takes precedence over [`AgentConfig::graph`].
    #[serde(default)]
    pub graph_inline: Option<Value>,
    /// Variables seeded into the run (available to `{{var}}` interpolation).
    #[serde(default)]
    pub seed_vars: Map<String, Value>,
}

/// The pipeline shape + the provider selections for a call.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "mode", rename_all = "lowercase")]
// Parsed once at startup; the size gap (Cascaded carries three ProviderSpecs) is
// irrelevant here, so boxing for it isn't worth the ergonomic cost.
#[allow(clippy::large_enum_variant)]
pub enum TopologyConfig {
    /// Realtime speech-to-speech (e.g. Gemini Live, OpenAI Realtime).
    Realtime {
        /// Realtime provider (default `gemini`).
        #[serde(default = "default_realtime_provider")]
        provider: String,
        /// Model id (empty → the connector's default).
        #[serde(default)]
        model: String,
        /// Provider-specific options (e.g. Vertex `project`/`location`, Azure `url`).
        #[serde(default)]
        options: Map<String, Value>,
    },
    /// Cascaded STT → LLM → TTS.
    Cascaded {
        /// Speech-to-text provider.
        stt: ProviderSpec,
        /// LLM provider.
        llm: ProviderSpec,
        /// Text-to-speech provider (`model` is the voice id).
        tts: ProviderSpec,
    },
}

fn default_realtime_provider() -> String {
    "gemini".to_string()
}

/// The media transport calls arrive on.
#[derive(Debug, Clone, Deserialize)]
pub struct TransportConfig {
    /// `webrtc` | `ws-plivo` | `sip` | `local` (default `webrtc`).
    #[serde(default = "default_transport_kind")]
    pub kind: String,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            kind: default_transport_kind(),
        }
    }
}

fn default_transport_kind() -> String {
    "webrtc".to_string()
}

/// HTTP server settings.
#[derive(Debug, Clone, Deserialize)]
pub struct HttpConfig {
    /// Bind address (default `0.0.0.0:6210`).
    #[serde(default = "default_bind")]
    pub bind: String,
}

impl Default for HttpConfig {
    fn default() -> Self {
        Self {
            bind: default_bind(),
        }
    }
}

fn default_bind() -> String {
    "0.0.0.0:6210".to_string()
}

/// A config load/parse error.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// Failed to read a file (the config itself or a referenced graph).
    #[error("read {path}: {source}")]
    Read {
        /// The path that failed to read.
        path: String,
        /// The underlying I/O error.
        source: std::io::Error,
    },
    /// Failed to parse the config or graph document.
    #[error("parse {path}: {msg}")]
    Parse {
        /// The path that failed to parse.
        path: String,
        /// The parser's message.
        msg: String,
    },
    /// The agent graph is not specified or is invalid.
    #[error("agent graph: {0}")]
    Graph(String),
}

impl ServerConfig {
    /// Load + parse a config file. `.yaml`/`.yml` → YAML, anything else → JSON.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
            path: path.display().to_string(),
            source,
        })?;
        let is_yaml = matches!(
            path.extension().and_then(|e| e.to_str()),
            Some("yaml") | Some("yml")
        );
        Self::parse(&text, is_yaml).map_err(|msg| ConfigError::Parse {
            path: path.display().to_string(),
            msg,
        })
    }

    /// Parse config text (YAML when `is_yaml`, else JSON).
    pub fn parse(text: &str, is_yaml: bool) -> Result<Self, String> {
        if is_yaml {
            serde_yaml::from_str(text).map_err(|e| e.to_string())
        } else {
            serde_json::from_str(text).map_err(|e| e.to_string())
        }
    }

    /// Resolve the agent's `graph_spec` to a JSON value: `graph_inline` if present,
    /// otherwise the `graph` file path read relative to `base_dir`.
    pub fn resolve_graph(&self, base_dir: &Path) -> Result<Value, ConfigError> {
        if let Some(inline) = &self.agent.graph_inline {
            return Ok(inline.clone());
        }
        let rel = self.agent.graph.as_deref().ok_or_else(|| {
            ConfigError::Graph("set `agent.graph` (a path) or `agent.graph_inline`".to_string())
        })?;
        let p = base_dir.join(rel);
        let text = std::fs::read_to_string(&p).map_err(|source| ConfigError::Read {
            path: p.display().to_string(),
            source,
        })?;
        let is_yaml = matches!(
            p.extension().and_then(|e| e.to_str()),
            Some("yaml") | Some("yml")
        );
        if is_yaml {
            serde_yaml::from_str(&text).map_err(|e| ConfigError::Graph(e.to_string()))
        } else {
            serde_json::from_str(&text).map_err(|e| ConfigError::Graph(e.to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_realtime_yaml_with_defaults() {
        let cfg = ServerConfig::parse(
            r#"
agent:
  graph_inline: { nodes: [], edges: [] }
  seed_vars: { brand: Acme }
topology:
  mode: realtime
"#,
            true,
        )
        .expect("valid realtime config");
        match cfg.topology {
            TopologyConfig::Realtime {
                provider, model, ..
            } => {
                assert_eq!(provider, "gemini", "realtime provider defaults to gemini");
                assert!(model.is_empty());
            }
            _ => panic!("expected realtime"),
        }
        // transport + server fall back to defaults.
        assert_eq!(cfg.transport.kind, "webrtc");
        assert_eq!(cfg.server.bind, "0.0.0.0:6210");
        assert_eq!(cfg.agent.seed_vars.get("brand").unwrap(), "Acme");
    }

    #[test]
    fn parses_cascaded_yaml_provider_specs() {
        let cfg = ServerConfig::parse(
            r#"
agent: { graph: ./booking.json }
topology:
  mode: cascaded
  stt: { provider: deepgram, model: nova-3 }
  llm: { provider: openai, model: gpt-4o }
  tts: { provider: cartesia, model: voice-xyz }
transport: { kind: ws-plivo }
server: { bind: "127.0.0.1:7000" }
"#,
            true,
        )
        .expect("valid cascaded config");
        match cfg.topology {
            TopologyConfig::Cascaded { stt, llm, tts } => {
                assert_eq!(stt.provider, "deepgram");
                assert_eq!(stt.model, "nova-3");
                assert_eq!(llm.provider, "openai");
                assert_eq!(llm.model, "gpt-4o");
                assert_eq!(tts.provider, "cartesia");
                assert_eq!(tts.model, "voice-xyz");
                // keys are never in the file.
                assert!(stt.api_key.is_empty() && llm.api_key.is_empty() && tts.api_key.is_empty());
            }
            _ => panic!("expected cascaded"),
        }
        assert_eq!(cfg.transport.kind, "ws-plivo");
        assert_eq!(cfg.server.bind, "127.0.0.1:7000");
    }

    #[test]
    fn parses_json_too() {
        let cfg = ServerConfig::parse(
            r#"{ "agent": { "graph_inline": {"nodes":[],"edges":[]} },
                 "topology": { "mode": "realtime", "provider": "openai" } }"#,
            false,
        )
        .expect("valid JSON config");
        match cfg.topology {
            TopologyConfig::Realtime { provider, .. } => assert_eq!(provider, "openai"),
            _ => panic!("expected realtime"),
        }
    }

    #[test]
    fn resolve_graph_prefers_inline() {
        let cfg = ServerConfig::parse(
            r#"{ "agent": { "graph_inline": {"nodes":[{"id":"s"}],"edges":[]} },
                 "topology": { "mode": "realtime" } }"#,
            false,
        )
        .unwrap();
        let g = cfg
            .resolve_graph(Path::new("."))
            .expect("inline graph resolves");
        assert_eq!(g["nodes"][0]["id"], "s");
    }

    #[test]
    fn resolve_graph_errors_when_neither_source_set() {
        let cfg = ServerConfig::parse(
            r#"{ "agent": {}, "topology": { "mode": "realtime" } }"#,
            false,
        )
        .unwrap();
        let err = cfg.resolve_graph(Path::new(".")).unwrap_err();
        assert!(matches!(err, ConfigError::Graph(_)));
    }
}
