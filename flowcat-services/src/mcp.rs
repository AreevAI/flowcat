// SPDX-License-Identifier: Apache-2.0
//
//! MCP-as-processor (behind the `mcp` feature).
//!
//! Bridges the agent's function/tool calls to an [MCP] (Model Context Protocol)
//! server: list the server's tools, expose them as function-callable, call a tool
//! when the model asks, and feed the result back as a frame. Port of pipecat's
//! `MCPClient` (`pipecat/src/pipecat/services/mcp_service.py`) into the flowcat
//! [`FrameProcessor`](flowcat_core::FrameProcessor) model.
//!
//! ## Why this lives in `flowcat-services`, not core
//!
//! An MCP client speaks JSON-RPC over a transport (stdio / streamable HTTP). The
//! HTTP transport pulls `reqwest`, which must stay out of dependency-light
//! flowcat-core — so MCP-as-processor lives here behind the `mcp` feature.
//!
//! ## Mockable transport
//!
//! The wire is abstracted behind the [`McpTransport`] trait (a single
//! `request(method, params) -> result` JSON-RPC call). [`HttpMcpTransport`] is the
//! real streamable-HTTP impl (feature-gated on `reqwest`); tests inject a mock
//! transport so tool-list discovery and a tool-call round-trip are verified with
//! **no real server**.
//!
//! ## Security note (for the reviewer)
//!
//! - The MCP server is a **remote tool executor**. [`McpProcessor`] guards it with
//!   an optional allow-list (`tools_filter`) so only sanctioned tools are exposed
//!   and callable — a tool name the model emits that is not in the discovered +
//!   allowed set is rejected without a network call.
//! - This module performs **no auth itself**: bearer/headers are the caller's job,
//!   threaded through [`HttpMcpTransport::with_header`]. Secrets must never be
//!   logged; this module logs tool *names*, never argument values or headers.
//! - The result content returned by a tool is passed back verbatim to the model
//!   as a [`Frame::FunctionCallResult`]; callers that need output redaction should
//!   wrap a filter around the result (mirrors pipecat's per-tool output filter).
//! - **SSRF guard:** the server URL is caller-influenceable, so
//!   [`HttpMcpTransport`] is **safe-by-construction** — on the first request it
//!   rejects a non-http(s) scheme, a `localhost`/private/CGNAT/link-local/metadata/
//!   ULA host, and **pins** the connection to the vetted public IP (defeating
//!   DNS-rebind TOCTOU). The guard is on by default; an explicit
//!   [`HttpMcpTransport::allow_private_url`] opt-out exists for trusted same-host /
//!   on-prem deployments. This follows the standard control-plane `ssrf_check`
//!   egress discipline (std-net only) so the OSS primitive never ships an SSRF
//!   hole regardless of consumer.
//!
//! [MCP]: https://modelcontextprotocol.io

use std::collections::HashSet;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::OnceCell;

use flowcat_core::processor::frame::{FunctionCall, FunctionCallResult};
use flowcat_core::processor::{Envelope, Link};
use flowcat_core::{Frame, FrameProcessor, Result, StartParams, ToolDecl};

/// Errors from the MCP layer.
#[derive(Debug, thiserror::Error)]
pub enum McpError {
    /// The transport (HTTP/stdio) failed.
    #[error("mcp transport error: {0}")]
    Transport(String),
    /// The server returned a JSON-RPC error object.
    #[error("mcp server error {code}: {message}")]
    Server {
        /// JSON-RPC error code.
        code: i64,
        /// JSON-RPC error message.
        message: String,
    },
    /// The server response did not match the expected shape.
    #[error("mcp protocol error: {0}")]
    Protocol(String),
}

impl From<McpError> for flowcat_core::FlowcatError {
    fn from(e: McpError) -> Self {
        flowcat_core::FlowcatError::Network(e.to_string())
    }
}

/// A JSON-RPC transport to an MCP server. One method: issue a request and get the
/// `result` value back (or an [`McpError`]). Abstracted so the processor is testable
/// against a mock — the real wire (HTTP/stdio) is an implementation detail.
#[async_trait]
pub trait McpTransport: Send + Sync {
    /// Issue a JSON-RPC `method` with `params`, returning the `result` value.
    async fn request(&self, method: &str, params: Value) -> std::result::Result<Value, McpError>;
}

/// One MCP tool as discovered from `tools/list`.
#[derive(Debug, Clone, PartialEq)]
pub struct McpTool {
    /// Tool name (the function name the model calls).
    pub name: String,
    /// Human/LLM-facing description.
    pub description: String,
    /// JSON-Schema for the tool's input (`inputSchema`).
    pub input_schema: Value,
}

impl McpTool {
    /// Convert to a flowcat [`ToolDecl`] (what the engine/model is told about).
    /// Mirrors pipecat `_convert_mcp_schema_to_pipecat`.
    pub fn to_tool_decl(&self) -> ToolDecl {
        ToolDecl {
            name: self.name.clone(),
            description: self.description.clone(),
            params: self.input_schema.clone(),
        }
    }
}

/// An MCP client over a pluggable [`McpTransport`]. Discovers tools and calls them;
/// transport-agnostic and fully testable with a mock transport.
pub struct McpClient {
    transport: Arc<dyn McpTransport>,
    /// Optional allow-list: when set, only these tool names are exposed/callable.
    tools_filter: Option<HashSet<String>>,
}

impl McpClient {
    /// A client over `transport`, exposing every tool the server advertises.
    pub fn new(transport: Arc<dyn McpTransport>) -> Self {
        Self {
            transport,
            tools_filter: None,
        }
    }

    /// Restrict the exposed/callable tools to `allowed` (an allow-list). A tool not
    /// in this set is neither advertised to the model nor dispatched if requested.
    pub fn with_tools_filter(mut self, allowed: impl IntoIterator<Item = String>) -> Self {
        self.tools_filter = Some(allowed.into_iter().collect());
        self
    }

    fn is_allowed(&self, name: &str) -> bool {
        self.tools_filter
            .as_ref()
            .is_none_or(|set| set.contains(name))
    }

    /// List the server's tools (`tools/list`), filtered by the allow-list.
    pub async fn list_tools(&self) -> std::result::Result<Vec<McpTool>, McpError> {
        let result = self.transport.request("tools/list", json!({})).await?;
        let tools = result
            .get("tools")
            .and_then(Value::as_array)
            .ok_or_else(|| McpError::Protocol("tools/list: missing `tools` array".into()))?;
        let mut out = Vec::with_capacity(tools.len());
        for t in tools {
            let name = t
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| McpError::Protocol("tool entry missing `name`".into()))?
                .to_string();
            if !self.is_allowed(&name) {
                continue;
            }
            let description = t
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let input_schema = t
                .get("inputSchema")
                .cloned()
                .unwrap_or_else(|| json!({"type": "object", "properties": {}}));
            out.push(McpTool {
                name,
                description,
                input_schema,
            });
        }
        Ok(out)
    }

    /// Call a tool (`tools/call`) and return its concatenated text content. Mirrors
    /// pipecat `_call_tool`: joins the `content[].text` chunks; a missing/empty
    /// result yields a polite fallback string so the model always gets *something*.
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: Value,
    ) -> std::result::Result<String, McpError> {
        if !self.is_allowed(name) {
            return Err(McpError::Protocol(format!(
                "tool `{name}` is not in the allow-list"
            )));
        }
        let result = self
            .transport
            .request(
                "tools/call",
                json!({ "name": name, "arguments": arguments }),
            )
            .await?;

        // MCP tools return `{ "content": [ { "type": "text", "text": "…" }, … ] }`.
        let mut response = String::new();
        if let Some(content) = result.get("content").and_then(Value::as_array) {
            for chunk in content {
                if let Some(text) = chunk.get("text").and_then(Value::as_str) {
                    response.push_str(text);
                }
            }
        }
        if response.is_empty() {
            response = "Sorry, could not call the mcp tool".to_string();
        }
        Ok(response)
    }
}

/// A [`FrameProcessor`] that bridges function-call frames to an MCP server.
///
/// - On [`Frame::Start`] it discovers the server's tools (`tools/list`); the
///   discovered set is what it will dispatch (an unknown tool is ignored).
/// - On [`Frame::FunctionCallsStarted`] it calls each requested tool whose name is
///   in the discovered+allowed set and emits a [`Frame::FunctionCallResult`] per
///   call downstream (fed back to the LLM). Unknown tools are passed through
///   untouched so a different processor can handle them.
/// - Every other frame is forwarded unchanged.
pub struct McpProcessor {
    name: &'static str,
    client: McpClient,
    /// Tool names discovered at `start` (the dispatchable set).
    known_tools: HashSet<String>,
}

impl McpProcessor {
    /// Wrap `client` as a processor.
    pub fn new(client: McpClient) -> Self {
        Self {
            name: "mcp",
            client,
            known_tools: HashSet::new(),
        }
    }

    /// The tools discovered at `start` (empty before start). Useful for the brain
    /// to merge MCP tools into the node's tool set.
    pub fn known_tools(&self) -> &HashSet<String> {
        &self.known_tools
    }

    /// Dispatch one call, producing its [`FunctionCallResult`]. Separated out for
    /// direct unit testing without the frame loop.
    pub async fn dispatch(&self, call: &FunctionCall) -> FunctionCallResult {
        let result_text = match self
            .client
            .call_tool(&call.function_name, call.arguments.clone())
            .await
        {
            Ok(text) => Value::String(text),
            Err(e) => json!({ "error": e.to_string() }),
        };
        FunctionCallResult {
            function_name: call.function_name.clone(),
            tool_call_id: call.tool_call_id.clone(),
            result: result_text,
        }
    }
}

#[async_trait]
impl FrameProcessor for McpProcessor {
    fn name(&self) -> &str {
        self.name
    }

    async fn start(
        &mut self,
        _setup: &flowcat_core::ProcessorSetup,
        _params: &StartParams,
    ) -> Result<()> {
        // Discover tools once; a failure is non-fatal (no MCP tools available).
        match self.client.list_tools().await {
            Ok(tools) => {
                self.known_tools = tools.into_iter().map(|t| t.name).collect();
            }
            Err(e) => {
                tracing::warn!(error = %e, "MCP tools/list failed; no MCP tools available");
            }
        }
        Ok(())
    }

    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        let Frame::FunctionCallsStarted(ref calls) = env.frame else {
            link.push(env.meta, env.frame, env.direction).await;
            return Ok(());
        };

        // Partition into MCP-dispatchable calls and the rest (forwarded onward).
        let mut passthrough: Vec<FunctionCall> = Vec::new();
        let mut results: Vec<FunctionCallResult> = Vec::new();
        for call in calls {
            if self.known_tools.contains(&call.function_name) {
                results.push(self.dispatch(call).await);
            } else {
                passthrough.push(call.clone());
            }
        }

        // Forward any non-MCP calls so another processor can handle them.
        if !passthrough.is_empty() {
            link.push(
                env.meta,
                Frame::FunctionCallsStarted(passthrough),
                env.direction,
            )
            .await;
        }
        // Emit each MCP tool result downstream (uninterruptible by frame class).
        for r in results {
            link.push_down(Frame::FunctionCallResult(r)).await;
        }
        Ok(())
    }
}

// ===========================================================================
// Real streamable-HTTP transport (feature `reqwest`, always on with `mcp`).
// ===========================================================================

/// A JSON-RPC-over-HTTP MCP transport (MCP "streamable HTTP"): each request is a
/// JSON-RPC POST to the server URL; the JSON response carries `result` or `error`.
/// Auth headers (bearer, etc.) are the caller's responsibility — set them with
/// [`HttpMcpTransport::with_header`]; this transport never logs header values.
///
/// **SSRF-safe by construction.** Because the URL is caller-influenceable, the
/// first request runs an [`ssrf_check`] (scheme + host + resolved-IP vetting) and
/// **pins** the HTTP client's connection to the vetted public IP — so a host that
/// resolves to (or later DNS-rebinds to) a loopback/private/CGNAT/link-local/
/// metadata/ULA address is rejected. The guard is on by default; call
/// [`allow_private_url`](HttpMcpTransport::allow_private_url) to opt out for a
/// trusted on-prem/same-host MCP server.
pub struct HttpMcpTransport {
    url: String,
    headers: Vec<(String, String)>,
    next_id: std::sync::atomic::AtomicI64,
    /// Whether the SSRF guard is enforced (default `true`). The explicit
    /// `allow_private_url` opt-out flips it off for trusted deployments.
    require_public_url: bool,
    /// The vetted, IP-pinned client, built lazily on the first request (after the
    /// SSRF check). `OnceCell` so the (async, DNS-resolving) build happens once and
    /// a failed check is surfaced per-request without caching a bad client.
    client: OnceCell<reqwest::Client>,
}

impl HttpMcpTransport {
    /// A transport posting JSON-RPC to `url`. The SSRF guard is **on by default**:
    /// the first request vets `url`'s scheme + host + resolved IPs and pins the
    /// connection to a vetted public IP.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            headers: Vec::new(),
            next_id: std::sync::atomic::AtomicI64::new(1),
            require_public_url: true,
            client: OnceCell::new(),
        }
    }

    /// Add a request header (e.g. `Authorization: Bearer …`). Values are never
    /// logged.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// **Explicitly** disable the public-URL SSRF guard so the transport may reach a
    /// private/loopback/on-prem MCP server. Off-by-default-secure: a caller must opt
    /// in to private egress, and the choice is visible at the call site.
    pub fn allow_private_url(mut self) -> Self {
        self.require_public_url = false;
        self
    }

    /// Build (once) the HTTP client used for every request. When the guard is on,
    /// runs [`ssrf_check`] and pins the connection to the vetted public IP(s)
    /// (defeating DNS-rebind between the check and the connect). When the guard is
    /// off (explicit opt-out), returns a plain client with no pinning.
    async fn client(&self) -> std::result::Result<&reqwest::Client, McpError> {
        self.client
            .get_or_try_init(|| async {
                if !self.require_public_url {
                    return reqwest::Client::builder()
                        .build()
                        .map_err(|e| McpError::Transport(e.to_string()));
                }
                let url = reqwest::Url::parse(&self.url)
                    .map_err(|e| McpError::Transport(format!("invalid MCP url: {e}")))?;
                let vetted = ssrf_check(&url).await.map_err(McpError::Transport)?;
                // Pin the resolved host → vetted public IPs so the connect cannot be
                // re-resolved to a blocked address (DNS-rebind TOCTOU).
                let mut builder = reqwest::Client::builder();
                if let Some(host) = url.host_str() {
                    builder = builder.resolve_to_addrs(host, &vetted);
                }
                builder
                    .build()
                    .map_err(|e| McpError::Transport(e.to_string()))
            })
            .await
    }
}

#[async_trait]
impl McpTransport for HttpMcpTransport {
    async fn request(&self, method: &str, params: Value) -> std::result::Result<Value, McpError> {
        let client = self.client().await?;
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let body = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        let mut req = client
            .post(&self.url)
            .header("content-type", "application/json")
            .header("accept", "application/json")
            .json(&body);
        for (k, v) in &self.headers {
            req = req.header(k.as_str(), v.as_str());
        }
        let resp = req
            .send()
            .await
            .map_err(|e| McpError::Transport(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(McpError::Transport(format!("HTTP {}", resp.status())));
        }
        let envelope: Value = resp
            .json()
            .await
            .map_err(|e| McpError::Protocol(e.to_string()))?;
        if let Some(err) = envelope.get("error") {
            let code = err.get("code").and_then(Value::as_i64).unwrap_or(0);
            let message = err
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string();
            return Err(McpError::Server { code, message });
        }
        envelope
            .get("result")
            .cloned()
            .ok_or_else(|| McpError::Protocol("response missing `result`".into()))
    }
}

// ===========================================================================
// SSRF guard — follows a standard `ssrf_check` / `resolve_to_addrs` egress
// discipline so the OSS HTTP MCP transport is safe-by-construction. std-net +
// tokio (already pulled by reqwest) only — no new dependency.
// ===========================================================================

/// Reject an outbound URL that isn't a public http(s) endpoint: only http/https,
/// never `localhost`, and every resolved IP must be a public address (blocks
/// loopback, RFC-1918 private, CGNAT/`100.64.0.0/10`, link-local incl. the
/// `169.254.169.254` cloud-metadata endpoint, ULA, unspecified, multicast,
/// broadcast). Returns the **vetted** socket addresses so the caller can pin
/// reqwest's connection to them — closing the DNS-rebinding TOCTOU (no
/// re-resolution between the check and the connect).
///
/// Self-contained (not imported from any control-plane crate) so flowcat-services
/// stays a standalone OSS crate that never ships an SSRF hole in its HTTP MCP
/// transport.
pub async fn ssrf_check(url: &reqwest::Url) -> std::result::Result<Vec<SocketAddr>, String> {
    match url.scheme() {
        "http" | "https" => {}
        s => return Err(format!("scheme '{s}' not allowed")),
    }
    let host = url.host_str().ok_or("missing host")?;
    let lower = host.to_ascii_lowercase();
    if lower == "localhost" || lower.ends_with(".localhost") {
        return Err("localhost is not allowed".into());
    }
    let port = url.port_or_known_default().unwrap_or(443);
    let vetted: Vec<SocketAddr> = tokio::net::lookup_host((host, port))
        .await
        .map_err(|e| format!("could not resolve host: {e}"))?
        .collect();
    if vetted.is_empty() {
        return Err("host did not resolve".into());
    }
    for addr in &vetted {
        if is_blocked_ip(&addr.ip()) {
            return Err(format!("resolves to a non-public address ({})", addr.ip()));
        }
    }
    Ok(vetted)
}

/// True if an IP must not be reached by a server-driven request (non-public ranges).
fn is_blocked_ip(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let o = v4.octets();
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local() // 169.254.0.0/16 — incl. 169.254.169.254 metadata
                || v4.is_unspecified()
                || v4.is_broadcast()
                || v4.is_documentation()
                || v4.is_multicast()
                // 100.64.0.0/10 — RFC 6598 CGNAT / shared address space (not
                // covered by `is_private`; `is_shared` is nightly-only).
                || (o[0] == 100 && (o[1] & 0xc0) == 64)
        }
        IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                // unique-local fc00::/7
                || (v6.segments()[0] & 0xfe00) == 0xfc00
                // link-local fe80::/10
                || (v6.segments()[0] & 0xffc0) == 0xfe80
                // v4-mapped/compatible that map to a blocked v4
                || v6.to_ipv4().map(|m| is_blocked_ip(&IpAddr::V4(m))).unwrap_or(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// A scripted mock transport: maps a method to a canned `result`, and records
    /// the requests it received so the test can assert what was called.
    #[derive(Default)]
    struct MockTransport {
        list_result: Value,
        call_results: std::collections::HashMap<String, Value>,
        calls: Mutex<Vec<(String, Value)>>,
    }

    #[async_trait]
    impl McpTransport for MockTransport {
        async fn request(
            &self,
            method: &str,
            params: Value,
        ) -> std::result::Result<Value, McpError> {
            self.calls
                .lock()
                .unwrap()
                .push((method.to_string(), params.clone()));
            match method {
                "tools/list" => Ok(self.list_result.clone()),
                "tools/call" => {
                    let name = params
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    self.call_results
                        .get(&name)
                        .cloned()
                        .ok_or_else(|| McpError::Server {
                            code: -32601,
                            message: format!("no such tool: {name}"),
                        })
                }
                other => Err(McpError::Protocol(format!("unexpected method {other}"))),
            }
        }
    }

    fn weather_list() -> Value {
        json!({
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get the weather for a city",
                    "inputSchema": {
                        "type": "object",
                        "properties": { "city": { "type": "string" } },
                        "required": ["city"]
                    }
                },
                {
                    "name": "get_time",
                    "description": "Get the current time",
                    "inputSchema": { "type": "object", "properties": {} }
                }
            ]
        })
    }

    fn make_client() -> (McpClient, Arc<MockTransport>) {
        let mut call_results = std::collections::HashMap::new();
        call_results.insert(
            "get_weather".to_string(),
            json!({ "content": [ { "type": "text", "text": "Sunny, 31C" } ] }),
        );
        let transport = Arc::new(MockTransport {
            list_result: weather_list(),
            call_results,
            calls: Mutex::new(Vec::new()),
        });
        (McpClient::new(transport.clone()), transport)
    }

    #[tokio::test]
    async fn lists_tools_and_converts_to_tool_decls() {
        let (client, _t) = make_client();
        let tools = client.list_tools().await.expect("list_tools");
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name, "get_weather");
        let decl = tools[0].to_tool_decl();
        assert_eq!(decl.name, "get_weather");
        assert_eq!(decl.params["required"][0], "city");
    }

    #[tokio::test]
    async fn tool_call_round_trips_text_content() {
        let (client, transport) = make_client();
        let out = client
            .call_tool("get_weather", json!({ "city": "Singapore" }))
            .await
            .expect("call_tool");
        assert_eq!(out, "Sunny, 31C");
        // The transport saw exactly the right tools/call request.
        let calls = transport.calls.lock().unwrap();
        assert!(calls.iter().any(|(m, p)| m == "tools/call"
            && p["name"] == "get_weather"
            && p["arguments"]["city"] == "Singapore"));
    }

    #[tokio::test]
    async fn allow_list_hides_and_blocks_filtered_tools() {
        let (_c, transport) = make_client();
        let client = McpClient::new(transport).with_tools_filter(["get_weather".to_string()]);
        // get_time is filtered out of discovery.
        let tools = client.list_tools().await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "get_weather");
        // …and a call to it is rejected before any network request.
        let err = client.call_tool("get_time", json!({})).await.unwrap_err();
        assert!(matches!(err, McpError::Protocol(_)));
    }

    #[tokio::test]
    async fn processor_dispatches_known_tool_via_direct_call() {
        let (client, _t) = make_client();
        let proc = McpProcessor::new(client);
        // A FunctionCallsStarted for a known MCP tool → FunctionCallResult out.
        let call = FunctionCall {
            function_name: "get_weather".into(),
            tool_call_id: "tc1".into(),
            arguments: json!({ "city": "Singapore" }),
        };
        let result = proc.dispatch(&call).await;
        assert_eq!(result.tool_call_id, "tc1");
        assert_eq!(result.result, Value::String("Sunny, 31C".into()));
    }

    #[tokio::test]
    async fn processor_discovers_tools_on_start_and_dispatches_through_pipeline() {
        let (client, _t) = make_client();
        let proc = McpProcessor::new(client);

        // Run the processor inside a real PipelineTask so the `Start` frame drives
        // its `start` hook (tools/list) and the FunctionCallsStarted flows through
        // the actual frame loop — captured by an observer at the tail.
        let captured = run_capture(
            Box::new(proc),
            vec![Frame::FunctionCallsStarted(vec![
                FunctionCall {
                    function_name: "get_weather".into(),
                    tool_call_id: "a".into(),
                    arguments: json!({ "city": "SG" }),
                },
                FunctionCall {
                    function_name: "transfer_call".into(), // not an MCP tool
                    tool_call_id: "b".into(),
                    arguments: json!({}),
                },
            ])],
        )
        .await;

        // The MCP tool produced a result; the unknown tool was forwarded.
        let results: Vec<_> = captured
            .iter()
            .filter_map(|f| match f {
                Frame::FunctionCallResult(r) => Some(r.function_name.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(results, vec!["get_weather"]);
        let forwarded: Vec<_> = captured
            .iter()
            .filter_map(|f| match f {
                Frame::FunctionCallsStarted(c) => Some(c.len()),
                _ => None,
            })
            .collect();
        assert_eq!(forwarded, vec![1], "unknown tool must be forwarded");
    }

    // ---- pipeline capture harness (uses only flowcat-core's PUBLIC API) ----

    use std::sync::Mutex as StdMutex;

    /// An observer that records every frame pushed into the pipeline's Sink.
    #[derive(Default)]
    struct CaptureObserver {
        frames: StdMutex<Vec<Frame>>,
    }

    #[async_trait]
    impl flowcat_core::FrameObserver for CaptureObserver {
        async fn on_push(&self, e: &flowcat_core::FramePushEvent<'_>) {
            // Capture frames arriving at the internal Sink (pipeline tail).
            if e.destination == "Sink" {
                self.frames.lock().unwrap().push(e.frame.clone());
            }
        }
    }

    /// Run `proc` in a one-element `PipelineTask`, inject `frames`, drain, and
    /// return everything the tail observed.
    async fn run_capture(proc: Box<dyn FrameProcessor>, frames: Vec<Frame>) -> Vec<Frame> {
        use flowcat_core::{Pipeline, PipelineTask, PipelineTaskParams};

        let observer = Arc::new(CaptureObserver::default());
        let pipeline = Pipeline::new(vec![proc]);
        let task = PipelineTask::new(
            pipeline,
            PipelineTaskParams::default(),
            vec![observer.clone()],
        );
        task.queue_frames(frames).await;
        task.stop_when_done().await;
        tokio::time::timeout(std::time::Duration::from_secs(5), task.run())
            .await
            .expect("pipeline task timed out")
            .expect("pipeline task errored");
        let frames = observer.frames.lock().unwrap().clone();
        frames
    }

    // ---- SSRF guard fixture ------------------------------------------------

    use std::net::{Ipv4Addr, Ipv6Addr};

    #[test]
    fn is_blocked_ip_blocks_private_metadata_cgnat_and_allows_public() {
        // Private / loopback / metadata / CGNAT / unspecified are all blocked.
        for ip in [
            "127.0.0.1",
            "10.0.0.1",
            "192.168.1.1",
            "172.16.0.1",
            "169.254.169.254", // cloud metadata
            "0.0.0.0",
            "100.64.0.1", // RFC 6598 CGNAT
            "100.127.255.255",
        ] {
            assert!(
                is_blocked_ip(&ip.parse::<IpAddr>().unwrap()),
                "{ip} must be blocked"
            );
        }
        assert!(is_blocked_ip(&IpAddr::V6(Ipv6Addr::LOCALHOST)));
        assert!(is_blocked_ip(&"fc00::1".parse::<IpAddr>().unwrap())); // ULA
        assert!(is_blocked_ip(&"fe80::1".parse::<IpAddr>().unwrap())); // link-local
                                                                       // Public addresses pass.
        assert!(!is_blocked_ip(&IpAddr::V4(Ipv4Addr::new(8, 8, 8, 8))));
        assert!(!is_blocked_ip(
            &"2001:4860:4860::8888".parse::<IpAddr>().unwrap()
        ));
    }

    #[tokio::test]
    async fn ssrf_check_rejects_unsafe_and_allows_public() {
        // Scheme + localhost are rejected before any DNS; IP literals resolve
        // locally (no network), so these are deterministic + offline.
        for bad in [
            "ftp://example.com/x",                      // non-http(s) scheme
            "file:///etc/passwd",                       // non-http(s) scheme
            "http://localhost/x",                       // localhost by name
            "http://api.localhost/mcp",                 // *.localhost
            "http://127.0.0.1/mcp",                     // loopback literal
            "https://10.0.0.5/mcp",                     // RFC-1918 private
            "http://192.168.1.10/mcp",                  // RFC-1918 private
            "http://169.254.169.254/latest/meta-data/", // metadata
            "http://100.64.0.1/mcp",                    // CGNAT
            "http://[::1]/mcp",                         // IPv6 loopback
        ] {
            let url = reqwest::Url::parse(bad).unwrap();
            assert!(
                ssrf_check(&url).await.is_err(),
                "{bad} must be rejected by the SSRF guard"
            );
        }

        // A public IP literal passes the guard and yields a pinnable address.
        let ok = reqwest::Url::parse("https://8.8.8.8/mcp").unwrap();
        let vetted = ssrf_check(&ok).await.expect("public IP must pass");
        assert!(
            vetted.iter().all(|a| !is_blocked_ip(&a.ip())),
            "vetted addrs must all be public"
        );
        assert_eq!(vetted[0].port(), 443, "https default port pinned");
    }

    #[tokio::test]
    async fn http_transport_guard_rejects_private_url_on_first_request() {
        // The guard is on by default: the first request to a loopback MCP URL is
        // rejected by the SSRF check, with no network call escaping.
        let t = HttpMcpTransport::new("http://127.0.0.1:9/mcp");
        let err = t.request("tools/list", json!({})).await.unwrap_err();
        match err {
            McpError::Transport(msg) => assert!(
                msg.contains("non-public") || msg.contains("localhost"),
                "expected an SSRF rejection, got: {msg}"
            ),
            other => panic!("expected a Transport SSRF error, got {other:?}"),
        }

        // A non-http(s) scheme is likewise rejected by the scheme check.
        let t2 = HttpMcpTransport::new("ftp://example.com/mcp");
        assert!(matches!(
            t2.request("tools/list", json!({})).await,
            Err(McpError::Transport(_))
        ));

        // The explicit opt-out builds a client (no SSRF rejection at guard time);
        // the request then fails only at the connect (port 9 / unreachable), proving
        // the guard — not the network — was the gate above.
        let t3 = HttpMcpTransport::new("http://127.0.0.1:9/mcp").allow_private_url();
        let err = t3.request("tools/list", json!({})).await.unwrap_err();
        match err {
            McpError::Transport(msg) => assert!(
                !msg.contains("non-public"),
                "opt-out must bypass the SSRF guard, got: {msg}"
            ),
            other => panic!("expected a connect-time Transport error, got {other:?}"),
        }
    }
}
