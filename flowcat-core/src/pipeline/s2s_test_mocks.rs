// SPDX-License-Identifier: Apache-2.0
//
//! Shared scripted mock seams for the S2S equivalence tests (test-only).
//!
//! These scripted mocks drive the S2S processor [`PipelineTask`](crate::pipeline::PipelineTask)
//! in `s2s::tests` (the §7.2 behaviour gate), letting the realtime pipeline be
//! exercised end-to-end off deterministic inputs with no network/Gemini.
//!
//! Test-only: the whole module is `#[cfg(test)]`-gated by its `mod` declaration
//! in `pipeline/mod.rs`, so it never ships in a non-test build.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::json;

use crate::brain::AgentBrain;
use crate::error::FlowcatError;
use crate::realtime::RealtimeLlm;
use crate::session::SessionSource;
use crate::transport::socket::MediaSocket;
use crate::types::{
    AudioChunk, BrainAction, Finalize, RealtimeEvent, RealtimeSetup, ResolvedCall, ToolDecl,
    UploadTarget, Usage, WsIn, WsOut,
};

use crate::realtime::RealtimeKickoff;

/// The carrier (telephony G.711) sample rate the scripted Plivo media uses.
pub(crate) const CARRIER_RATE: u32 = 8_000;
/// The tool name the mock brain treats as "end the call".
pub(crate) const END_TOOL: &str = "end_call";

pub(crate) fn b64(bytes: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// A Plivo `start` text frame (so `PlivoSerializer` yields `StreamStart`).
pub(crate) fn plivo_start() -> WsIn {
    WsIn::Text(
        json!({
            "event": "start",
            "start": { "streamId": "strm-test", "callId": "call-test",
                       "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000} }
        })
        .to_string(),
    )
}

/// A Plivo `media` text frame carrying `n` μ-law samples (decoded to PCM by the
/// serializer). `n` is chosen ≥ the resampler block so audio reaches the model on
/// a single frame.
pub(crate) fn plivo_media(n: usize) -> WsIn {
    // Encode a constant-ish PCM tone to μ-law, then base64 (the Plivo wire).
    let pcm: Vec<i16> = (0..n).map(|i| ((i as i16 % 64) * 200) - 6400).collect();
    let ulaw = crate::codec::pcm16_to_ulaw(&pcm);
    WsIn::Text(json!({ "event": "media", "media": { "payload": b64(&ulaw) } }).to_string())
}

// ---- Mock MediaSocket ------------------------------------------------------

/// Scripted media socket: yields a queue of inbound frames, then (for the
/// scripted trailing `Stop`) delays so the realtime branch can finish its own
/// script first (the loop ends on the brain's `End`, not this `Stop`). Captures
/// every outbound frame so the test can assert bot audio flowed out.
pub(crate) struct MockSocket {
    /// Frames returned immediately, in order (StreamStart, Audio, Audio, …).
    ready: VecDeque<WsIn>,
    /// Frame returned only after a delay (the trailing scripted `Stop`).
    delayed_stop: Option<WsIn>,
    /// Outbound frames the pipeline sent back (shared for assertions).
    sent: Arc<Mutex<Vec<WsOut>>>,
}

impl MockSocket {
    pub(crate) fn new(sent: Arc<Mutex<Vec<WsOut>>>) -> Self {
        let mut ready = VecDeque::new();
        ready.push_back(plivo_start());
        // Two audio frames, each large enough to clear the 8k→16k block.
        ready.push_back(plivo_media(400));
        ready.push_back(plivo_media(400));
        Self {
            ready,
            // Scripted trailing Stop — delivered late so it is the fallback
            // terminator, not the actual one (the brain End wins first).
            delayed_stop: Some(WsIn::Text(json!({ "event": "stop" }).to_string())),
            sent,
        }
    }
}

#[async_trait]
impl MediaSocket for MockSocket {
    async fn recv(&mut self) -> Option<WsIn> {
        if let Some(f) = self.ready.pop_front() {
            return Some(f);
        }
        // All immediate frames drained: deliver the scripted Stop, but only after
        // a delay long enough for the realtime script (paced at ~1ms) to reach its
        // End. Keeps the loop's terminator deterministic.
        if let Some(stop) = self.delayed_stop.take() {
            tokio::time::sleep(Duration::from_millis(200)).await;
            return Some(stop);
        }
        // Nothing left: pend forever (the loop will already have broken).
        std::future::pending::<()>().await;
        None
    }

    async fn send_text(&mut self, s: String) -> Result<(), FlowcatError> {
        self.sent.lock().unwrap().push(WsOut::Text(s));
        Ok(())
    }

    async fn send_binary(&mut self, b: Vec<u8>) -> Result<(), FlowcatError> {
        self.sent.lock().unwrap().push(WsOut::Binary(b));
        Ok(())
    }
}

// ---- Mock RealtimeLlm ------------------------------------------------------

/// Scripted realtime model. Records `connect`, `kickoff`, and every `send_audio`
/// call, then (once kicked off) emits a fixed event script: AudioOut(24k),
/// BotText, UserText, Usage, ToolCall(END_TOOL), Closed.
pub(crate) struct MockRealtime {
    /// Set true by `connect` (shared so the test can assert we connected).
    connected: Arc<AtomicBool>,
    /// Set by `kickoff` (a different method than `next_event`); an atomic so the
    /// gate in `next_event` reads a flag toggled "out of band" cleanly.
    kicked_off: Arc<AtomicBool>,
    /// Count of caller-audio chunks the pipeline pushed in.
    audio_received: Arc<Mutex<usize>>,
    /// Tool-result statuses the pipeline sent back (id, status).
    tool_results: Arc<Mutex<Vec<(String, String)>>>,
    /// The pending event script (drained by `next_event`).
    script: VecDeque<RealtimeEvent>,
}

impl MockRealtime {
    pub(crate) fn new(
        connected: Arc<AtomicBool>,
        audio_received: Arc<Mutex<usize>>,
        tool_results: Arc<Mutex<Vec<(String, String)>>>,
    ) -> Self {
        // 24k bot audio, large enough to clear the 24k→8k block so it is encoded
        // + sent to the carrier.
        let bot_pcm: Vec<i16> = (0..720).map(|i| ((i as i16 % 32) * 400) - 6400).collect();
        let mut script = VecDeque::new();
        script.push_back(RealtimeEvent::AudioOut(AudioChunk::new(bot_pcm, 24_000)));
        script.push_back(RealtimeEvent::BotText("Hello, how can I help?".into()));
        // Streaming user partials (accumulated into one interim line) then the
        // finalized utterance — exercises the delta→interim, completed→final path.
        script.push_back(RealtimeEvent::UserInterimText("I'd ".into()));
        script.push_back(RealtimeEvent::UserInterimText("like to end now".into()));
        script.push_back(RealtimeEvent::UserText("I'd like to end now".into()));
        script.push_back(RealtimeEvent::Usage(Usage {
            input_tokens: Some(10),
            output_tokens: Some(5),
            total_tokens: Some(15),
            extra: None,
        }));
        script.push_back(RealtimeEvent::ToolCall {
            id: "fc-end-1".into(),
            name: END_TOOL.into(),
            args: json!({}),
        });
        // Trailing Closed (scripted but unreached: End breaks the loop first).
        script.push_back(RealtimeEvent::Closed);
        Self {
            connected,
            kicked_off: Arc::new(AtomicBool::new(false)),
            audio_received,
            tool_results,
            script,
        }
    }
}

#[async_trait]
impl RealtimeLlm for MockRealtime {
    async fn connect(&mut self, _setup: RealtimeSetup) -> Result<(), FlowcatError> {
        self.connected.store(true, Ordering::SeqCst);
        Ok(())
    }

    async fn send_audio(&mut self, _chunk: AudioChunk) -> Result<(), FlowcatError> {
        *self.audio_received.lock().unwrap() += 1;
        Ok(())
    }

    async fn update_system(
        &mut self,
        _prompt: String,
        _tools: Vec<ToolDecl>,
    ) -> Result<(), FlowcatError> {
        Ok(())
    }

    async fn send_tool_result(
        &mut self,
        id: String,
        result: serde_json::Value,
    ) -> Result<(), FlowcatError> {
        let status = result
            .get("status")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        self.tool_results.lock().unwrap().push((id, status));
        Ok(())
    }

    async fn next_event(&mut self) -> Option<RealtimeEvent> {
        // Bot-first: emit nothing until the pipeline has kicked us off (the flag
        // is flipped by `kickoff` between select polls).
        while !self.kicked_off.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        // A small pace so the transport's immediately-ready frames win the select
        // first (caller audio is consumed before the bot replies).
        tokio::time::sleep(Duration::from_millis(1)).await;
        self.script.pop_front()
    }
}

#[async_trait]
impl RealtimeKickoff for MockRealtime {
    async fn kickoff(&mut self) -> Result<(), FlowcatError> {
        self.kicked_off.store(true, Ordering::SeqCst);
        Ok(())
    }
}

// ---- Mock AgentBrain -------------------------------------------------------

/// Mock brain: any tool other than `END_TOOL` is a `Stay`; `END_TOOL` ends the
/// call with a fixed disposition. Exposes fixed collected vars.
pub(crate) struct MockBrain {
    /// Recorded tool-call names (for assertions).
    seen_tools: Arc<Mutex<Vec<String>>>,
}

impl MockBrain {
    pub(crate) fn new(seen_tools: Arc<Mutex<Vec<String>>>) -> Self {
        Self { seen_tools }
    }
}

impl AgentBrain for MockBrain {
    fn system_prompt(&self) -> String {
        "You are a test agent.".into()
    }

    fn tools(&self) -> Vec<ToolDecl> {
        vec![ToolDecl {
            name: END_TOOL.into(),
            description: "End the call.".into(),
            params: json!({ "type": "object", "properties": {} }),
        }]
    }

    fn current_node_id(&self) -> String {
        "start".into()
    }

    fn on_tool_call(&mut self, name: &str, _args: &serde_json::Value) -> BrainAction {
        self.seen_tools.lock().unwrap().push(name.to_string());
        if name == END_TOOL {
            BrainAction::End {
                disposition: Some("completed".into()),
            }
        } else {
            BrainAction::Stay
        }
    }

    fn is_finished(&self) -> bool {
        false
    }

    fn collected_vars(&self) -> serde_json::Value {
        json!({ "name": "Ada", "intent": "support" })
    }
}

// ---- Mock SessionSource ----------------------------------------------------

/// Captures the finalize payload + the artifact uploads it received.
#[derive(Default)]
pub(crate) struct Captured {
    pub(crate) complete_calls: usize,
    pub(crate) finalize: Option<Finalize>,
    /// (kind-ish url, content_type, byte_len) for each PUT.
    pub(crate) uploads: Vec<(String, String, usize)>,
    /// The uploaded transcript artifact body (UTF-8), so tests can assert its
    /// content (e.g. the `[transition: <node name>]` marker). `None` until the
    /// transcript artifact is PUT.
    pub(crate) transcript_body: Option<String>,
    /// Every `tool_call` relayed to the control plane:
    /// (node_id, tool_name, arguments).
    pub(crate) tool_calls: Vec<(String, String, serde_json::Value)>,
}

pub(crate) struct MockSession {
    captured: Arc<Mutex<Captured>>,
    /// MCP/HTTP workflow tools to advertise for any node (empty by default).
    node_tools: Vec<ToolDecl>,
    /// The `content` string `tool_call` returns.
    tool_result: String,
}

impl MockSession {
    pub(crate) fn new(captured: Arc<Mutex<Captured>>) -> Self {
        Self {
            captured,
            node_tools: vec![],
            tool_result: String::new(),
        }
    }

    /// Configure the session to advertise `tools` as the node's MCP tools and
    /// return `result` as the `tool_call` content.
    pub(crate) fn with_node_tools(
        captured: Arc<Mutex<Captured>>,
        tools: Vec<ToolDecl>,
        result: impl Into<String>,
    ) -> Self {
        Self {
            captured,
            node_tools: tools,
            tool_result: result.into(),
        }
    }
}

#[async_trait]
impl SessionSource for MockSession {
    async fn resolve(&self, _run_id: i64, _token: &str) -> Result<ResolvedCall, FlowcatError> {
        Ok(ResolvedCall {
            provider: "plivo".into(),
            brain_config: json!({}),
            is_completed: false,
        })
    }

    async fn complete(
        &self,
        _run_id: i64,
        _token: &str,
        fin: Finalize,
    ) -> Result<(), FlowcatError> {
        let mut c = self.captured.lock().unwrap();
        c.complete_calls += 1;
        c.finalize = Some(fin);
        Ok(())
    }

    async fn artifact_upload_url(
        &self,
        _run_id: i64,
        _token: &str,
        kind: &str,
    ) -> Result<UploadTarget, FlowcatError> {
        Ok(UploadTarget {
            url: format!("https://uploads.test/{kind}?sig=secret"),
            // The stored key is distinct from the presigned URL — finalize must
            // persist THIS, not the (expiring, secret-bearing) upload URL.
            key: format!("runs/4242/{kind}"),
            content_type: String::new(), // let the caller's hint stand
        })
    }

    async fn put_bytes(
        &self,
        url: &str,
        bytes: Vec<u8>,
        content_type: &str,
    ) -> Result<(), FlowcatError> {
        let mut c = self.captured.lock().unwrap();
        // The transcript artifact (kind "transcript" → upload url) — keep its body
        // so tests can assert the rendered transcript content.
        if url.contains("transcript") {
            c.transcript_body = Some(String::from_utf8_lossy(&bytes).to_string());
        }
        c.uploads
            .push((url.to_string(), content_type.to_string(), bytes.len()));
        Ok(())
    }

    async fn node_tools(
        &self,
        _run_id: i64,
        _token: &str,
        _node_id: &str,
    ) -> Result<Vec<ToolDecl>, FlowcatError> {
        Ok(self.node_tools.clone())
    }

    async fn tool_call(
        &self,
        _run_id: i64,
        _token: &str,
        node_id: &str,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> Result<String, FlowcatError> {
        self.captured.lock().unwrap().tool_calls.push((
            node_id.to_string(),
            tool_name.to_string(),
            args.clone(),
        ));
        Ok(self.tool_result.clone())
    }
}

// ---- Mock RealtimeLlm for the MCP-relay scenario ---------------------------

/// A scripted realtime that emits one MCP `ToolCall`, then an `endCall` `ToolCall`
/// (so the brain ends the call and the loop exits), capturing the RAW JSON of every
/// `send_tool_result` so the test can assert the MCP result content is returned
/// verbatim (not a `{status}` envelope).
pub(crate) struct McpMockRealtime {
    kicked_off: Arc<AtomicBool>,
    /// Raw (id, result-json) of each send_tool_result.
    raw_results: Arc<Mutex<Vec<(String, serde_json::Value)>>>,
    /// Tools the model was (re-)advertised at connect + each update_system.
    advertised: Arc<Mutex<Vec<Vec<String>>>>,
    script: VecDeque<RealtimeEvent>,
}

impl McpMockRealtime {
    pub(crate) fn new(
        raw_results: Arc<Mutex<Vec<(String, serde_json::Value)>>>,
        advertised: Arc<Mutex<Vec<Vec<String>>>>,
        mcp_tool_name: &str,
    ) -> Self {
        let mut script = VecDeque::new();
        // 1. The model calls the workflow (MCP) tool with some args.
        script.push_back(RealtimeEvent::ToolCall {
            id: "fc-mcp-1".into(),
            name: mcp_tool_name.into(),
            args: json!({ "specialty": "dentistry" }),
        });
        // 2. Then it ends the call (a real transition/end via the brain).
        script.push_back(RealtimeEvent::ToolCall {
            id: "fc-end-1".into(),
            name: END_TOOL.into(),
            args: json!({}),
        });
        script.push_back(RealtimeEvent::Closed);
        Self {
            kicked_off: Arc::new(AtomicBool::new(false)),
            raw_results,
            advertised,
            script,
        }
    }
}

#[async_trait]
impl RealtimeLlm for McpMockRealtime {
    async fn connect(&mut self, setup: RealtimeSetup) -> Result<(), FlowcatError> {
        self.advertised
            .lock()
            .unwrap()
            .push(setup.tools.iter().map(|t| t.name.clone()).collect());
        Ok(())
    }

    async fn send_audio(&mut self, _chunk: AudioChunk) -> Result<(), FlowcatError> {
        Ok(())
    }

    async fn update_system(
        &mut self,
        _prompt: String,
        tools: Vec<ToolDecl>,
    ) -> Result<(), FlowcatError> {
        self.advertised
            .lock()
            .unwrap()
            .push(tools.iter().map(|t| t.name.clone()).collect());
        Ok(())
    }

    async fn send_tool_result(
        &mut self,
        id: String,
        result: serde_json::Value,
    ) -> Result<(), FlowcatError> {
        self.raw_results.lock().unwrap().push((id, result));
        Ok(())
    }

    async fn next_event(&mut self) -> Option<RealtimeEvent> {
        while !self.kicked_off.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
        self.script.pop_front()
    }
}

#[async_trait]
impl RealtimeKickoff for McpMockRealtime {
    async fn kickoff(&mut self) -> Result<(), FlowcatError> {
        self.kicked_off.store(true, Ordering::SeqCst);
        Ok(())
    }
}
