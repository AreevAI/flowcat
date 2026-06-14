// SPDX-License-Identifier: Apache-2.0
//
//! **Amazon Nova Sonic** (speech-to-speech) client — fixture-skeleton.
//!
//! Nova Sonic does **not** use a WebSocket. It runs over the AWS Bedrock Runtime
//! `InvokeModelWithBidirectionalStream` API: each direction is a stream of
//! length-framed chunks, and the payloads are **JSON event envelopes** of the
//! form `{"event": {"<eventType>": {…}}}`. The client→server event sequence is
//! `sessionStart → promptStart → contentStart(SYSTEM text) → … → audioInput* →
//! promptEnd → sessionEnd`; the server→client stream carries
//! `contentStart / textOutput / audioOutput / toolUse / contentEnd /
//! completionEnd`.
//!
//! Cross-checked against `pipecat/src/pipecat/services/aws/nova_sonic/llm.py`.
//!
//! **Fixture-skeleton scope:** the wire **encode/decode** of every key event
//! envelope is implemented + tested (the part that is provider-protocol, not
//! transport). The **AWS Bedrock bidirectional-stream transport** (SigV4 auth +
//! the event-stream framing) is a live-only follow-up — it needs the AWS SDK /
//! credentials and a network. So [`NovaSonicRealtime::connect`]/`send_audio`
//! return a documented "transport not wired" error until that lands, while the
//! encoders/decoders are usable + tested now.
//!
//! ## Keys / auth (security note)
//!
//! Nova Sonic authenticates with **AWS credentials** (SigV4 on the Bedrock
//! stream), not a bearer token. This skeleton holds an opaque
//! [`NovaSonicAuth`] (region + credentials) and never logs it; wiring SigV4 is
//! the follow-up.

use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine as _;
use serde_json::{json, Value};

use flowcat_core::error::FlowcatError;
use flowcat_core::processor::frame::AudioFrame;
use flowcat_core::service::{RealtimeLlmService, RealtimeServiceSetup, Tool};
use flowcat_core::types::{AudioChunk, RealtimeEvent, ToolDecl};

/// Nova Sonic input/output PCM rate (mono 16-bit). The reference uses 16 kHz in,
/// 24 kHz out; the output rate is read from the setup.
const NOVA_SONIC_OUTPUT_RATE: u32 = 24_000;

/// Opaque AWS auth for the Bedrock bidi stream (region + credentials). Held but
/// never logged; SigV4 wiring is the follow-up.
#[derive(Clone)]
pub struct NovaSonicAuth {
    pub region: String,
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: Option<String>,
}

impl std::fmt::Debug for NovaSonicAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never leak the secret key in logs/Debug output.
        f.debug_struct("NovaSonicAuth")
            .field("region", &self.region)
            .field("access_key_id", &"<redacted>")
            .field("secret_access_key", &"<redacted>")
            .field(
                "session_token",
                &self.session_token.as_ref().map(|_| "<redacted>"),
            )
            .finish()
    }
}

/// An Amazon Nova Sonic realtime session (fixture-skeleton; see module docs).
pub struct NovaSonicRealtime {
    auth: NovaSonicAuth,
    model: String,
    /// A stable prompt name for the session's events (Nova Sonic keys content by
    /// `promptName`). Generated per session.
    prompt_name: String,
    output_rate: u32,
    /// Whether a live Bedrock stream is wired. Always false in the skeleton.
    connected: bool,
}

impl NovaSonicRealtime {
    /// Construct a client with AWS auth + a model id (e.g.
    /// `amazon.nova-2-sonic-v1:0`).
    pub fn new(auth: NovaSonicAuth, model: impl Into<String>) -> Self {
        Self {
            auth,
            model: model.into(),
            prompt_name: "flowcat-nova-prompt".to_string(),
            output_rate: NOVA_SONIC_OUTPUT_RATE,
            connected: false,
        }
    }

    /// The model id this session targets.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The AWS region (never logs the secret key).
    pub fn region(&self) -> &str {
        &self.auth.region
    }
}

#[async_trait]
impl RealtimeLlmService for NovaSonicRealtime {
    async fn connect(&mut self, setup: RealtimeServiceSetup) -> Result<(), FlowcatError> {
        if setup.output_sample_rate != 0 {
            self.output_rate = setup.output_sample_rate;
        }
        // The event *envelopes* for the open sequence are built + validated here
        // (so the encoders are exercised), but the AWS Bedrock bidi transport is
        // a follow-up — surface a clear, non-panicking error.
        let _open_sequence = [
            encode_session_start(),
            encode_prompt_start(&self.prompt_name, &setup.tools),
            encode_text_content(&self.prompt_name, "SYSTEM", &setup.system_prompt),
        ];
        Err(FlowcatError::Realtime(
            "nova-sonic: Bedrock bidirectional-stream transport not wired yet (SigV4 + AWS \
             event-stream is the follow-up); encoders/decoders are usable + fixture-tested"
                .into(),
        ))
    }

    async fn send_audio(&mut self, chunk: Arc<AudioFrame>) -> Result<(), FlowcatError> {
        if !self.connected {
            return Err(FlowcatError::Realtime("nova-sonic: not connected".into()));
        }
        // Encoder is exercised; transport send is the follow-up.
        let _event = encode_audio_input(&self.prompt_name, "audio-content", &chunk.pcm);
        Ok(())
    }

    async fn update_system(
        &mut self,
        prompt: String,
        _tools: Vec<Tool>,
    ) -> Result<(), FlowcatError> {
        if !self.connected {
            return Err(FlowcatError::Realtime("nova-sonic: not connected".into()));
        }
        // A SYSTEM text content block carries an updated instruction.
        let _event = encode_text_content(&self.prompt_name, "SYSTEM", &prompt);
        Ok(())
    }

    async fn send_tool_result(&mut self, id: String, result: Value) -> Result<(), FlowcatError> {
        if !self.connected {
            return Err(FlowcatError::Realtime("nova-sonic: not connected".into()));
        }
        let _event = encode_tool_result(&self.prompt_name, &id, &result);
        Ok(())
    }

    async fn next_event(&mut self) -> Option<RealtimeEvent> {
        // No live stream in the skeleton.
        None
    }
}

// ---------------------------------------------------------------------------
// Encoders (client → server event envelopes).
// ---------------------------------------------------------------------------

/// `{"event":{"sessionStart":{"inferenceConfiguration":{…}}}}`.
fn encode_session_start() -> Value {
    json!({
        "event": {
            "sessionStart": {
                "inferenceConfiguration": {
                    "maxTokens": 1024,
                    "topP": 0.9,
                    "temperature": 0.7
                }
            }
        }
    })
}

/// `{"event":{"promptStart":{promptName, textOutputConfiguration,
/// audioOutputConfiguration, toolConfiguration}}}`. Tools are advertised here.
fn encode_prompt_start(prompt_name: &str, tools: &[ToolDecl]) -> Value {
    let mut prompt_start = json!({
        "promptName": prompt_name,
        "textOutputConfiguration": { "mediaType": "text/plain" },
        "audioOutputConfiguration": {
            "mediaType": "audio/lpcm",
            "sampleRateHertz": 24000,
            "sampleSizeBits": 16,
            "channelCount": 1,
            "voiceId": "matthew",
            "encoding": "base64",
            "audioType": "SPEECH"
        }
    });
    if !tools.is_empty() {
        let specs: Vec<Value> = tools.iter().map(encode_tool_spec).collect();
        prompt_start["toolConfiguration"] = json!({ "tools": specs });
    }
    json!({ "event": { "promptStart": prompt_start } })
}

/// A single tool spec for Nova Sonic's `toolConfiguration.tools`.
fn encode_tool_spec(tool: &ToolDecl) -> Value {
    json!({
        "toolSpec": {
            "name": tool.name,
            "description": tool.description,
            // Nova Sonic takes the JSON schema as a *string* under inputSchema.json.
            "inputSchema": { "json": tool.params.to_string() }
        }
    })
}

/// A text content block (`contentStart` text + `textInput` + `contentEnd`)
/// collapsed into a single representative envelope for the skeleton. The role is
/// `SYSTEM`/`USER`/`ASSISTANT`.
fn encode_text_content(prompt_name: &str, role: &str, text: &str) -> Value {
    json!({
        "event": {
            "textInput": {
                "promptName": prompt_name,
                "role": role,
                "content": text
            }
        }
    })
}

/// `{"event":{"audioInput":{promptName, contentName, content:<base64 PCM>}}}`.
fn encode_audio_input(prompt_name: &str, content_name: &str, pcm: &[i16]) -> Value {
    let bytes = pcm_to_le_bytes(pcm);
    let content = base64::engine::general_purpose::STANDARD.encode(&bytes);
    json!({
        "event": {
            "audioInput": {
                "promptName": prompt_name,
                "contentName": content_name,
                "content": content
            }
        }
    })
}

/// `{"event":{"toolResult":{promptName, contentName, content:<stringified>}}}`.
fn encode_tool_result(prompt_name: &str, tool_use_id: &str, result: &Value) -> Value {
    let content = match result {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    json!({
        "event": {
            "toolResult": {
                "promptName": prompt_name,
                "contentName": tool_use_id,
                "content": content
            }
        }
    })
}

fn pcm_to_le_bytes(pcm: &[i16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(pcm.len() * 2);
    for s in pcm {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

// Used by `decode_server_event` (the fixture-tested server decoder). Both are
// exercised by the unit tests but not yet by a live `next_event` — the Bedrock
// bidi transport that drives them is the follow-up — so allow dead_code here
// rather than leave the decoder un-wired/untested.
#[allow(dead_code)]
fn le_bytes_to_pcm(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

// ---------------------------------------------------------------------------
// Decoder (server → client event envelopes).
// ---------------------------------------------------------------------------

/// Map one Nova Sonic server event envelope (`{"event":{…}}`) into zero or more
/// [`RealtimeEvent`]s. `out_rate` is the negotiated audio-output rate.
///
/// Recognised inner event types:
/// - `audioOutput.content` (base64 PCM) → `AudioOut`
/// - `textOutput{role,content}` → `BotText` (ASSISTANT) / `UserText` (USER)
/// - `toolUse{toolName,toolUseId,content}` → `ToolCall`
/// - `completionEnd` → `Closed`
///
/// Fixture-tested now; the live Bedrock bidi transport that calls it from
/// `next_event` is the follow-up (hence `allow(dead_code)` in this skeleton).
#[allow(dead_code)]
pub(crate) fn decode_server_event(value: &Value, out_rate: u32) -> Vec<RealtimeEvent> {
    let mut out = Vec::new();
    let Some(event) = value.get("event") else {
        return out;
    };

    if let Some(audio) = event.get("audioOutput") {
        if let Some(b64) = audio.get("content").and_then(Value::as_str) {
            if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(b64) {
                out.push(RealtimeEvent::AudioOut(AudioChunk::new(
                    le_bytes_to_pcm(&bytes),
                    out_rate,
                )));
            }
        }
    }

    if let Some(text) = event.get("textOutput") {
        if let Some(content) = text.get("content").and_then(Value::as_str) {
            if !content.is_empty() {
                let role = text
                    .get("role")
                    .and_then(Value::as_str)
                    .unwrap_or("ASSISTANT");
                // USER role = the model's transcription of the caller.
                if role.eq_ignore_ascii_case("USER") {
                    out.push(RealtimeEvent::UserText(content.to_owned()));
                } else {
                    out.push(RealtimeEvent::BotText(content.to_owned()));
                }
            }
        }
    }

    if let Some(tool) = event.get("toolUse") {
        let name = tool
            .get("toolName")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let id = tool
            .get("toolUseId")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        // `content` is a JSON string of the arguments.
        let args = tool
            .get("content")
            .and_then(Value::as_str)
            .and_then(|s| serde_json::from_str::<Value>(s).ok())
            .or_else(|| tool.get("content").cloned())
            .unwrap_or(Value::Null);
        out.push(RealtimeEvent::ToolCall { name, args, id });
    }

    if event.get("completionEnd").is_some() {
        out.push(RealtimeEvent::Closed);
    }

    out
}

// ===========================================================================
// Tests — pure encode/decode against hand-written fixtures, NO live transport.
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn auth() -> NovaSonicAuth {
        NovaSonicAuth {
            region: "us-east-1".into(),
            access_key_id: "AKIAEXAMPLEKEYID".into(),
            secret_access_key: "tOpS3cretValueXYZ".into(),
            session_token: None,
        }
    }

    fn tools() -> Vec<ToolDecl> {
        vec![ToolDecl {
            name: "transition_to_billing".into(),
            description: "Move to billing.".into(),
            params: json!({ "type": "object", "properties": {} }),
        }]
    }

    // ---- ENCODE -----------------------------------------------------------

    #[test]
    fn session_start_has_inference_config() {
        let v = encode_session_start();
        assert!(v["event"]["sessionStart"]["inferenceConfiguration"]["maxTokens"].is_number());
    }

    #[test]
    fn prompt_start_advertises_tools_and_audio_config() {
        let v = encode_prompt_start("p1", &tools());
        let ps = &v["event"]["promptStart"];
        assert_eq!(ps["promptName"], "p1");
        assert_eq!(ps["audioOutputConfiguration"]["mediaType"], "audio/lpcm");
        let spec = &ps["toolConfiguration"]["tools"][0]["toolSpec"];
        assert_eq!(spec["name"], "transition_to_billing");
        // The JSON schema is carried as a *string* under inputSchema.json.
        assert_eq!(
            spec["inputSchema"]["json"],
            json!({ "type": "object", "properties": {} }).to_string()
        );
    }

    #[test]
    fn prompt_start_omits_tool_config_when_empty() {
        let v = encode_prompt_start("p1", &[]);
        assert!(v["event"]["promptStart"].get("toolConfiguration").is_none());
    }

    #[test]
    fn text_content_carries_role_and_text() {
        let v = encode_text_content("p1", "SYSTEM", "You are helpful.");
        let ti = &v["event"]["textInput"];
        assert_eq!(ti["role"], "SYSTEM");
        assert_eq!(ti["content"], "You are helpful.");
    }

    #[test]
    fn audio_input_base64_round_trips() {
        let pcm = vec![1_i16, -1, 256, i16::MIN];
        let v = encode_audio_input("p1", "c1", &pcm);
        let b64 = v["event"]["audioInput"]["content"].as_str().unwrap();
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(b64)
            .unwrap();
        assert_eq!(le_bytes_to_pcm(&bytes), pcm);
    }

    #[test]
    fn tool_result_stringifies_json() {
        let v = encode_tool_result("p1", "tu-1", &json!({ "ok": true }));
        let tr = &v["event"]["toolResult"];
        assert_eq!(tr["contentName"], "tu-1");
        assert_eq!(tr["content"], json!({ "ok": true }).to_string());
    }

    // ---- DECODE -----------------------------------------------------------

    #[test]
    fn decode_audio_output() {
        let pcm = vec![3_i16, -3];
        let b64 = base64::engine::general_purpose::STANDARD.encode(pcm_to_le_bytes(&pcm));
        let frame = json!({ "event": { "audioOutput": { "content": b64 } } });
        match &decode_server_event(&frame, 24_000)[0] {
            RealtimeEvent::AudioOut(c) => {
                assert_eq!(c.sample_rate, 24_000);
                assert_eq!(c.pcm, pcm);
            }
            other => panic!("expected AudioOut, got {other:?}"),
        }
    }

    #[test]
    fn decode_text_output_role_routing() {
        let bot = json!({ "event": { "textOutput": { "role": "ASSISTANT", "content": "hi" } } });
        assert!(
            matches!(&decode_server_event(&bot, 24_000)[0], RealtimeEvent::BotText(t) if t == "hi")
        );

        let user = json!({ "event": { "textOutput": { "role": "USER", "content": "hello" } } });
        assert!(
            matches!(&decode_server_event(&user, 24_000)[0], RealtimeEvent::UserText(t) if t == "hello")
        );
    }

    #[test]
    fn decode_tool_use_parses_json_string_args() {
        let frame = json!({
            "event": { "toolUse": {
                "toolName": "transition_to_billing",
                "toolUseId": "tu-9",
                "content": "{\"reason\":\"asked\"}"
            }}
        });
        match &decode_server_event(&frame, 24_000)[0] {
            RealtimeEvent::ToolCall { id, name, args } => {
                assert_eq!(id, "tu-9");
                assert_eq!(name, "transition_to_billing");
                assert_eq!(args, &json!({ "reason": "asked" }));
            }
            other => panic!("expected ToolCall, got {other:?}"),
        }
    }

    #[test]
    fn decode_completion_end_is_closed() {
        let frame = json!({ "event": { "completionEnd": {} } });
        assert!(matches!(
            decode_server_event(&frame, 24_000)[0],
            RealtimeEvent::Closed
        ));
    }

    #[test]
    fn decode_unknown_event_yields_nothing() {
        assert!(
            decode_server_event(&json!({ "event": { "contentStart": {} } }), 24_000).is_empty()
        );
        // A frame without the "event" envelope is ignored.
        assert!(decode_server_event(&json!({ "noise": 1 }), 24_000).is_empty());
    }

    #[test]
    fn auth_debug_redacts_the_secret() {
        let dbg = format!("{:?}", auth());
        assert!(dbg.contains("us-east-1"), "region is fine to show");
        // The secret + access-key *values* must never appear in Debug output.
        assert!(
            !dbg.contains("tOpS3cretValueXYZ"),
            "secret key must not leak in Debug"
        );
        assert!(
            !dbg.contains("AKIAEXAMPLEKEYID"),
            "access key id must not leak in Debug"
        );
        assert!(dbg.contains("<redacted>"), "fields are redacted");
    }

    #[tokio::test]
    async fn connect_errors_until_transport_is_wired() {
        // The skeleton returns a clear, non-panicking error (not a todo!()).
        let mut c = NovaSonicRealtime::new(auth(), "amazon.nova-2-sonic-v1:0");
        let err = c
            .connect(RealtimeServiceSetup {
                model: "amazon.nova-2-sonic-v1:0".into(),
                system_prompt: "hi".into(),
                tools: tools(),
                input_sample_rate: 16_000,
                output_sample_rate: 24_000,
            })
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("transport not wired"));
        assert_eq!(c.model(), "amazon.nova-2-sonic-v1:0");
        assert_eq!(c.region(), "us-east-1");
    }

    /// `AWS_REGION=… AWS_ACCESS_KEY_ID=… AWS_SECRET_ACCESS_KEY=… cargo test \
    ///   -p flowcat-services --features realtime-novasonic -- \
    ///   realtime::nova_sonic::tests::live_nova_sonic_smoke --ignored --nocapture`
    ///
    /// Currently a no-op placeholder: the Bedrock bidi transport is the
    /// follow-up; this documents the credentials env it will use.
    #[tokio::test]
    #[ignore = "live: needs AWS creds + the Bedrock bidi transport (follow-up)"]
    async fn live_nova_sonic_smoke() {
        let _ = (
            std::env::var("AWS_REGION"),
            std::env::var("AWS_ACCESS_KEY_ID"),
            std::env::var("AWS_SECRET_ACCESS_KEY"),
        );
    }
}
