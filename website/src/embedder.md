# Build an embedder

The [Quickstart](./quickstart.md) runs the runtime in isolation. To carry a real
call you write one small host binary — the **embedder** — that Flowcat
deliberately leaves to you. This is the piece that keeps the whole call on
infrastructure you control: Flowcat owns the media loop; *you* own the call
contract, routing, and credentials.

> The repo ships **no** embedder — only the credential-free `flowcat` demos. The
> wiring shown here lives in flowcat-core's own tests
> ([`pipeline/s2s.rs`](https://github.com/AreevAI/flowcat/blob/main/flowcat-core/src/pipeline/s2s.rs)).
> The fastest on-ramp without writing Rust brain logic is the
> [Python `RemoteBrain`](./quickstart.md#4-drive-the-conversation-from-python).

## The four seams

An embedder supplies four things, then calls one builder. Two have ready-made
implementations you can use as-is; two you implement over your own systems.

| Seam | Trait | Use the built-in… | or implement for… |
|---|---|---|---|
| Media in/out | `MediaTransport` | `SipAgent` / `SipTransport` (native SIP/RTP), or a WS carrier | a custom transport |
| The model | `RealtimeLlm + RealtimeKickoff` | `GeminiLive` (speech-to-speech) | another realtime model |
| The conversation | `AgentBrain` | `RemoteBrain` (HTTP, [step 4](./quickstart.md#4-drive-the-conversation-from-python)) | your own engine, in-process |
| Call resolution + finalize | `SessionSource` | — | your control plane (always yours) |

### `AgentBrain` — what the conversation does

Synchronous on purpose: pure decision logic, no I/O inside.

```rust
pub trait AgentBrain: Send {
    fn system_prompt(&self) -> String;
    fn tools(&self) -> Vec<ToolDecl>;
    fn current_node_id(&self) -> String;
    fn on_tool_call(&mut self, name: &str, args: &serde_json::Value) -> BrainAction;
    fn is_finished(&self) -> bool;
    fn collected_vars(&self) -> serde_json::Value;
}
```

Implement it over your graph/state machine, **or** use `RemoteBrain` to drive
decisions from an HTTP service (your Python, your DB, your LLM) — same trait,
over the wire.

### `SessionSource` — your control plane

How Flowcat resolves a call to a config and reports the result back. Always
yours; Flowcat never sees your API contract.

```rust
#[async_trait]
pub trait SessionSource: Send + Sync {
    async fn resolve(&self, run_id: i64, token: &str) -> Result<ResolvedCall, FlowcatError>;
    async fn complete(&self, run_id: i64, token: &str, fin: Finalize) -> Result<(), FlowcatError>;
    async fn artifact_upload_url(/* run_id, token, kind */) -> Result<UploadTarget, FlowcatError>;
    async fn put_bytes(/* url, bytes, content_type */) -> Result<(), FlowcatError>;
    async fn node_tools(/* run_id, token, node_id */) -> Result<Vec<ToolDecl>, FlowcatError>;
    async fn tool_call(/* run_id, token, node_id, tool, args */) -> Result<String, FlowcatError>;
}
```

> Exact signatures and the `ResolvedCall` / `Finalize` / `UploadTarget` shapes:
> [`flowcat-core/src/session.rs`](https://github.com/AreevAI/flowcat/blob/main/flowcat-core/src/session.rs).

## The entry point

One builder wires the four seams into the running pipeline and returns a task you
drive to completion:

```rust
pub async fn build_s2s_task<T, R, B, S>(
    transport: T,        // MediaTransport       — e.g. SipTransport
    realtime: R,         // RealtimeLlm+Kickoff  — e.g. GeminiLive
    brain: B,            // AgentBrain           — e.g. RemoteBrain
    session: S,          // SessionSource        — yours
    run_id: i64,
    token: String,
    model: String,
) -> Result<S2sTask>;
```

## A minimal SIP embedder

Illustrative — accept inbound SIP calls and drive each with Gemini Live + a
remote brain. Constructor argument lists are abbreviated; take the exact ones
from the linked source / [API reference](./api-reference.md).

```rust
// RemoteBrain needs a multi-threaded runtime.
#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Bring up the native SIP user-agent and register with your trunk.
    let agent = SipAgent::start(SipConfig {
        server:   "sip:sip.example.com".into(),
        login:    std::env::var("SIP_LOGIN")?,
        password: std::env::var("SIP_PASSWORD")?,
        caller_id:"+15551230000".into(),
        public_ip: None,                  // set on a NAT'd host
        sip_port: None,                   // → 5060
        rtp_port_base: 16000,
        rtp_port_tries: 200,              // caps concurrent calls
    }).await?;
    agent.register().await?;

    // 2. One call per inbound INVITE.
    while let Some(invite) = agent.next_inbound().await {
        let transport = invite.answer().await?;      // → SipTransport (MediaTransport)

        let realtime = GeminiLive::new(/* api_key, voice/model from your config */);
        let brain = RemoteBrain::connect(
            "http://127.0.0.1:8080",
            serde_json::json!({ "graph": "receptionist" }),
            "gemini",
            None,
        ).await?;
        let session = MyControlPlane::new(/* … */);   // your SessionSource impl

        let task = build_s2s_task(
            transport, realtime, brain, session,
            /* run_id */ 1, /* token */ "…".into(),
            "models/gemini-live".into(),
        ).await?;

        tokio::spawn(async move { let _ = task.run().await; });
    }
    Ok(())
}
```

`Cargo.toml` (point the deps at the published crates or a git/path checkout):

```toml
[dependencies]
flowcat-core = { git = "https://github.com/AreevAI/flowcat" }
flowcat-services = { git = "https://github.com/AreevAI/flowcat", features = ["brain-http"] }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
serde_json = "1"
```

## Choosing S2S vs cascaded

Two pipeline shapes, same four seams:

- **Speech-to-speech (S2S)** — one realtime model both listens and speaks
  (`build_s2s_task` + a `RealtimeLlm` such as Gemini Live): fewest moving parts and
  a single hop on the hot path.
- **Cascaded** — separate STT → LLM → TTS stages (`build_cascaded_task`): mix and
  match a provider per stage and swap any one independently — more control, more to
  wire.

Rule of thumb: start with S2S; reach for cascaded when you need a specific
STT/LLM/TTS combination a single realtime model can't give you.

## What's verified vs. coming

- **Live-verified today:** speech-to-speech via `build_s2s_task` with **Gemini
  Live + Plivo / native SIP**. Start there.
- **Cascaded STT → LLM → TTS** has a parallel builder, `build_cascaded_task`, with
  the same seam shape; the realtime path is the one verified end-to-end.

## Where to read the exact API

- Trait seams: [`brain.rs`](https://github.com/AreevAI/flowcat/blob/main/flowcat-core/src/brain.rs),
  [`session.rs`](https://github.com/AreevAI/flowcat/blob/main/flowcat-core/src/session.rs),
  [`realtime/mod.rs`](https://github.com/AreevAI/flowcat/blob/main/flowcat-core/src/realtime/mod.rs),
  [`transport`](https://github.com/AreevAI/flowcat/tree/main/flowcat-core/src/transport)
- Builders: [`pipeline/s2s.rs`](https://github.com/AreevAI/flowcat/blob/main/flowcat-core/src/pipeline/s2s.rs)
- SIP: [`sip/agent.rs`](https://github.com/AreevAI/flowcat/blob/main/flowcat-core/src/sip/agent.rs) · [SIP design](./sip-design.md)
- The full contract & call lifecycle: [Design overview](./design.md)
- Browse it all as rustdoc: [API reference](./api-reference.md)

> **Next:** [Configuration](./configuration.md) to tune turn-taking and voice,
> then [Deployment](./deployment.md) to ship it.
