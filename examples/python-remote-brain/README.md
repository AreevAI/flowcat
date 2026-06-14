<!-- SPDX-License-Identifier: Apache-2.0 -->
# Remote brain (Python webhook) example

Drive Flowcat's **conversation policy** from Python — no Rust, no bindings. Your
Python service decides what the agent says, when to change state, and when to end
the call; Flowcat's Rust runtime owns the latency-critical media loop. Your code
is consulted at **turn granularity** (between turns), never on the per-audio-frame
path, so the tail-latency profile is unaffected.

This is powered by the `RemoteBrain` adapter in `flowcat-services` (feature
`brain-http`), which implements the `AgentBrain` seam by calling two JSON
endpoints on the service you run.

## Run the reference server

No dependencies — just Python 3:

```bash
python3 brain_server.py     # listens on http://127.0.0.1:8080
```

It implements a tiny receptionist flow (greeting → confirm → end) showing all
three actions and variable accumulation. Replace `decide()` with your own logic
(an LLM call, a DB lookup, a state machine, …).

## Point Flowcat at it (Rust embedder)

```rust
use flowcat_services::RemoteBrain; // requires feature `brain-http`

// RemoteBrain requires a MULTI-THREADED tokio runtime (#[tokio::main]).
let brain = RemoteBrain::connect(
    "http://127.0.0.1:8080",
    brain_config,        // serde_json::Value — your opaque policy/graph config
    "gemini",            // the realtime/LLM provider name
    None,                // Some("token") to send `Authorization: Bearer token`
)
.await?;

// Pass `brain` as the `AgentBrain` argument to the pipeline builder —
// `flowcat_core::pipeline::build_s2s_task` (realtime S2S) or
// `build_cascaded_task` (STT/LLM/TTS). Both accept any `AgentBrain`.
```

## The wire contract

### `POST /session` — start a session
Request:
```json
{ "brain_config": { "graph": "demo" }, "provider": "gemini" }
```
Response (seeds the initial state):
```json
{
  "system_prompt": "You are a friendly receptionist…",
  "tools": [ { "name": "book_appointment", "description": "…", "params": { "type": "object" } } ],
  "node_id": "greeting",
  "collected_vars": {}
}
```

### `POST /tool-call` — interpret a model tool call
Request:
```json
{
  "node_id": "greeting",
  "tool": { "name": "book_appointment", "args": { "day": "Tuesday" } },
  "collected_vars": {}
}
```
Response — `action` is `"transition"`, `"stay"`, or `"end"`:
```json
{
  "action": "transition",
  "system_prompt": "Confirm the appointment day…",
  "tools": [ … ],
  "say": "Sure — booking you for Tuesday. Shall I confirm?",
  "node_id": "confirm",
  "collected_vars": { "requested_day": "Tuesday" },
  "finished": false
}
```

- `system_prompt` and `tools` are **required** when `action == "transition"` (a
  transition that omits them is treated as a protocol error).
- `disposition` is optional and used with `action == "end"`.
- `node_id`, `collected_vars`, and `finished` are returned on every response and
  become the brain's new cached state.
- If `RemoteBrain` is configured with a token, every request carries
  `Authorization: Bearer <token>` — authenticate it in your handler.

## Fail-safe

If a `/tool-call` request fails (network error, non-2xx, malformed body),
`RemoteBrain` logs a warning and returns `Stay` without changing state — a
transient policy-service hiccup never crashes a live call.
