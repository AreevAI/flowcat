<!-- SPDX-License-Identifier: Apache-2.0 -->
# Flowcat quickstart

From `git clone` to "I've watched the runtime move real audio and drive a
conversation" in about five minutes — no credentials, no accounts, no cloud.

**What this gets you:** a built Flowcat, the `FrameProcessor` pipeline moving audio
end-to-end, a real WebSocket media round-trip, and your conversation policy driven
from Python over the `RemoteBrain` seam.

**Then, with one provider key:**
[step 5](#5-talk-to-a-real-agent-in-your-browser-no-rust) runs a *real* agent you
talk to in your browser — **no Rust** — via the `flowcat-server` binary + a YAML
config. A live **PSTN** call adds your own *embedder* (carrier + control plane);
[step 6](#6-carry-a-pstn-call-with-your-own-embedder) shows what that piece is and
where the live-verified path starts.

Everything in steps 1–4 is exercised in CI, so it runs on the first try.

## Prerequisites

- A recent **stable Rust toolchain** ([rustup](https://rustup.rs)) — `cargo` on your `PATH`.
- **Python 3** (standard library only) for step 4 — no `pip install`.
- `git`.

## 1. Clone & build

```bash
git clone https://github.com/AreevAI/flowcat.git
cd flowcat
cargo build -p flowcat-cli      # default features → no provider/network deps
```

The default build pulls **no** provider client dependencies — every STT/TTS/LLM,
transport, and exporter is an opt-in Cargo feature. The first build compiles the
workspace (a minute or two); after that, runs are instant.

## 2. Watch the pipeline move audio

```bash
cargo run -p flowcat-cli -- pipeline
```

A synthetic 440 Hz sine wave is pumped through a composable `FrameProcessor` graph
(`Source → Echo → Tap → Sink`) while a `FrameObserver` counts frames:

```
flowcat pipeline demo
  source        : 440 Hz sine, 16000 Hz mono, 320-sample frames
  audio         : 50 frames (~1.00 s)
  chain         : Source -> Echo -> Tap -> Sink
  frames in     : 50 (InputAudio observed)
  frames out    : 50 (OutputAudio observed)
  echoed        : 50 (counted in Echo)
  wall time     : 2.071 ms
  result        : OK (in == out == sourced)
```

This is Flowcat's core: each stage is its own tokio task behind a bounded channel
(natural backpressure), and the hot audio frame is an `Arc` — each hop moves a
pointer, not a buffer. `in == out == sourced` means nothing was dropped.

## 3. Real audio over the WebSocket transport

```bash
cargo run -p flowcat-cli -- ws-echo
```

This stands up the actual generic WebSocket media transport — the same one a
WS-media carrier connects to — streams PCM frames through it, and echoes them back,
asserting they return byte-for-byte:

```
ws-echo: loopback server listening on ws://127.0.0.1:<port>
ws-echo: stream started (call_id=loopback)
ws-echo: echoed frame 1 (7 samples)
ws-echo: echoed frame 2 (6 samples)
ws-echo: echoed frame 3 (64 samples)
ws-echo: stream stopped after 3 echoed frame(s)
ws-echo: loopback OK — 3 frame(s) round-tripped byte-for-byte (3 echoed server-side)
```

Pass `--connect ws://<host>:<port>` to point the echo at a live peer instead of the
in-process loopback.

## 4. Drive the conversation from Python

You don't have to write Rust to control the agent. Flowcat consults a "brain" at
**turn granularity** (between turns) — your Python never touches the
per-audio-frame path, so the runtime's latency profile is unaffected. Start the
pure-stdlib reference server:

```bash
python3 examples/python-remote-brain/brain_server.py   # http://127.0.0.1:8080
```

In another terminal, play the role Flowcat plays on a call — start a session, then
interpret a model tool call:

```bash
curl -s -X POST http://127.0.0.1:8080/session \
  -H 'Content-Type: application/json' \
  -d '{"brain_config":{"graph":"demo"},"provider":"gemini"}'
```

```json
{ "system_prompt": "You are a friendly receptionist. Greet the caller and ask how you can help.",
  "tools": [ { "name": "book_appointment", "...": "..." }, { "name": "end_call", "...": "..." } ],
  "node_id": "greeting", "collected_vars": {} }
```

```bash
curl -s -X POST http://127.0.0.1:8080/tool-call \
  -H 'Content-Type: application/json' \
  -d '{"node_id":"greeting","tool":{"name":"book_appointment","args":{"day":"Tuesday"}},"collected_vars":{}}'
```

```json
{ "action": "transition",
  "system_prompt": "Confirm the appointment day with the caller, then ask them to say 'confirm'.",
  "tools": [ { "name": "confirm", "...": "..." }, { "name": "end_call", "...": "..." } ],
  "say": "Sure — booking you for Tuesday. Shall I confirm?",
  "node_id": "confirm", "collected_vars": { "requested_day": "Tuesday" }, "finished": false }
```

That's the whole `RemoteBrain` wire contract: `/session` seeds state, and
`/tool-call` returns one of `transition` / `stay` / `end`. Replace the `decide()`
function in `brain_server.py` with your own logic — an LLM call, a DB lookup, a
state machine. A Rust embedder wires this in with `RemoteBrain::connect(...)`; see
[`examples/python-remote-brain`](examples/python-remote-brain). To expose Python
*functions* as model tools instead, see
[`examples/python-mcp-tools`](examples/python-mcp-tools).

## 5. Talk to a real agent in your browser (no Rust)

Steps 1–4 are credential-free. To run a *real* agent end-to-end with **no Rust**,
use the **`flowcat-server`** binary: define the agent in a YAML config and serve
it; the browser playground (`--features webrtc`) lets you talk to it directly.

You need one provider key — the live-verified path is **Gemini Live**, so set
`GOOGLE_API_KEY` (a free key from [Google AI Studio](https://aistudio.google.com/)
works):

```bash
cargo build --release -p flowcat-server --features webrtc
GOOGLE_API_KEY=… ./target/release/flowcat-server --config deploy/agent.example.yaml
```

Open **<http://localhost:6210/>**, allow the microphone, click **Start call** —
and you're talking to the agent defined in
[`deploy/agent.example.yaml`](deploy/agent.example.yaml): a node/edge graph you
edit, with no control plane and no database. The live transcript renders as you
speak. The server resolves providers **by name** from the config and reads their
keys from the environment; the full schema is in `flowcat-server/src/config.rs`.

> Prefer telephony? Point a Plivo number's answer webhook at the server's
> `/telephony/answer/plivo/{run_id}` and it bridges the media WebSocket. Or run
> the whole thing in Docker: `docker compose -f deploy/docker-compose.yml up
> --build` — see [`deploy/README.md`](deploy/README.md).

## 6. Carry a PSTN call with your own embedder

For full control — your own routing, control plane, and in-process brain logic —
you write a small **embedder**: a host binary that

- **terminates the call** — the native in-process `SipTransport` (SIP/RTP, no
  softswitch required), or a carrier WebSocket transport if you already run one;
- **resolves & finalizes the call** — your `SessionSource`, talking to your control
  plane (routing, auth, recording/transcript upload);
- **supplies the brain** — your own `AgentBrain`, or the `RemoteBrain` from step 4.

Flowcat owns the media loop; you own the contract, routing, and credentials — which
is what keeps the whole call on infrastructure you control. The combination
verified end-to-end today is **Gemini Live (speech-to-speech) + Plivo** telephony,
so start there. The trait seams and full call lifecycle are specified in
[`DESIGN.md`](DESIGN.md); the provider/transport surface and the "use it from
Python" model are in the [`README`](README.md).

> **Fully on-prem / air-gapped?** Swap the cloud providers for the local connectors
> (Whisper STT; Kokoro / Piper / XTTS TTS; Ollama LLM) and no call audio ever leaves
> your infrastructure.

## Troubleshooting

- **`cargo: command not found`** — install Rust via [rustup](https://rustup.rs) and
  reopen your shell.
- **First build is slow** — that's the one-time dependency compile; re-runs are instant.
- **Port 8080 already in use** — change `PORT` near the top of `brain_server.py`.
- **Run the full offline test suite** — `cargo test` (no network, no credentials).
