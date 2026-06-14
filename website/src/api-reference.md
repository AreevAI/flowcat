# API reference

Flowcat is a set of Rust crates; the canonical, type-level API reference is
**rustdoc**.

> **Not yet on crates.io / docs.rs.** Flowcat is pre-1.0 and not yet published, so
> there is no docs.rs page yet. Until then, generate the docs locally — it takes
> seconds. (When published, the hosted link will live here.)

## Generate the docs locally

```bash
git clone https://github.com/AreevAI/flowcat.git
cd flowcat

# Core seams (no provider features needed):
cargo doc --no-deps -p flowcat-core --open

# Include the provider/transport surface (enable the features you care about):
cargo doc --no-deps \
  -p flowcat-core -p flowcat-services -p flowcat-transports -p flowcat-telephony \
  --features "flowcat-services/realtime-all,flowcat-transports/ws"
```

Feature-gated items only appear in rustdoc when their feature is enabled — pass
the same features you'd build with (see [Deployment](./deployment.md#1-build-a-release-binary)).

## Crate map

| Crate | What it exposes |
|---|---|
| [`flowcat-core`](https://github.com/AreevAI/flowcat/tree/main/flowcat-core) | The seams and runtime: `AgentBrain`, `SessionSource`, `MediaTransport`, `RealtimeLlm` / `RealtimeKickoff`, the `FrameProcessor` graph + `Frame` taxonomy, `build_s2s_task` / `build_cascaded_task`, native `SipAgent` / `SipTransport`, and the `GeminiLive` backend |
| [`flowcat-services`](https://github.com/AreevAI/flowcat/tree/main/flowcat-services) | STT / TTS / LLM / realtime provider adapters (feature-gated), observability exporters, the `RemoteBrain` HTTP adapter (`brain-http`) |
| [`flowcat-transports`](https://github.com/AreevAI/flowcat/tree/main/flowcat-transports) | WebRTC (`webrtc-str0m`), WebSocket (`ws`), and other media transports |
| [`flowcat-telephony`](https://github.com/AreevAI/flowcat/tree/main/flowcat-telephony) | Carrier serializers (Plivo, Twilio, Telnyx, …) and DTMF |
| [`flowcat-cli`](https://github.com/AreevAI/flowcat/tree/main/flowcat-cli) | The credential-free `flowcat` demo binary (`pipeline`, `ws-echo`) |

## Starting points

- **Writing an embedder?** → [Build an embedder](./embedder.md), then the seam
  sources linked there.
- **Architecture & call lifecycle?** → [Design overview](./design.md) and
  [Frame processor model](./processor-design.md).
- **Which providers exist?** → [Providers & connectors](./providers.md).
