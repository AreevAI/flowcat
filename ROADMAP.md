<!-- SPDX-License-Identifier: Apache-2.0 -->
# Flowcat roadmap

This is a forward-looking, best-effort roadmap — directional, not a commitment.
Shipped capabilities are described in the [`README`](README.md) and
[`FEATURES.md`](FEATURES.md); this file tracks what is *not yet* done.

## Recently landed

- **Composable `FrameProcessor` pipeline** — typed frame taxonomy, system-frame
  priority/interruption model, linear + parallel pipelines. (Frozen API in
  [`PROCESSOR-DESIGN.md`](PROCESSOR-DESIGN.md).)
- **In-process native SIP/RTP/SDP** user-agent for G.711 telephony.
- **~80 STT/TTS/LLM + realtime connectors**, each a `dep:`-gated Cargo feature
  ([`FEATURES.md`](FEATURES.md)).
- **Runnable demos** in `flowcat-cli`: an in-process pipeline demo and a
  WebSocket PCM echo-bot (`pipeline`, `ws-echo`).
- **Use it from Python (out-of-process)** — the `RemoteBrain` HTTP adapter
  (`brain-http`) drives conversation policy from a Python service, and the `mcp`
  client exposes Python functions as tools. Reference servers in
  [`examples/`](examples/). This is the supported Python path today.

## Planned

### Python bindings (PyO3)
The out-of-process path above already lets Python drive Flowcat without writing
Rust. The next step is **in-process** bindings: `import flowcat`, build a pipeline
in Python, and pass Python callables/objects as the brain — single process, no
service to host. The binding will keep Python at **turn granularity** (releasing
the GIL during the Rust media loop) so the tail-latency guarantees hold. This is a
developer-ergonomics step, not a capability gap. **Not yet started.**

### WebRTC browser transport
Complete the `str0m`-based WebRTC transport (+ Opus) so a browser client can
connect directly, alongside the existing carrier/WebSocket paths.

### Local device backend
The `local` mic/speaker transport currently ships its shape as a stub (no device
backend, to keep the default build free of a heavy platform audio dependency).
Add an optional, feature-gated device backend so the local demo can capture and
play real audio.

### Broader live-verified provider coverage
Connectors are currently **fixture/wire-tested** — each provider's message
framing is pinned by unit tests, but end-to-end live calls require that vendor's
credentials and are not exercised in CI. Expand the set of provider paths that
have been verified against the live service.

### LiveKit transport
The `livekit` transport is currently a stub; wire up real signaling.

## How to influence the roadmap

Open an issue or start a discussion. Adding a new connector is usually a small,
self-contained contribution — see [`CONTRIBUTING.md`](CONTRIBUTING.md) and the
provider-family taxonomy in [`PROVIDERS.md`](PROVIDERS.md).
