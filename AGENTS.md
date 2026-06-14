<!-- SPDX-License-Identifier: Apache-2.0 -->
# AGENTS.md — guide for AI coding agents (and new contributors)

Flowcat is a self-hosted, native-Rust runtime for real-time voice agents: a Cargo
workspace of four library crates plus a demo CLI. This file is the fast
orientation for an automated coding agent or a first-time contributor.

The authoritative docs win if they disagree with this file (and then update this
file): [`CONTRIBUTING.md`](CONTRIBUTING.md), [`PROCESSOR-DESIGN.md`](PROCESSOR-DESIGN.md)
(the frozen pipeline API), [`DESIGN.md`](DESIGN.md) (runtime + trait seams),
[`FEATURES.md`](FEATURES.md) (the feature matrix), and [`PROVIDERS.md`](PROVIDERS.md)
(the provider protocol-family map).

## Workspace

- `flowcat-core/` — framework: `Frame`, `FrameProcessor`, `Pipeline`/`Task`/`Runner`,
  audio codec/resample/recorder, native SIP/RTP/SDP, Gemini Live, all trait seams.
- `flowcat-services/` — STT/TTS/LLM/realtime providers + observability exporters +
  MCP, one Cargo feature each.
- `flowcat-transports/` — WebRTC (`str0m`) / WebSocket / Daily / LiveKit / local.
- `flowcat-telephony/` — carrier serializers (Twilio/Telnyx/Plivo/…) + DTMF.
- `flowcat-cli/` — the `flowcat` demo binary.
- `bench/`, `bench-rs/` — the reproducible pipecat-vs-flowcat benchmark kit.

## Build & test (offline, no credentials)

```bash
cargo build -p flowcat-cli     # default features — pulls NO provider/network deps
cargo test                     # full suite: hermetic, no network, no credentials
cargo fmt --all                # CI gates on `cargo fmt --all --check`
cargo clippy --workspace --all-targets -- -D warnings   # CI denies warnings
```

`cargo run -p flowcat-cli -- pipeline` and `... ws-echo --loopback` are the two
credential-free demos. Live provider tests are `#[ignore]`d and read keys from the
environment (a `PROVIDER_API_KEY` convention) — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Golden rules (what agents most often get wrong)

1. **Every new source file starts with the SPDX header** —
   `// SPDX-License-Identifier: Apache-2.0` (`<!-- … -->` in markdown).
2. **The default build stays dependency-free.** Every provider/transport/exporter is
   one `dep:`-gated Cargo feature; `flowcat-services`/`flowcat-transports` default to
   `[]`. If `cargo tree` on the default build grows, the change is wrong.
3. **Tests are offline.** Provider tests are pure encode/decode fixtures asserting on
   exact bytes/JSON — never a live socket. A new auth/signing path (e.g. SigV4) needs
   a known-answer test.
4. **Don't break the frozen API** ([`PROCESSOR-DESIGN.md`](PROCESSOR-DESIGN.md)
   §2.1–§2.3): lifecycle/system frames (`Start`/`Stop`/`Cancel`/`Interruption`) bypass
   `process_frame` — use `start`/`stop`; `process_frame` must not block; push results
   via the `Link`, never call another processor directly; the hot frame is
   `Arc<AudioFrame>` (clone the Arc, don't copy PCM).
5. **Keep `flowcat-core` contract-agnostic** — no embedder/control-plane/SQL/wire
   knowledge there; that lives behind the `AgentBrain` / `SessionSource` /
   `MediaTransport` trait seams an embedder implements.
6. **Comments and docs are public.** Don't reference internal-only tickets, decision
   records, or work-stream labels; write for an outside reader.
7. **Be honest about provider status** — "fixture-tested" ≠ "live-verified". Don't
   overclaim a working provider in code comments or the PR.

## Add a provider (the most common PR)

Triage first ([`PROVIDERS.md`](PROVIDERS.md) §0): a **(D)** distinct client (its own
wire protocol → a real client + a pure encode/decode seam + fixtures) or a **(W)**
thin wrapper (OpenAI-compatible etc. → a ~30-line struct that constructs the (D)
family client with a different `base_url`/auth and delegates). Then:

1. Impl the category service trait (`SttService` / `TtsService` / `LlmService` /
   `RealtimeLlmService`, in `flowcat-core::service`) in
   `flowcat-services/src/<cat>/<name>.rs`.
2. Add a `dep:`-gated feature in `flowcat-services/Cargo.toml` (a (W) just enables its
   (D) family feature, e.g. `llm-groq = ["llm-openai"]`); register `mod` + `pub use`
   in the category `mod.rs`; add the feature to the relevant `*-all` umbrella; keep
   [`FEATURES.md`](FEATURES.md) in sync.
3. Write the fixture test. A "not yet wired" stub returns
   `FlowcatError::Other("<provider>: not yet wired")` and still compiles + passes —
   that is the floor; a real PR replaces it with the client + fixtures.

Full detail (templates, the per-PR bar, the processor-author contract):
[`CONTRIBUTING.md`](CONTRIBUTING.md).
