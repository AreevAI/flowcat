<!-- SPDX-License-Identifier: Apache-2.0 -->
# Contributing to Flowcat

Flowcat is Apache-2.0. By contributing you agree your contribution is licensed
under those terms. Every source file carries the SPDX header
`// SPDX-License-Identifier: Apache-2.0` as its first line — keep that on any new
file.

By participating you also agree to abide by our
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). Security issues should be reported
privately — see [`SECURITY.md`](SECURITY.md).

This guide covers the two things a contributor most often does: **add a provider**
and **write a processor**. Read [`PROCESSOR-DESIGN.md`](PROCESSOR-DESIGN.md)
(the frozen pipeline API) and [`DESIGN.md`](DESIGN.md) (the runtime + trait seams)
first.

---

## Building and running tests

**Prerequisites:** a stable Rust toolchain ([rustup](https://rustup.rs)); Python 3
(standard library only) for the `examples/`. The default build needs nothing else.

```bash
git clone https://github.com/AreevAI/flowcat.git && cd flowcat
cargo build -p flowcat-cli     # default features — no provider/network deps
cargo test                     # the full OFFLINE suite: no network, no credentials
```

`cargo test` is the green bar below — pure encode/decode fixtures, hermetic.

**Live integration tests** exercise a real provider over the network. They live in
`#[cfg(test)]` blocks (most `#[ignore]`d, so they're skipped by default) and read
their credentials from the environment. These are **test credentials, not
deployment configuration** — in production the runtime never reads provider keys
from the environment; an embedder passes them to each service constructor.

Each provider follows a `PROVIDER_API_KEY` (+ optional `PROVIDER_VOICE_ID` /
`PROVIDER_MODEL`) convention; representative variables:

| Variable | Used by |
|---|---|
| `OPENAI_API_KEY` | OpenAI STT / TTS / LLM / Realtime |
| `ANTHROPIC_API_KEY` | Anthropic LLM |
| `GEMINI_API_KEY` | Gemini Live realtime |
| `GEMINI_LIVE_MODEL` | override the Gemini Live model id in tests |
| `DEEPGRAM_API_KEY` | Deepgram STT / TTS |
| `CARTESIA_API_KEY`, `CARTESIA_VOICE_ID` | Cartesia STT / TTS |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` | Bedrock / Transcribe / Polly / Nova Sonic |
| `XTTS_BASE_URL`, `KOKORO_BASE_URL`, `PIPER_BASE_URL`, `WHISPER_MODEL_PATH` | local connectors |

The authoritative list is the `#[cfg(test)]` blocks in each provider module under
[`flowcat-services/src`](flowcat-services/src). Run one live test by name with its
credentials set:

```bash
GEMINI_API_KEY=… cargo test -p flowcat-core -- --ignored gemini_live
```

---

## The bar (every PR)

- `cargo build` (default features) compiles and pulls **no** new networked
  dependency into the default build.
- `cargo test` is green — **offline**. Tests must not hit the network; provider
  tests are pure encode/decode fixtures (see below).
- `cargo clippy --all-targets` is clean. The
  only sanctioned escape is `#[allow(dead_code)]` on a held-but-not-yet-wired
  config field of a stub.
- New crypto / auth / signing paths (e.g. a new SigV4 or signature provider) get
  an independent **known-answer test** and an extra reviewer pass.

---

## Adding a provider (STT / TTS / LLM / realtime)

Providers live in `flowcat-services` (one per file), each behind its own Cargo
feature. The catalogue is organised by **protocol family**, and a family's real
client is implemented **once** ([`PROVIDERS.md`](PROVIDERS.md) §0).

**Triage your provider first:**

- **(D) Distinct client** — its own wire protocol (own WS/HTTP framing, own auth,
  own message schema). Write a real client: the network transport + a **pure
  encode/decode seam** + unit tests. Templates: Deepgram (STT-WS),
  Cartesia (TTS-WS), OpenAI (LLM/Whisper-HTTP).
- **(W) Thin wrapper** — protocol-compatible with an existing (D) family member
  (OpenAI-compatible, Whisper-HTTP, OpenAI-Realtime, …), differing only in
  `base_url` + auth header + default model. Write a ~30-line struct that
  constructs the family's (D) client with that config and delegates the trait.
  Template: any of the `llm-*` OpenAI wrappers.

**The mechanics:**

1. Implement the trait seam for your category (`Stt` / `Tts` / `Llm` /
   `RealtimeLlm`, defined in `flowcat-core`) in `flowcat-services/src/<cat>/<name>.rs`.
2. Add a `dep:`-gated Cargo feature in `flowcat-services/Cargo.toml`. A (W) just
   enables its (D) family feature (e.g. `llm-groq = ["llm-openai"]`). A (D) pulls
   only its client dep (`reqwest`, `tokio-tungstenite`, `tonic`, …). **No new dep
   may land in the default build** — `default = []` in `flowcat-services`.
3. Register the `mod` + `pub use` in the category's `mod.rs`, and add your feature
   to the relevant `*-all` umbrella so the CLI/CI fat build covers it.
4. **Write the fixture test.** This is the coverage bar: a pure function that
   builds your provider's outbound request frame(s) and parses a recorded
   inbound response, asserting on the exact bytes/JSON — **no live socket**. For a
   SigV4 provider (AWS Bedrock/Transcribe/Polly), pin a **known-answer** signing
   test against the AWS-published vectors. Live verification needs the vendor's
   credentials and is out of scope for CI.

A "not yet wired" stub returns `FlowcatError::Other("<provider>: not yet wired")`
and still compiles + passes — that's the floor, a real PR replaces it with the
client + fixtures.

---

## Writing a processor (the contract every author must know)

Each processor is a `FrameProcessor` (`flowcat-core/src/processor/`). The
framework owns the per-processor tokio task, the bounded/priority channels, and
the lifecycle; you write `process_frame` and optionally `start`/`stop`.

**The one contract that surprises people — lifecycle/system frames bypass
`process_frame`** ([`PROCESSOR-DESIGN.md`](PROCESSOR-DESIGN.md) §2.1–§2.3):

- `Start` → the framework calls your `async fn start(&mut self, setup, params)`
  (open sockets, spawn provider reader tasks here), then forwards the frame. It
  does **not** reach `process_frame`.
- `End` / `Stop` / `Cancel` → the framework calls your `async fn stop(&mut self,
  reason)` (flush + close), then forwards. Also **not** via `process_frame`.
- `Interruption` and other **System** frames ride an unbounded priority channel
  and are drained ahead of data/control by a `biased` select; the task loop
  handles interruption (draining interruptible queued frames, keeping
  uninterruptible ones, cancelling an in-flight interruptible `process_frame`).

So **`process_frame` only ever sees Data/Control frames.** Do not put socket
open/close in `process_frame` — it will never run for the lifecycle frames that
should trigger it. Other rules:

- **`process_frame` must not block.** Long work (a provider round-trip) is driven
  by an internally-spawned task that feeds results back as frames — the Gemini
  reader-task pattern (`flowcat-core/src/realtime/gemini_live.rs`).
- Push results via the `Link` (`push` / `push_down` / `push_error` / `broadcast`),
  never by calling another processor directly.
- The hot audio frame is `Arc<AudioFrame>` — clone the Arc, don't copy PCM.
- An `Err` returned from `process_frame` becomes an upstream non-fatal `Error`
  frame; return `Err` for recoverable faults, set `fatal` for terminal ones.
- A pure observer/no-op processor is the default `process_frame` (forward
  unchanged) — don't override what you don't need.

Flowcat stays **contract-agnostic**: keep any embedder/control-plane knowledge
out of `flowcat-core`. Conversation decisions and call bootstrap/finalize are the
`AgentBrain` / `SessionSource` trait seams an embedder implements; the runtime
treats `brain_config` as opaque bytes ([`DESIGN.md`](DESIGN.md)).

---

## Feature-flag discipline (no default-build cost)

- `flowcat-core` default = `["sip", "recorder"]` — no HTTP/gRPC/ONNX. The only
  optional core deps are `ort` (`vad-ort`) and `nnnoiseless` (`filter-rnnoise`).
- `flowcat-services` / `flowcat-transports` default = `[]`. Every provider and
  transport is `dep:`-gated; the default build links **none** of their clients.
- `flowcat-telephony` default = `["plivo"]` (serializers are deps-free flags).
- Adding a provider must not move any dependency out of `optional`/`dep:` gating.
  If a reviewer sees `cargo tree` on the default build grow, the PR is wrong.

The exhaustive matrix is [`FEATURES.md`](FEATURES.md) — keep it in sync when you
add a feature.

---

## Review

Substantive PRs get a code review; anything touching auth, signing, or signature
verification also gets a security review. Be honest in the PR about what is
**fixture-tested** vs **live-verified** — overclaiming a live-working provider is
the thing reviewers push back on hardest.
