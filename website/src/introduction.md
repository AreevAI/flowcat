# Flowcat

**A native-Rust runtime for real-time voice agents — built to run on your own
infrastructure.** Flowcat carries a phone or WebRTC call through a composable
media pipeline — transport in → VAD / turn-taking → STT · LLM · TTS (or a single
speech-to-speech model) → transport out — as **one self-contained binary you
deploy in your own VPC** (or fully air-gapped). No hosted control plane, no
phone-home, no Python or FreeSWITCH sidecar to operate. You bring your own
provider credentials; a call's audio and data never leave infrastructure you
control.

It is a clean-room, native-Rust counterpart to the design of
[pipecat](https://github.com/pipecat-ai/pipecat): the same `FrameProcessor`
pipeline model and the same provider breadth, packaged for teams that need to
**own the stack** — self-hosted, auditable, and dense enough to run serious call
volume per box.

**License:** Apache-2.0 · **Status:** pre-1.0, building in the open.

> **New here?** The [Quickstart](./quickstart.md) takes you from `git clone` to a
> running pipeline, a real WebSocket audio round-trip, and a Python-driven brain
> in about five minutes — no credentials.

## Where to go next

**Building on Flowcat?** Follow the path in order:

1. **[Quickstart](./quickstart.md)** — clone → build → watch real audio move.
2. **[Build an embedder](./embedder.md)** — the host binary that carries a call.
3. **[Configuration](./configuration.md)** — runtime knobs and credentials.
4. **[Providers & features](./features.md)** — the STT / TTS / LLM / transport surface.
5. **[Deployment](./deployment.md)** — ship a release binary in your own VPC.

**Contributing to Flowcat?** Start with
**[Contributing](./contributing.md)** (build, test, add a provider) and the
architecture docs beside it.

> This site is generated from the Markdown in the
> [Flowcat repository](https://github.com/AreevAI/flowcat) with
> [mdBook](https://rust-lang.github.io/mdBook/).
