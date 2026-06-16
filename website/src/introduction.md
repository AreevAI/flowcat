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

<!-- Overview video — replace VIDEO_ID with the YouTube id (the part after
     youtu.be/ or watch?v=). Rendered by mdBook on the docs home page; GitHub
     strips iframes from README.md, which is why this lives in the site source. -->
<div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;max-width:960px;margin:1.5rem auto;border-radius:8px;">
  <iframe style="position:absolute;top:0;left:0;width:100%;height:100%;border:0;"
    src="https://www.youtube-nocookie.com/embed/VIDEO_ID"
    title="Flowcat — overview"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    referrerpolicy="strict-origin-when-cross-origin"
    allowfullscreen></iframe>
</div>

**Status:** pre-1.0, building in the open.

> **New here?** The [Quickstart](./quickstart.md) goes from `git clone` to a
> running pipeline and a real audio round-trip in about five minutes (no
> credentials), then to a **real agent you talk to in your browser** — defined in
> YAML and run with one binary (`flowcat-server`), no Rust required.

## Where to go next

**Building on Flowcat?** Follow the path in order:

1. **[Quickstart](./quickstart.md)** — clone → build → watch real audio move, then
   talk to a real agent in your browser (`flowcat-server`, no Rust).
2. **[Build an embedder](./embedder.md)** — the host binary that carries a call,
   when you need more than the config-driven server.
3. **[Configuration](./configuration.md)** — runtime knobs and credentials.
4. **[Providers & features](./features.md)** — the STT / TTS / LLM / transport surface.
5. **[Deployment](./deployment.md)** — ship a release binary (or `flowcat-server`) in your own VPC.

**Contributing to Flowcat?** Start with
**[Contributing](./contributing.md)** (build, test, add a provider) and the
architecture docs beside it.

> This site is generated from the Markdown in the
> [Flowcat repository](https://github.com/AreevAI/flowcat) with
> [mdBook](https://rust-lang.github.io/mdBook/).
