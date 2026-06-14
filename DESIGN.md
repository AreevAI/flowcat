# Flowcat — design

**Flowcat** is a native-Rust real-time voice-agent runtime built to run on
infrastructure you own: it carries a phone/WebRTC call's audio, runs a
speech-to-speech model, and lets a pluggable "brain" drive the conversation — all
as **one self-contained binary** you deploy in your own VPC (or air-gapped), with
no hosted control plane and no Python or FreeSWITCH sidecar. Because the media
loop is one `tokio` process (no GIL, no FFI), that single binary also holds
serious call volume per box. It is an **isolated, Apache-2.0 cargo workspace**
designed to be embedded by a host application.

> Motivation + benchmark: `bench/RESULTS.md` — one Flowcat process holds a flat
> p99 from 10 to 2,000 concurrent calls where a Python deployment grows a
> multi-second tail and needs a worker fleet. Read this as runtime **capacity and
> reliability headroom**, not conversational latency (which your STT/LLM/TTS
> providers dominate). Native SIP is the current approach — softswitch optional,
> superseding the earlier FreeSWITCH gateway design. Native SIP details:
> `SIP-DESIGN.md`.

> **Scope of this document.** This is the architecture-and-trait-seam spec, written
> at the first milestone (Plivo WS-media + native SIP, with **Gemini Live** as the
> first speech-to-speech brain). The **seams below are unchanged**, but the runtime
> has broadened since:
>
> - **Two pipeline shapes**, not just S2S — a single speech-to-speech model
>   (`RealtimeLlm`, e.g. Gemini Live) *or* a cascaded **STT → LLM → TTS** pipeline.
>   Both are built from `flowcat-core::pipeline` (`build_s2s_task` /
>   `build_cascaded_task`).
> - **~80 STT/TTS/LLM + realtime providers**, 5 transports, and 9 telephony
>   serializers, now split into sibling crates (`flowcat-services`,
>   `flowcat-transports`, `flowcat-telephony`), each behind one Cargo feature — see
>   the [`README`](README.md) crate map + connector table and [`FEATURES.md`](FEATURES.md).
> - **Fully local / air-gapped** by swapping in the local connectors (Whisper STT;
>   Kokoro / Piper / XTTS TTS; Ollama LLM) — no call audio leaves your infrastructure.
> - **Python without Rust** — the `RemoteBrain` HTTP adapter drives the `AgentBrain`
>   seam from a Python service (see [`QUICKSTART.md`](QUICKSTART.md) and
>   [`examples/`](examples/)).
>
> Treat the [`README`](README.md), [`FEATURES.md`](FEATURES.md), and
> [`PROCESSOR-DESIGN.md`](PROCESSOR-DESIGN.md) as authoritative for the *current*
> surface; this doc remains the reference for the **trait seams and call lifecycle**,
> which are stable. Concrete provider/crate names below (e.g. "the Gemini Live
> client") are the milestone's first implementation, not the limit of what ships.

## Goal of the first milestone

A telephony call works **end to end** through Flowcat for two carrier styles:

- **WebSocket-media carriers (e.g. Plivo)** — audio already arrives over a WS. Pure-Rust path,
  live-testable without extra infra. This is the *easy* path and the integration baseline.
- **SIP/RTP-only carriers** — no provider WS-media. Flowcat speaks **SIP/RTP
  natively** (no softswitch): a `SipAgent` in `flowcat-core` REGISTERs the carrier's trunk and
  terminates INVITE/RTP in-process, and a `SipTransport` presents the call to the pipeline
  through the same `MediaTransport` seam the WS path uses. This is the **native-SIP** decision
  — one single Rust binary, no FreeSWITCH/`mod_audio_stream` gateway. (The earlier
  FreeSWITCH gateway approach is superseded — see `SIP-DESIGN.md`.)

The first milestone's brain is **native Rust Gemini Live** (the speech-to-speech path
live-verified end to end); a cascaded **STT → LLM → TTS** pipeline is the other supported
shape. Either way the conversation logic lives behind the `AgentBrain` trait — the embedder's
own engine linked as an **rlib** (no PyO3 FFI), or the ready-made `RemoteBrain` HTTP adapter
driving it from a Python service.

## Two-plane fit + where Flowcat sits

```
 PSTN ─SIP/RTP ───────────────────────> SipAgent (in flowcat-core, runs in the embedder) ─┐
 PSTN ─Plivo <Stream> WS ─> host WS ─> WsCarrierTransport ────────────────────────────────┤
                                                                                          ▼
                          ┌──────────── the embedder (binary, host workspace) ──────────┐
                          │  HTTP: /telephony/ws/{provider}/{run} · answer-XML · health  │
                          │  runs flowcat SipAgent (SIP trunk REGISTER) + control plane  │
                          │  adapts its inbound WS → MediaSocket → WsCarrierTransport     │
                          │  impl AgentBrain  (→ the host's engine rlib, NO PyO3)        │
                          │  impl SessionSource (→ the host's control-plane API)         │
                          └───────────────────────────┬─────────────────────────────────┘
                                                       │ uses
                          ┌──────────── flowcat-core (lib, OSS workspace) ──────────────┐
                          │  MediaTransport seam: SipTransport (SIP/RTP) · WsCarrier+    │
                          │    MediaSerializer(plivo) · RealtimeLlm(GeminiLive)          │
                          │  Call pipeline · codec · recorder · sip/ (SipAgent, RTP/SDP) │
                          │  traits: AgentBrain·SessionSource·MediaTransport·RealtimeLlm │
                          └──────────────────────────────────────────────────────────────┘
                          shared database / object store (via the embedder's control plane only)
```

`flowcat-core` knows **nothing** about the embedder, web routing, SQL, or the wire contract.
The embedder-specific glue (engine adapter, control-plane client, auth/routing) lives in the
consumer crate.

## Crate layout

> The tree below is the milestone's **core-centric** view (providers, transports, and
> serializers shown inside `flowcat-core`). They have since been split into the sibling
> crates `flowcat-services` / `flowcat-transports` / `flowcat-telephony`; the
> [`README`](README.md) crate map is the current authority. The `flowcat-core` module
> seams shown here are still accurate.

```
flowcat/                         # ← own cargo workspace (Apache-2.0)
  Cargo.toml                     # [workspace] members = ["flowcat-core", "flowcat-cli", ...]
  DESIGN.md  LICENSE
  flowcat-core/                  # the runtime library
    src/
      lib.rs
      frame.rs                   # AudioFrame, ControlEvent, etc.
      error.rs                   # FlowcatError
      codec.rs                   # g711 ↔ pcm16, resample (rubato)
      audio.rs                   # AudioRecorder (mono mix → WAV bytes)
      transport/{mod.rs, media.rs, carrier.rs, socket.rs, ws_media.rs}  # MediaTransport seam + WsCarrierTransport
      serializer/{mod.rs, plivo.rs}
      sip/{mod.rs, agent.rs, transport.rs, rtp.rs, sdp.rs}  # native SIP UA + SipTransport (rsipstack + RTP/SDP/jitter)
      realtime/{mod.rs, gemini_live.rs}
      brain.rs                   # trait AgentBrain + ToolDecl + BrainAction
      session.rs                 # trait SessionSource + ResolvedCall + Usage
      pipeline.rs                # Call::run(...) — the orchestration loop
      transcript.rs              # transcript collector
  flowcat-cli/                   # example: local-mic / ws demo (DX), embedder-agnostic
  bench/  bench-rs/              # (existing benchmark kit)

# the embedder (lives in the host's own workspace):
#   glue binary that runs the SipAgent + the control-plane originate endpoint,
#   a telephony provider, and the carrier routes / inbound-resolve / originate.
```

The embedder's `Cargo.toml` path-deps (or git/crates.io-deps) `flowcat-core` and links its
own engine. Cross-workspace deps are fine — flowcat-core builds in each consumer's graph and
standalone via its own lockfile.

## Trait contracts (the seams everything plugs into)

All async traits use `async_trait`. Audio is 16-bit little-endian mono PCM internally;
sample rate is explicit on every buffer.

```rust
// frame.rs
pub struct AudioChunk { pub pcm: Vec<i16>, pub sample_rate: u32 } // mono

// transport/media.rs — THE pipeline seam. The pipeline never cares whether audio
// arrived as carrier WS frames or as RTP. SipTransport and WsCarrierTransport both impl it.
#[async_trait] pub trait MediaTransport: Send {
    async fn recv(&mut self) -> Option<MediaIn>;       // StreamStart{call_id} | Audio(@carrier_rate) | Stop
    async fn send_audio(&mut self, chunk: AudioChunk) -> Result<(), FlowcatError>; // bot audio out
    async fn send_clear(&mut self) -> Result<(), FlowcatError>;                    // barge-in flush (no-op for RTP)
    fn carrier_rate(&self) -> u32;                                                 // 8000 for telephony G.711
}
pub enum MediaIn { StreamStart { call_id: String }, Audio(AudioChunk), Stop }

// transport/socket.rs — WS building block; the host provides the raw socket. Used (with a
// serializer) by `WsCarrierTransport: MediaTransport` for the Plivo path. Native SIP bypasses this.
#[async_trait] pub trait MediaSocket: Send {
    async fn recv(&mut self) -> Option<WsIn>;        // Text(String) | Binary(Vec<u8>) | Close
    async fn send_text(&mut self, s: String) -> Result<(), FlowcatError>;
    async fn send_binary(&mut self, b: Vec<u8>) -> Result<(), FlowcatError>;
}

// serializer/mod.rs — per-carrier WS framing for WsCarrierTransport. Pure (no I/O). plivo only.
pub trait MediaSerializer: Send {
    fn on_message(&mut self, msg: &WsIn) -> SerIn;   // StreamStart{call_id,..} | Audio(AudioChunk) | Stop | Ignore
    fn encode_audio(&self, chunk: &AudioChunk) -> WsOut;   // text/binary to send back
    fn encode_clear(&self) -> Option<WsOut>;               // barge-in / interruption
    fn carrier_rate(&self) -> u32;                          // 8000 for telephony μ-law
}

// realtime/mod.rs — the speech-to-speech model abstraction (GeminiLive first).
#[async_trait] pub trait RealtimeLlm: Send {
    async fn connect(&mut self, setup: RealtimeSetup) -> Result<(), FlowcatError>; // system prompt+tools
    async fn send_audio(&mut self, chunk: AudioChunk) -> Result<(), FlowcatError>; // 16k PCM in
    async fn update_system(&mut self, prompt: String, tools: Vec<ToolDecl>) -> Result<(), FlowcatError>;
    async fn send_tool_result(&mut self, id: String, result: serde_json::Value) -> Result<(), FlowcatError>;
    async fn next_event(&mut self) -> Option<RealtimeEvent>; // AudioOut(24k) | UserText | BotText | ToolCall | Interrupted | Usage | Closed
}

// brain.rs — the conversation decision-maker. The embedder impls this over its own engine.
pub trait AgentBrain: Send {
    fn system_prompt(&self) -> String;
    fn tools(&self) -> Vec<ToolDecl>;           // transitions + endCall (+ later: node tools)
    fn on_tool_call(&mut self, name: &str, args: &serde_json::Value) -> BrainAction;
    fn is_finished(&self) -> bool;
    fn collected_vars(&self) -> serde_json::Value;
}
pub enum BrainAction { Transition { system_prompt: String, tools: Vec<ToolDecl>, say: Option<String> }, Stay, End { disposition: Option<String> } }
pub struct ToolDecl { pub name: String, pub description: String, pub params: serde_json::Value } // JSON-schema params

// session.rs — call bootstrap + finalize. The embedder impls this over its control-plane HTTP.
#[async_trait] pub trait SessionSource: Send + Sync {
    async fn resolve(&self, run_id: i64, token: &str) -> Result<ResolvedCall, FlowcatError>;
    async fn complete(&self, run_id: i64, token: &str, fin: Finalize) -> Result<(), FlowcatError>;
    async fn artifact_upload_url(&self, run_id: i64, token: &str, kind: &str) -> Result<UploadTarget, FlowcatError>;
    async fn put_bytes(&self, url: &str, bytes: Vec<u8>, content_type: &str) -> Result<(), FlowcatError>;
}
pub struct ResolvedCall { pub provider: String, pub brain_config: serde_json::Value, /* graph_spec+runtime+seed */ pub is_completed: bool }
pub struct Finalize { pub usage: serde_json::Value, pub collected_vars: serde_json::Value, pub recording_url: Option<String>, pub transcript_url: Option<String> }
```

`brain_config` is opaque to flowcat-core (it's the embedder's graph/spec + runtime options +
seed vars); the embedder builds its brain from it. Flowcat never sees the contract.

The embedder assembles these seams into a runnable call with one of the two builders in
`flowcat-core::pipeline`: **`build_s2s_task`** (a single `RealtimeLlm` such as Gemini Live) or
**`build_cascaded_task`** (a `MediaTransport` + STT + LLM + TTS chain). Both accept any
`AgentBrain` and drive the same lifecycle below.

## Call lifecycle

The embedder owns the control plane; the shapes below are the typical wiring it provides.

**WS-media inbound (e.g. Plivo):** carrier → the embedder's answer proxy →
inbound-run endpoint (verify sig, route, create run, token) → answer XML with `<Stream>` →
the carrier opens a WS to the embedder's `/telephony/ws/{provider}/{run}?token=` →
`SessionSource.resolve` → the call pipeline runs.

**WS-media outbound:** initiate-call endpoint → run+token → carrier originate (answer_url) →
the carrier GETs answer XML → `<Stream>` WS → same pipeline.

**SIP inbound (native SIP):** PSTN → SIP → the `SipAgent` running inside the embedder
(the trunk is REGISTERed at startup) accepts the INVITE → the embedder resolves the call over its
control plane (DID → workflow → create run+token) → builds a `SipTransport` for the dialog and
runs the pipeline. Carrier CDR/recording webhooks are **side-effect-only**, never the media trigger.

**SIP outbound (native SIP):** initiate-call endpoint → run+token → the control plane POSTs to
the embedder's originate endpoint (`{run_id, token, to_number}`, no ESL) → the `SipAgent` originates
the INVITE to the E.164 → on answer builds a `SipTransport` and runs the pipeline. CallerID is
configured on the trunk by the embedder's SIP trunk configuration.

## Audio path

Telephony is **G.711 μ-law 8 kHz**. Gemini Live wants **16 kHz PCM in**, emits **24 kHz PCM out**.

```
carrier → μ-law decode → 8k→16k upsample (rubato) → Gemini
Gemini → 24k→8k downsample (rubato) → μ-law encode → carrier
recorder taps both legs → mono mix → WAV → object store
```

On the **SIP path** the `SipTransport` decodes inbound RTP (G.711 PCMU/PCMA per the negotiated
codec) straight to 8 kHz PCM and re-encodes the bot leg to RTP — no WS hop, no intermediate L16
framing. On the **Plivo path** the `PlivoSerializer` handles the μ-law WS framing. Both feed the
same resample/recorder. Crates: `audio-codec-algorithms` (G.711), `rubato` (resample), hand mix
for the recorder.

The rates above are Gemini Live's (16 kHz in / 24 kHz out). A **cascaded** pipeline instead
resamples the same 8 kHz carrier audio to whatever each STT/TTS provider expects; the carrier
codec, `rubato` resampling, and the dual-leg recorder are identical either way.

## Gemini Live protocol (native client)

WSS `wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key=<API_KEY>` — **JSON** frames (not protobuf).

- **client→server:** `setup` (model, `systemInstruction`, `tools`, `responseModalities:["AUDIO"]`,
  input/output transcription on) · `realtimeInput.mediaChunks[{mimeType:"audio/pcm;rate=16000", data:b64}]` ·
  `toolResponse.functionResponses` · `clientContent` (kickoff turn).
- **server→client:** `setupComplete` · `serverContent.modelTurn.parts[].inlineData` (24 kHz PCM b64,
  the bot audio) · `serverContent.inputTranscription`/`outputTranscription` · `serverContent.interrupted`
  (barge-in) · `toolCall.functionCalls[{name,args,id}]` · `usageMetadata` · `goAway`/`sessionResumptionUpdate`.

Tools = the brain's transitions as no-arg functions + `endCall`. On `toolCall`: call
`AgentBrain.on_tool_call`; for `Transition`, `RealtimeLlm.update_system(new prompt, new tools)` +
`send_tool_result`; for `End`, drain + finalize. v1 handles `interrupted` (clear carrier audio);
`goAway`/reconnect is a documented follow-up (see ROADMAP.md).

## What the embedder's control plane provides

Flowcat itself does not implement the control plane. To wire a SIP/RTP carrier, the embedder
supplies (in its own, separately-reviewed code):

1. A **telephony provider** for the carrier: `sample_rate() = 8000`; per-event webhook signature
   verification (e.g. HMAC over the carrier's event string); inbound parsing (DID/caller/call-id)
   and status → lifecycle mapping; plus any REST auth signer the carrier's API needs.
2. **Routes** the carrier talks to: CDR/event webhooks (side-effect-only), a service-authed
   `sip/inbound-resolve` (DID → workflow → run+token), and an `initiate-call` branch whose
   originate path POSTs to the embedder's media-binary originate endpoint (`{run_id, token,
   to_number}`) — no ESL.
3. A **credential shape** for the carrier (API key/secret, SIP login/password/server, caller-id),
   sealed at rest. The SIP trunk that actually REGISTERs is configured by the embedder's SIP trunk
   configuration (server / login / password / caller-id) passed to `SipConfig`/`SipAgent`.

## OSS boundary & license

- **Flowcat (Apache-2.0):** `flowcat-core` (runtime + the four traits
  `MediaTransport`·`RealtimeLlm`·`AgentBrain`·`SessionSource` + native SIP UA `sip/` +
  `WsCarrierTransport` + codec/recorder + a demo brain) and the sibling crates
  `flowcat-services` (~80 STT/TTS/LLM/realtime providers + obs exporters + MCP),
  `flowcat-transports`, and `flowcat-telephony` (carrier serializers + DTMF), plus the
  `flowcat-cli` demo binary. Every provider/transport is one opt-in Cargo feature, so a fully
  local/air-gapped build is just a feature selection. No embedder contract.
- **The embedder (its own code):** the glue binary (engine adapter + control-plane client +
  auth/routing), plus whatever editor, campaigns, billing, and multi-tenant control plane it
  provides. The host's brain implementation plugs in via the `AgentBrain` trait.
- The **Memory** trait is OSS; a concrete backend is the embedder's choice (not wired here).

## Security

- **Per-call token** authorizes the media WS + every control-plane call. The WS query token is
  checked by the embedder before `resolve`.
- **Carrier signature verification** (CDR webhooks) stays in the embedder's control plane
  (constant-time compare, replay window). The model never receives carrier/embedder credentials.
- **SIP trunk credentials live in the embedder's config** — never logged. Inbound INVITE trust
  flows through the embedder's `sip/inbound-resolve` (service-authed; identity from the DID route,
  never the INVITE body). The outbound originate endpoint authenticates (run token / service auth)
  and validates `to_number` (E.164). RTP from an unexpected source addr is dropped (symmetric RTP).
  No new secret crosses the wire to the model/caller.
- The embedder seals carrier config at rest. Both review gates (code + security) apply to all
  control-plane changes.

## Testing strategy

- **Unit (CI, no infra):** G.711 round-trip; resample ratios; Plivo serializer parse/encode;
  **SIP**: SDP offer/answer build+parse (PCMU/PCMA pick, ptime), RTP packetize/depacketize
  (seq/ts/PT), the `SipTransport`→`MediaTransport` mapping (fed RTP → Audio; synthesized BYE →
  Stop); carrier signature accept/reject/tamper/replay + REST signing known-answer vector;
  the brain adapter (transition→tool→re-prompt→end); Gemini Live JSON message encode/decode
  against captured fixtures.
- **Integration (CI):** a mock `MediaTransport` + mock `RealtimeLlm` driving the call pipeline to
  a clean finalize (no network).
- **Live (gated — needs infra + user OK):** a carrier dev number (NEVER a production number) for
  the WS-media path; a SIP trunk registered by the native `SipAgent` for the SIP path. **Inform
  the user before any live call** (account guardrail).
