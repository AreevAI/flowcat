<!-- SPDX-License-Identifier: Apache-2.0 -->
# Flowcat provider protocol-family map

> Mirrors the per-provider feature matrix + shared-file pattern used elsewhere;
> covers the STT / TTS / LLM provider breadth (see `ROADMAP.md`).
>
> **Purpose.** This map covers the ~70 remaining providers, each landing in an
> isolated worktree. The merge hazard is contention on the two shared files —
> `flowcat-services/Cargo.toml` (`[features]`/`[dependencies]`) and each category's
> `src/{stt,tts,llm}/mod.rs` (the `mod`-decl + `pub use` list). Pre-creating **every**
> provider's module home + feature + dep + a compiling stub means each provider
> **fills one body — never adds a `mod` decl or a Cargo line** → conflict-free merge.
> It also classifies every provider as **(D) distinct client**
> or **(W) thin wrapper** so a protocol family's real client is implemented **once**
> and the wrappers reuse it.

---

## 0. The two-letter triage

| Tag | Meaning | What the fan-out agent does |
|---|---|---|
| **(D)** | **Distinct client** — its own wire protocol (own WS/HTTP framing, own auth, own message schema). Needs a real, from-scratch impl + wire-fixture tests. | Implement the client + pure encode/decode seam + unit tests (the Deepgram/Cartesia/OpenAI template). |
| **(W)** | **Thin wrapper** — OpenAI-/Whisper-/an-existing-client-compatible: same wire protocol as a (D) family member, differing only in `base_url` + auth header + default model. | A small struct that constructs the family's (D) client with the provider's base URL/auth and delegates the trait (the `GrokRealtime`-over-`OpenAiRealtime` template). |

**The discipline:** implement each **family's (D) client once**; every (W) in that
family is ~30 lines of config + delegation. A category agent should own a whole family
(the (D) client + all its (W)s) so the wrapper contract is set by the same hand.

---

## 1. LLM (26 providers; openai✓ done, anthropic = stub-home here)

**The headline finding (confirmed against pipecat):** the LLM list is *overwhelmingly*
OpenAI-compatible. In pipecat **15 of the LLM services literally `class XLLMService(OpenAILLMService)`** —
they are a `base_url` + default-model change. They are all **(W)** over the existing
`OpenAiLlm` (which already has `.base_url()` + `.model()` builders and was written for
exactly this — its doc already names OpenRouter). Only **3 are (D)**.

| Provider | Tag | Family / how | base_url (verified in pipecat) |
|---|---|---|---|
| openai | **(D) ✓done** | `OpenAiLlm` (chat-completions SSE) | `https://api.openai.com/v1` |
| openai_responses | **(D)** | OpenAI **Responses API** (`/responses`, different request+event schema from chat-completions) — its own decode | `https://api.openai.com/v1` |
| anthropic | **(D)** | Messages API (`/v1/messages`, `anthropic-version` header, `content-block-delta` SSE) — own client | `https://api.anthropic.com` |
| google (gemini) | **(D)** | `generativelanguage` `streamGenerateContent` (Gemini text API; distinct from the realtime client already in core) | `https://generativelanguage.googleapis.com` |
| aws_bedrock | **(D)** | Bedrock `InvokeModelWithResponseStream` (SigV4 + AWS event-stream framing) — own SigV4 path | (region host) |
| groq | (W) | OpenAiLlm | `https://api.groq.com/openai/v1` |
| together | (W) | OpenAiLlm | `https://api.together.xyz/v1` |
| fireworks | (W) | OpenAiLlm | `https://api.fireworks.ai/inference/v1` |
| openrouter | (W) | OpenAiLlm | `https://openrouter.ai/api/v1` |
| perplexity | (W) | OpenAiLlm | `https://api.perplexity.ai` |
| deepseek | (W) | OpenAiLlm | `https://api.deepseek.com/v1` |
| cerebras | (W) | OpenAiLlm | `https://api.cerebras.ai/v1` |
| sambanova | (W) | OpenAiLlm | `https://api.sambanova.ai/v1` |
| nebius | (W) | OpenAiLlm | `https://api.tokenfactory.nebius.com/v1/` |
| novita | (W) | OpenAiLlm | `https://api.novita.ai/openai` |
| qwen | (W) | OpenAiLlm | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` |
| grok (xai) | (W) | OpenAiLlm | `https://api.x.ai/v1` |
| nvidia_nim | (W) | OpenAiLlm | `https://integrate.api.nvidia.com/v1` |
| ollama | (W) | OpenAiLlm | `http://localhost:11434/v1` |
| sarvam | (W) | OpenAiLlm | `https://api.sarvam.ai/v1` |
| mistral | (W)¹ | OpenAiLlm | `https://api.mistral.ai/v1` |
| azure | (W)² | OpenAiLlm | Azure endpoint + `api-version`, `api-key` auth |
| speaches | (W) | OpenAiLlm | `http://localhost:11434/v1` (self-hosted, OpenAI-compatible) |

¹ **mistral**: the brief flags it as arguably (D), but pipecat's `MistralLLMService(OpenAILLMService)`
is OpenAI-compatible → **(W)**. (Mistral STT/TTS are (D) — see below; only the LLM is a wrapper.)
² **azure**: OpenAI-compatible wire but a different auth (`api-key` header) + URL shape (`?api-version=`).
It needs `OpenAiLlm` to accept an `api-key`-style header — a tiny (D-ish) seam on the OpenAI client
(an auth-mode toggle), then the rest is (W). Grouped with the (W) cohort but flagged for the auth seam.

**LLM (D) count: 5** (openai✓, openai_responses, anthropic, google/gemini, aws_bedrock).
**LLM (W) count: 18.** → One agent implements the 4 new (D) clients; one agent fans out
all 18 (W)s (each a base_url+model struct over `OpenAiLlm`). The `anthropic` stub-home is
set up here because its `llm-anthropic` feature already exists from earlier
but the impl was never landed (the earlier "anthropic done" is the *feature*, not the
module) — this map fills it.

---

## 2. STT (~21 remaining; deepgram✓ done)

Protocol families (cross-checked against pipecat base classes):

- **Streaming-WebSocket (each its own JSON schema)** — `WebsocketSTTService` subclasses.
  Mostly **(D)** (distinct message schemas), except where they ride a sibling's protocol.
- **Whisper-HTTP segmented** — `BaseWhisperSTTService` / `SegmentedSTTService`: POST a
  finished audio segment to an OpenAI-`/audio/transcriptions`-shaped endpoint. **One (D)
  Whisper-HTTP client**, the rest **(W)** over it.
- **gRPC** — Google + NVIDIA Riva use `tonic`. **(D)** each (distinct protos).
- **Local** — whisper.cpp via `whisper-rs`. **(D)**, C-toolchain (see §5).

| Provider | Tag | Family / protocol | Notes |
|---|---|---|---|
| deepgram | **(D) ✓done** | streaming WS `/v1/listen` | reference impl |
| assemblyai | **(D)** | streaming WS v3 `wss://streaming.assemblyai.com/v3/ws` | own JSON schema |
| gladia | **(D)** | streaming WS (init-then-stream session) | own schema |
| soniox | **(D)** | streaming WS `wss://stt-rt.soniox.com/transcribe-websocket` | own schema |
| speechmatics | **(D)** | streaming WS `wss://*.rt.speechmatics.com/v2` | own schema |
| cartesia | **(D)** | streaming WS (`/stt/websocket`) | own schema (sibling of the TTS WS) |
| aws_transcribe | **(D)** | AWS Transcribe streaming WS (SigV4-signed) + event framing | SigV4 (see §5) |
| azure | **(D)** | Azure Speech SDK / WS | own protocol |
| elevenlabs | **(D)** | segmented HTTP `/v1/speech-to-text` | own (not Whisper-shaped) |
| **openai** | **(D) Whisper-HTTP base** | POST `/audio/transcriptions` (Whisper) | **the Whisper-HTTP (D) family client** |
| groq | (W) | Whisper-HTTP | `https://api.groq.com/openai/v1` |
| fal | (W) | Whisper-HTTP | fal endpoint |
| speaches | (W) | Whisper-HTTP | self-hosted OpenAI-compatible |
| xai | (W)³ | Whisper-HTTP / OpenAI-STT WS | `https://api.x.ai/v1` |
| sarvam | **(D)** | sarvam REST STT | own schema |
| mistral | **(D)** | mistral STT REST | own schema |
| google | **(D) gRPC** | Cloud Speech `tonic` streaming | gRPC (see §5) |
| nvidia (riva) | **(D) gRPC** | Riva ASR `tonic` | gRPC (see §5) |
| gradium | **(D)** | gradium WS STT | own schema |
| whisper_local | **(D)** | `whisper-rs` (whisper.cpp) | C-toolchain (see §5) |
| speaches¹ | see groq row | — | listed once above |

³ **xai STT**: pipecat's is `WebsocketSTTService`; xAI exposes an OpenAI-compatible STT — treat as
(W) over the Whisper-HTTP/OpenAI-STT family unless a live test shows a distinct WS schema (flag for
the implementer). Conservative grouping: (W).

**STT (D) count: ~16** (deepgram✓ + assemblyai, gladia, soniox, speechmatics, cartesia,
aws_transcribe, azure, elevenlabs, openai-Whisper-base, sarvam, mistral, google-gRPC,
nvidia/riva-gRPC, gradium, whisper_local). **STT (W) count: ~4** (groq, fal, speaches, xai —
all Whisper-HTTP over the openai client). The brief also names `nvidia`, `riva`, `speaches` —
`riva` **is** nvidia's STT (one provider, one feature `stt-nvidia`); `speaches` is the (W).

---

## 3. TTS (~30 remaining; cartesia✓ done)

Two big families + a long tail of HTTP-POST-audio providers:

- **Streaming-WebSocket TTS** — `WebsocketTTSService` subclasses: connect a WS, send a
  synthesis request, read base64/binary PCM chunks. The Cartesia ref impl is the template.
  Each has its **own request/response schema → (D)**.
- **HTTP-POST-audio (one-shot / segmented)** — `TTSService` subclasses that POST text and
  read an audio body (mp3/pcm/opus). The wire shape is similar (POST → audio bytes) but the
  request schemas + audio containers differ enough that there is **no single reusable client**;
  most are **(D)**, a few are **(W)** where they share an OpenAI-TTS shape.

| Provider | Tag | Family / protocol |
|---|---|---|
| cartesia | **(D) ✓done** | WS streaming (reference) |
| elevenlabs | **(D)** | WS streaming `/v1/text-to-speech/{voice}/stream-input` |
| deepgram | **(D)** | WS streaming `/v1/speak` |
| rime | **(D)** | WS streaming |
| asyncai | **(D)** | WS streaming |
| gradium | **(D)** | WS streaming |
| soniox | **(D)** | WS streaming |
| resemble | **(D)** | WS streaming |
| openai | **(D) HTTP base** | POST `/audio/speech` → audio body (the **OpenAI-TTS-HTTP family client**) |
| groq | (W) | OpenAI-TTS-HTTP (`/audio/speech`) over the openai client |
| xai | (W) | OpenAI-TTS-HTTP shape (`XAIHttpTTSService`) |
| aws_polly | **(D)** | AWS Polly (SigV4 + `SynthesizeSpeech`) — §5 |
| azure | **(D)** | Azure Speech SSML over WS/HTTP |
| google | **(D)** | Google Cloud TTS (HTTP/gRPC) — §5 for the gRPC variant |
| sarvam | **(D)** | sarvam HTTP TTS |
| mistral | **(D)** | mistral HTTP TTS |
| nvidia | **(D)** | NVIDIA Riva TTS (gRPC) — §5 |
| hume | **(D)** | Hume HTTP |
| inworld | **(D)** | Inworld HTTP |
| minimax | **(D)** | MiniMax HTTP |
| camb | **(D)** | Camb HTTP |
| fish | **(D)** | Fish Audio (interruptible) |
| lmnt | **(D)** | LMNT (interruptible) |
| neuphonic | **(D)** | Neuphonic (interruptible) |
| smallest | **(D)** | Smallest (interruptible) |
| speechmatics | **(D)** | Speechmatics TTS |
| kokoro | **(D)** | Kokoro (local model) — C/ONNX-ish, may need a model file |
| piper | **(D)** | Piper (local model, subprocess/ONNX) |
| xtts | **(D)** | XTTS (local model) |
| asyncai¹ | see WS row | — | listed above |

**TTS (D) count: ~28** (cartesia✓ + 27). **TTS (W) count: ~2** (groq, xai — OpenAI-TTS-HTTP).
TTS is the least wrapper-friendly category — almost every vendor has a bespoke request body.
The leverage here is **two shared client helpers** (a `WsTtsClient` connect/send/read-base64
skeleton mirroring Cartesia, and an `HttpTtsClient` POST-and-read-audio skeleton) that the (D)
impls reuse for the transport plumbing while each supplies its own request encode + container
decode. (Those helpers are an implementation convenience for the implementations, not part of this
map.)

---

## 4. Recommended fan-out grouping (which agent owns what)

To keep each protocol family's (D) client authored once and its (W)s consistent, group by
**family**, not by alphabetical slice:

| Agent | Owns | Why grouped |
|---|---|---|
| **A — LLM/OpenAI-compatible** | the 18 LLM **(W)**s over `OpenAiLlm` | identical pattern; one PR, 18 ~30-line structs + the base_url table |
| **B — LLM/distinct** | anthropic, google-gemini, openai_responses, aws_bedrock (the 4 new LLM **(D)**) | 4 distinct clients incl. the SigV4 Bedrock path |
| **C — STT/Whisper-HTTP** | openai-STT **(D)** client + groq/fal/speaches/xai **(W)** | one Whisper-HTTP client, 4 wrappers |
| **D — STT/streaming-WS** | assemblyai, gladia, soniox, speechmatics, cartesia-STT, elevenlabs-STT, azure-STT, gradium-STT, sarvam, mistral-STT | all WS/REST distinct schemas; the Deepgram template |
| **E — STT/gRPC+AWS+local** | google-gRPC, nvidia/riva-gRPC, aws_transcribe (SigV4), whisper_local | the toolchain-heavy STT (§5) — keep together so the `tonic`/SigV4/`whisper-rs` plumbing is solved once |
| **F — TTS/streaming-WS** | elevenlabs, deepgram, rime, asyncai, gradium, soniox, resemble TTS | share the `WsTtsClient` helper (Cartesia template) |
| **G — TTS/HTTP cloud** | openai-TTS **(D)** + groq/xai **(W)**, hume, inworld, minimax, camb, sarvam, mistral, speechmatics, azure-TTS | share the `HttpTtsClient` helper |
| **H — TTS/interruptible+local+AWS+gRPC** | fish, lmnt, neuphonic, smallest (interruptible); kokoro, piper, xtts (local models); aws_polly (SigV4); google-TTS / nvidia-TTS (gRPC) | the bespoke + toolchain-heavy TTS tail (§5) |

8 groups, family-coherent, no two touching the same provider. The new auth/SigV4 paths
(aws_bedrock, aws_transcribe, aws_polly, the azure api-key seam) are the security-sensitive ones.

---

## 5. Toolchain / dependency notes for the fan-out

This crate **declares every dep `optional` + `dep:`-gated** so a default build pulls
nothing and `--features <provider>` compiles the stub. Three families need a toolchain/dep beyond
the existing `reqwest`/`tokio-tungstenite`/`tokio`/`base64`:

| Family | Dep | Gate(s) | Toolchain note |
|---|---|---|---|
| Google / NVIDIA-Riva STT+TTS | `tonic` 0.14 (already declared) | `stt-google`, `stt-nvidia`, `tts-google`, `tts-nvidia` | gRPC; needs the `.proto`s compiled — `tonic-build` at build time (a `build.rs`, added by the impl, not here). `protoc` must be on PATH for codegen. |
| whisper_local | `whisper-rs` 0.16 (already declared) | `stt-whisper-local` | bundles whisper.cpp → **needs `cmake` + a C/C++ toolchain** to build. **Not present by default on this machine** unless cmake is installed — building `--features stt-whisper-local` may fail at the C build step. The Rust stub itself compiles; the dep's C build is the gate. |
| AWS (Bedrock LLM, Transcribe STT, Polly TTS) | hand-rolled **SigV4** — **no AWS SDK** | `llm-aws-bedrock`, `stt-aws-transcribe`, `tts-aws-polly` | no new crate; SigV4 over `reqwest` + a SHA-256/HMAC (`hmac`/`sha2`) — declared optional. |

**`opentelemetry`/gRPC/whisper-rs are the only non-pure-Rust-network deps.** Everything else is
`reqwest` (rustls) or `tokio-tungstenite` (rustls) — the rustls-only + zero-default-cost discipline
holds. This adds **`hmac` + `sha2`** (optional) for the AWS SigV4 family; no other new dep.

---

## 6. Acceptance

- `cargo build -p flowcat-services` (default) → pulls **nothing**, compiles the trait re-exports +
  stubs only.
- `cargo build -p flowcat-services --features stt-all,tts-all,llm-all` → compiles every stub
  (modulo `stt-whisper-local`'s C build, which needs `cmake` — see §5).
- `cargo test -p flowcat-services` (existing tests) → still green.
- Every stub: Apache SPDX header + `//! WS-n: <provider> — TODO` + a frozen-trait `impl` whose
  bodies return `FlowcatError::Other("<provider>: not yet wired (WS-n)")` (or `todo!()` for the
  `&str` `name()` accessor) so the crate compiles. `#[allow(dead_code)]` on the held config fields
  is the sanctioned clippy escape on a stub.

## 7. What this map deliberately does NOT do

- It does **not** implement any provider body (that is the fan-out).
- It does **not** add the WS/HTTP TTS helper clients (`WsTtsClient`/`HttpTtsClient`) or the gRPC
  `build.rs` / `.proto`s — those are owned by the impls.
- It does **not** touch `flowcat-core`, the Deepgram/Cartesia/OpenAI reference impls' logic,
  other crates, or the embedder.
- It does **not** create a per-(W) stub where the (W) is *literally* the (D) client + a base_url
  (the LLM wrappers) — those get a thin module stub anyway (so the `mod`/feature exists for the
  fan-out to fill), but their stub body just notes "wrapper over OpenAiLlm — set base_url".
