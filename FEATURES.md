<!-- SPDX-License-Identifier: Apache-2.0 -->
# Flowcat feature-flag matrix

Every provider, transport, and exporter is a single Cargo feature, `dep:`-gated
so the **default build pulls none of their client dependencies**. This file
enumerates every feature across the four crates; it is derived from the
`Cargo.toml`s and is the exhaustive companion to the README's summary table and
[`PROVIDERS.md`](PROVIDERS.md).

**(D)** = distinct client (own wire protocol). **(W)** = thin wrapper over a (D)
family client (base_url + auth + default-model change). See
[`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## `flowcat-core` — `default = ["sip", "recorder"]`

| Feature | Default | Pulls | What it gates |
| --- | --- | --- | --- |
| `sip` | ✅ | — | native SIP user-agent (REGISTER/INVITE/ACK/BYE) |
| `recorder` | ✅ | — | call recorder-as-processor (WAV via `hound`) |
| `vad-ort` | — | `ort`, `ndarray` | Silero VAD + Smart-Turn ONNX impls |
| `filter-rnnoise` | — | `nnnoiseless` | pure-Rust RNNoise noise-suppression filter |

The trait seams (Transport/Stt/Tts/Llm/RealtimeLlm/Vad/Turn/FrameSerializer/
Brain/SessionSource), the Gemini Live client, codec/resample, and the SIP/RTP/SDP
stack are always present; only the heavy ONNX/RNNoise *bodies* are gated.

## `flowcat-services` — `default = []`

Nothing is on by default. Umbrellas: `stt-all`, `tts-all`, `llm-all`,
`realtime-all`, `obs-all`.

### Realtime / speech-to-speech (7 incl. core Gemini)

| Feature | Tag | Pulls |
| --- | --- | --- |
| `realtime-openai` | (D) | `tokio-tungstenite`, `tokio`, `base64` |
| `realtime-azure` | (W) over openai | — |
| `realtime-grok` | (W) over openai | — |
| `realtime-inworld` | (W) over openai | — |
| `realtime-ultravox` | (D) | `tokio-tungstenite`, `tokio` |
| `realtime-novasonic` | (D) | `tokio`, `base64` |

> Gemini Live is the 7th realtime impl and lives in `flowcat-core`
> (re-exported as `flowcat_core::GeminiLive`); it has no `flowcat-services` feature.

### STT (20)

| Feature | Tag | Transport |
| --- | --- | --- |
| `stt-deepgram` | (D) | WS (ref impl) |
| `stt-assemblyai` | (D) | WS |
| `stt-gladia` | (D) | WS |
| `stt-soniox` | (D) | WS |
| `stt-speechmatics` | (D) | WS |
| `stt-cartesia` | (D) | WS |
| `stt-azure` | (D) | WS |
| `stt-gradium` | (D) | WS |
| `stt-elevenlabs` | (D) | segmented HTTP |
| `stt-sarvam` | (D) | REST |
| `stt-mistral` | (D) | REST |
| `stt-openai` | (D) | Whisper-HTTP base |
| `stt-groq` / `stt-fal` / `stt-speaches` / `stt-xai` | (W) over openai | Whisper-HTTP |
| `stt-google` | (D) | gRPC (`tonic`) |
| `stt-nvidia` | (D) | gRPC / Riva (`tonic`) |
| `stt-aws-transcribe` | (D) | SigV4 WS (`hmac`/`sha2`) |
| `stt-whisper-local` | (D) | local (`whisper-rs`; **C build, needs `cmake`**) |

### TTS (31)

| Feature | Tag | Transport |
| --- | --- | --- |
| `tts-cartesia` | (D) | WS (ref impl) |
| `tts-elevenlabs` / `tts-deepgram` / `tts-rime` / `tts-asyncai` / `tts-gradium` / `tts-soniox` / `tts-resemble` | (D) | WS |
| `tts-openai` | (D) | HTTP base |
| `tts-groq` / `tts-xai` | (W) over openai | OpenAI-TTS-HTTP |
| `tts-speaches` / `tts-kitten` | (W) over openai | OpenAI-TTS-HTTP (self-hosted) |
| `tts-azure` | (D) | SSML WS/HTTP |
| `tts-sarvam` / `tts-mistral` / `tts-hume` / `tts-inworld` / `tts-minimax` / `tts-camb` / `tts-speechmatics` | (D) | HTTP |
| `tts-fish` / `tts-lmnt` / `tts-neuphonic` / `tts-smallest` | (D) | interruptible HTTP |
| `tts-kokoro` / `tts-piper` / `tts-xtts` | (D) | local/model HTTP |
| `tts-google` / `tts-nvidia` | (D) | gRPC (`tonic`) |
| `tts-aws-polly` | (D) | SigV4 (`hmac`/`sha2`) |

### LLM (23)

| Feature | Tag |
| --- | --- |
| `llm-openai` | (D) — chat-completions SSE, ref impl |
| `llm-openai-responses` | (D) — Responses API |
| `llm-anthropic` | (D) — Messages API |
| `llm-google` | (D) — Gemini `generateContent` |
| `llm-aws-bedrock` | (D) — SigV4 event-stream (`hmac`/`sha2`) |
| `llm-groq`, `llm-together`, `llm-fireworks`, `llm-openrouter`, `llm-perplexity`, `llm-deepseek`, `llm-cerebras`, `llm-sambanova`, `llm-nebius`, `llm-novita`, `llm-qwen`, `llm-grok`, `llm-nvidia-nim`, `llm-ollama`, `llm-sarvam`, `llm-mistral`, `llm-azure`, `llm-speaches` | (W) over `llm-openai` (base_url + auth) — 18 wrappers |

### Observability + MCP

| Feature | Pulls |
| --- | --- |
| `obs-otel` | `opentelemetry` |
| `obs-sentry` | `reqwest` |
| `obs-langfuse` | `reqwest` |
| `mcp` | `reqwest` — MCP-as-processor client |

### Brain adapters

| Feature | Pulls |
| --- | --- |
| `brain-http` | `reqwest`, `tokio/rt-multi-thread` — `RemoteBrain`: drives conversation policy from an HTTP service (e.g. a Python webhook). See [`examples/python-remote-brain`](examples/python-remote-brain). |

## `flowcat-transports` — `default = []`

| Feature | Pulls |
| --- | --- |
| `webrtc-str0m` | `str0m`, `audiopus`, `tokio`, `tokio-util` |
| `ws` | `tokio-tungstenite`, `tokio`, `tokio-util` |
| `daily` | `reqwest` |
| `livekit` | — (stub) |
| `local` | `audiopus`, `tokio` (local mic/speaker) |

## `flowcat-telephony` — `default = ["plivo"]`

Serializers are dependency-free flags (pure framing).

| Feature | Default |
| --- | --- |
| `plivo` | ✅ |
| `twilio`, `telnyx`, `exotel`, `vonage`, `genesys`, `asterisk`, `cloudonix`, `vobiz` | — |
| `dtmf-inband` | — (in-band Goertzel DTMF; RFC2833 is always available) |

## `flowcat-agent` — `default = ["brain"]`

A declarative, graph-based agent: define an agent as a node/edge `graph_spec`
(JSON/YAML) instead of hand-writing an `AgentBrain`. `DeclarativeBrain` is a
ready-to-use `flowcat_core::AgentBrain` over the graph engine.

| Feature | Pulls |
| --- | --- |
| `brain` | `flowcat-core` — the `DeclarativeBrain` `AgentBrain` adapter (default on) |

With `default-features = false` the pure graph engine (parse / validate /
`{{var}}` interpolation) builds with no `flowcat-core` dependency.

---

## Toolchain caveats

- **gRPC** (`stt-google`/`stt-nvidia`/`tts-google`/`tts-nvidia`) compiles `.proto`s
  via `tonic-build` → needs `protoc` on PATH.
- **`stt-whisper-local`** bundles whisper.cpp → needs `cmake` + a C/C++ toolchain.
  The Rust stub compiles; the dep's C build is the gate.

Everything else is rustls-only `reqwest` / `tokio-tungstenite` — no system
OpenSSL, no AWS SDK (the AWS providers hand-roll SigV4 over `hmac`/`sha2`).
