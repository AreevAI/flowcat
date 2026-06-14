# Configuration

Flowcat has **no central config file and no settings framework** (no `figment`,
`envy`, or `config` crate, no `flowcat.toml`). That is deliberate: Flowcat is a
*library* you embed, so configuration lives in two clearly separated places.

| Layer | Who reads it | How it's set |
|---|---|---|
| **Runtime knobs** | the Flowcat runtime, at call time | `FLOWCAT_*` environment variables |
| **Credentials & call settings** | your embedder, passed into constructors | Rust code — `SipConfig`, each service's constructor |

The important consequence: **the runtime does not read provider API keys from the
environment in production.** Setting `OPENAI_API_KEY` does not make a running
Flowcat pick it up — your embedder reads its own credentials however it likes and
passes them to the provider constructor.

> Contributing to Flowcat and running the live integration tests? Those use their
> own `*_API_KEY` environment variables — see
> [Building and running tests](./contributing.md#building-and-running-tests).

---

## 1. Runtime environment variables

These are the variables the runtime itself reads. All are **optional** — each
falls back to a built-in default. They tune the realtime (speech-to-speech)
turn-taking and voice; they have no effect on the cascaded path.

| Variable | Applies to | Values | Default |
|---|---|---|---|
| `FLOWCAT_VOICE` | Gemini Live, OpenAI Realtime | provider voice name | `Fenrir` (Gemini), `alloy` (OpenAI) |
| `FLOWCAT_VAD_START_SENSITIVITY` | Gemini Live | `START_SENSITIVITY_UNSPECIFIED` · `_LOW` · `_HIGH` | `START_SENSITIVITY_LOW` |
| `FLOWCAT_VAD_END_SENSITIVITY` | Gemini Live | `END_SENSITIVITY_UNSPECIFIED` · `_LOW` · `_HIGH` | `END_SENSITIVITY_HIGH` |
| `FLOWCAT_VAD_PREFIX_PADDING_MS` | Gemini Live | `u32` milliseconds | `500` |
| `FLOWCAT_VAD_SILENCE_DURATION_MS` | Gemini Live | `u32` milliseconds | `350` |

**Turn-taking, in plain terms.** `START_SENSITIVITY_LOW` + a 500 ms
`PREFIX_PADDING` means a brief caller sound — a backchannel ("uh-huh"), a cough,
line noise — is *not* committed as a turn, so the agent stops cutting off its own
speech. The end side stays eager (`END_SENSITIVITY_HIGH` + 350 ms trailing
silence) so the agent still replies promptly once the caller actually finishes.
Invalid values fall back to the default rather than erroring.

> Defined in
> [`flowcat-core/src/realtime/gemini_live.rs`](https://github.com/AreevAI/flowcat/blob/main/flowcat-core/src/realtime/gemini_live.rs)
> (`VadConfig::from_env`) and
> [`flowcat-services/src/realtime/openai.rs`](https://github.com/AreevAI/flowcat/blob/main/flowcat-services/src/realtime/openai.rs).

---

## 2. Credentials & call settings (programmatic)

Everything else is configured in Rust, by your [embedder](./embedder.md), and
passed into the relevant constructor. This is what keeps credentials on
infrastructure you control — they never transit a Flowcat-owned config surface.

### SIP / telephony — `SipConfig`

Passed to `SipAgent::start(cfg)`. Telephony trunk credentials live here and reach
nothing else.

| Field | Type | Notes |
|---|---|---|
| `server` | `String` | Registrar / proxy URI, e.g. `sip:sip.example.com` |
| `login` | `String` | SIP auth username (trunk login) |
| `password` | `String` | SIP auth password |
| `caller_id` | `String` | E.164 / trunk number used as the From user on outbound |
| `public_ip` | `Option<Ipv4Addr>` | Advertise in Via/Contact/SDP for NAT; `None` → bound local address |
| `sip_port` | `Option<u16>` | Local SIP signaling port; `None` → `5060` |
| `rtp_port_base` | `u16` | First (even) RTP port to probe; default `16000` |
| `rtp_port_tries` | `u16` | Even ports to probe from the base; default `200`. **Caps concurrent call media to this number.** |

> Defined in
> [`flowcat-core/src/sip/agent.rs`](https://github.com/AreevAI/flowcat/blob/main/flowcat-core/src/sip/agent.rs).
> See the [Deployment guide](./deployment.md#networking) for the firewall
> implications of the RTP range.

### Provider credentials

Each STT / TTS / LLM / realtime service takes its key (and any voice / model
settings) through its own constructor. Your embedder decides where those come
from — its own env vars, a secrets manager, a vault. A common, simple choice is
to read an env var named like the provider expects and pass it in:

```rust
// illustrative — your embedder owns this
let api_key = std::env::var("OPENAI_API_KEY")?;   // your choice, not the runtime's
let tts = OpenAiTts::new(&api_key, /* voice, model, … */);
```

The full set of provider constructors and their arguments is in
[`flowcat-services`](https://github.com/AreevAI/flowcat/tree/main/flowcat-services/src);
see the [API reference](./api-reference.md) for how to browse it as rustdoc.

---

> **Next:** [Deployment](./deployment.md) — build a release binary and ship it in
> your own VPC.
