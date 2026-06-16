# Deployment

This guide is for operators **self-hosting Flowcat in their own VPC** (or fully
air-gapped). Flowcat is one self-contained binary with no hosted control plane
and no phone-home — deployment is "ship a binary, open the right ports, give it
your provider credentials."

> **Two ways to deploy.** The bundled **`flowcat-server`** binary runs one agent
> from a YAML config (no Rust) and is deployable on its own — start with its
> [Docker setup](https://github.com/AreevAI/flowcat/tree/main/deploy) and the
> [Quickstart](./quickstart.md#5-talk-to-a-real-agent-in-your-browser-no-rust). For
> full control you instead ship *your own* [embedder](./embedder.md). Either way the
> build mechanics below are identical; the credential-free `flowcat` CLI
> (`pipeline`, `ws-echo`) is for demos, **not** a server.

---

## 1. Build a release binary

Flowcat is a Cargo workspace. The default build pulls **no** provider or network
dependencies — every STT/TTS/LLM, transport, carrier, and exporter is an opt-in
feature, so you compile only what you ship.

```bash
# Demo binary, default features (native SIP + recorder, no providers):
cargo build --release -p flowcat-cli

# A cloud build — Gemini Live S2S over a WebSocket carrier, with telemetry:
cargo build --release -p flowcat-cli \
  --features "flowcat-services/realtime-all,flowcat-services/llm-all,flowcat-transports/ws,flowcat-services/obs-otel"

# Fully on-prem / air-gapped — local connectors only, no cloud egress:
cargo build --release -p flowcat-cli \
  --features "flowcat-services/stt-whisper-local,flowcat-services/tts-kokoro,flowcat-services/llm-ollama"
```

Feature groups (umbrellas in parentheses): `stt-*` (`stt-all`), `tts-*`
(`tts-all`), `llm-*` (`llm-all`), `realtime-*` (`realtime-all`), `obs-*`
(`obs-otel`, `obs-sentry`, `obs-langfuse`), transports `ws` / `webrtc-str0m`,
carriers `plivo` (default) / `twilio` / `telnyx` / … . The **authoritative,
always-current list** is the `[features]` table in each crate's `Cargo.toml`
([core](https://github.com/AreevAI/flowcat/blob/main/flowcat-core/Cargo.toml) ·
[services](https://github.com/AreevAI/flowcat/blob/main/flowcat-services/Cargo.toml) ·
[transports](https://github.com/AreevAI/flowcat/blob/main/flowcat-transports/Cargo.toml) ·
[telephony](https://github.com/AreevAI/flowcat/blob/main/flowcat-telephony/Cargo.toml)).

> Some local connectors pull native build deps — e.g. `stt-whisper-local` needs a
> C toolchain (`cmake`), and `vad-ort` pulls ONNX Runtime. Install those in your
> build image.

### Fully static binary (optional)

For a dependency-free artifact you can drop into a scratch container or an
air-gapped host, target musl:

```bash
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl -p flowcat-cli
```

(Pure-cloud feature sets build cleanly on musl; feature sets that pull native C
libraries — Whisper, ONNX — may need the matching musl system libraries.)

---

## 2. Containerize

The repo ships a multi-stage
[`Dockerfile`](https://github.com/AreevAI/flowcat/blob/main/Dockerfile) that builds
**`flowcat-server`**, plus a
[`deploy/`](https://github.com/AreevAI/flowcat/tree/main/deploy) compose file +
sample config + env template:

```bash
cp deploy/.env.example deploy/.env          # fill the provider key(s) you use
FLOWCAT_FEATURES=webrtc \
  docker compose -f deploy/docker-compose.yml up --build   # then open :6210
```

To containerize *your own embedder* instead, the same multi-stage pattern applies
— swap the build package + features and the `EXPOSE`d ports for your transport:

```dockerfile
# ---- build ----
FROM rust:1-bookworm AS build
WORKDIR /src
COPY . .
# swap in your embedder package + the features you need
RUN cargo build --release -p flowcat-cli --features "flowcat-services/realtime-all"

# ---- runtime ----
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/flowcat /usr/local/bin/flowcat
# UDP for SIP signaling + the RTP media range (see §Networking)
EXPOSE 5060/udp 16000-16398/udp
ENTRYPOINT ["flowcat"]
```

A `musl` static build collapses the runtime stage to `FROM scratch` + the binary
+ `ca-certificates`.

---

## 3. Run under systemd

```ini
# /etc/systemd/system/flowcat.service
[Unit]
Description=Flowcat voice runtime
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=/usr/local/bin/your-embedder
Restart=on-failure
RestartSec=2
# Runtime knobs (see Configuration). Credentials belong in an EnvironmentFile
# your embedder reads, not in the unit:
Environment=FLOWCAT_VAD_SILENCE_DURATION_MS=350
EnvironmentFile=/etc/flowcat/secrets.env
# Hardening
DynamicUser=yes
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes

[Install]
WantedBy=multi-user.target
```

---

## 4. Networking {#networking}

The native SIP transport binds:

| Purpose | Default | Configured by |
|---|---|---|
| SIP signaling | UDP **5060** | `SipConfig.sip_port` |
| RTP media | UDP **16000**, even ports, up to `rtp_port_tries` (default 200 → 16000–16398) | `SipConfig.rtp_port_base` / `rtp_port_tries` |
| Outbound to providers | TCP 443 (HTTPS/WSS) | per provider |

Operational notes:

- **The RTP range caps concurrent calls.** `rtp_port_tries` is the hard ceiling
  on simultaneous media streams — size it to your expected concurrency, and open
  exactly that even-port range in your firewall / security group.
- **Behind NAT** (most cloud VMs), set `SipConfig.public_ip` so Flowcat
  advertises the reachable address in Via/Contact/SDP — otherwise media is
  offered on an unroutable internal IP.
- A **WebSocket carrier** (e.g. Plivo/Twilio media streams) needs only outbound
  TCP 443 instead of the SIP/RTP ports — your embedder chooses the transport.
- Flowcat exposes **no HTTP control port of its own**. Health checks, metrics
  endpoints, and admin APIs belong to your embedder.

---

## 5. Scale & capacity

The media loop is Rust — no GC, no GIL — so **one process uses every core**.
There is no worker-fleet to size: scale vertically first, then add processes.

From the published [benchmark](https://github.com/AreevAI/flowcat/blob/main/bench/RESULTS.md)
(Azure 16-vCPU VM, WebSocket + μ-law load, 50 fps/call):

- flat **p99 ≤ 0.61 ms** from 10 to **2,000 concurrent calls** in a single process;
- ~**19.6 KB** RAM per idle session, **7** tokio tasks per call.

Practical sizing:

- Bound concurrency with `rtp_port_tries` (SIP) and your carrier's limits.
- One process per box is usually right; run several behind your SIP proxy /
  carrier load balancer only when you exceed a single host.
- Reproduce the benchmark on your own hardware:
  `docker compose -f bench/compose.yml up --build` (see
  [`bench/README.md`](https://github.com/AreevAI/flowcat/blob/main/bench/README.md)).

---

## 6. Observability

Build with an exporter feature and wire it in your embedder:

- `obs-otel` — OpenTelemetry traces/metrics
- `obs-sentry` — error reporting
- `obs-langfuse` — LLM call tracing

All are zero-cost when the feature is off. Per-call metrics and transcripts flow
through the pipeline's `FrameObserver` seam; finalized recordings/transcripts are
uploaded via your `SessionSource` (your storage, your URLs).

---

## 7. Production checklist

- [ ] Release binary built with **only** the features you use.
- [ ] `public_ip` set if the host is behind NAT.
- [ ] Firewall opens UDP 5060 + your RTP even-port range (or just outbound 443 for a WS carrier).
- [ ] `rtp_port_tries` sized to target concurrency.
- [ ] Provider credentials supplied via your embedder (secrets manager / `EnvironmentFile`), **not** baked into the image.
- [ ] `FLOWCAT_VAD_*` / `FLOWCAT_VOICE` tuned for your use case (see [Configuration](./configuration.md)).
- [ ] An exporter feature enabled and pointed at your collector.
- [ ] Restart policy + health checks owned by the embedder.
