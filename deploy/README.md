<!-- SPDX-License-Identifier: Apache-2.0 -->
# Deploying `flowcat-server`

`flowcat-server` runs **one agent** (defined in a YAML/JSON config) over HTTP —
health probes, a Plivo media WebSocket + answer XML, and, with the `webrtc`
feature, a browser playground at `/`. No control plane, no database.

## Docker Compose

```bash
cp deploy/.env.example deploy/.env      # fill the key(s) your topology uses
docker compose -f deploy/docker-compose.yml up --build
```

- Health: `curl localhost:6210/healthz` → `{"status":"ok"}`
- Readiness: `curl localhost:6210/readyz` (200 once the topology's provider key is set)

For the **browser playground** (str0m WebRTC), build the `webrtc` feature:

```bash
FLOWCAT_FEATURES=webrtc docker compose -f deploy/docker-compose.yml up --build
# then open http://localhost:6210/  (set FLOWCAT_WEBRTC_BIND_IP to a reachable host IP)
```

## Without Docker

```bash
cargo build --release -p flowcat-server --features webrtc   # or just "server"
GOOGLE_API_KEY=… ./target/release/flowcat-server --config deploy/agent.example.yaml
```

## Configuration

- **Agent config** — see [`agent.example.yaml`](agent.example.yaml) and the schema
  in `flowcat-server/src/config.rs` (agent graph, `realtime`/`cascaded` topology,
  transport, bind).
- **Provider keys** come from the environment as `<PROVIDER>_API_KEY` (e.g.
  `GOOGLE_API_KEY`, `DEEPGRAM_API_KEY`), the Gemini family also accepts
  `GOOGLE_API_KEY`. Keys are **never** read from the config file.

> The build compiles the full provider connector set, so the build stage needs
> `protobuf-compiler`, `cmake`, and `clang` (already installed in the Dockerfile)
> for the gRPC + local-Whisper connectors. Trim `--features` to speed it up.
