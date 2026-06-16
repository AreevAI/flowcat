# SPDX-License-Identifier: Apache-2.0
#
# Multi-stage build for the `flowcat-server` binary.
#
# Build arg FEATURES selects the flowcat-server Cargo feature set:
#   server  (default) — the HTTP server + the full provider connector set
#   webrtc            — server + the browser WebRTC playground (adds str0m)
ARG FEATURES=server

FROM rust:1-bookworm AS build
ARG FEATURES
WORKDIR /src
# Build deps for the connector set: protobuf-compiler (Google/NVIDIA gRPC),
# cmake + clang (local Whisper STT via whisper-rs).
RUN apt-get update \
 && apt-get install -y --no-install-recommends protobuf-compiler cmake clang \
 && rm -rf /var/lib/apt/lists/*
COPY . .
RUN cargo build --release -p flowcat-server --features "${FEATURES}"

FROM debian:bookworm-slim AS runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/flowcat-server /usr/local/bin/flowcat-server
# Mount your agent config here (see deploy/agent.example.yaml).
ENV FLOWCAT_CONFIG=/etc/flowcat/agent.yaml
EXPOSE 6210
ENTRYPOINT ["flowcat-server"]
