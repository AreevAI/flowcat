# syntax=docker/dockerfile:1
#
# Flowcat reproducible benchmark image — pipecat (Python) vs flowcat (Rust) under
# a real WebSocket + μ-law load, on one box. Build context = the repo root; run
# from there:  docker compose -f bench/compose.yml up --build
# (or `docker build -f bench/Dockerfile -t flowcat-bench .`)

# ---- Rust builder ----------------------------------------------------------
FROM rust:1-bookworm AS rust
WORKDIR /build
COPY bench-rs/ ./bench-rs/
RUN cd bench-rs && cargo build --release --bin io_harness --bin flowcat-bench

# ---- Python runtime + benchmark -------------------------------------------
# 3.12 (not 3.13): the standard-library `audioop` μ-law codec is removed in 3.13.
FROM python:3.12-slim AS run
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git util-linux procps ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# The pipecat you're comparing against — clone it into ./pipecat (gitignored; see
# bench/README.md). Benchmarks the source via editable install, not a PyPI build.
COPY pipecat/ /app/pipecat/
COPY bench/pipecat_sut.py bench/pipecat_profile.py /app/
COPY bench/run-sweep.sh /app/run-sweep.sh
COPY --from=rust /build/bench-rs/target/release/io_harness /app/io_harness
COPY --from=rust /build/bench-rs/target/release/flowcat-bench /app/flowcat-bench

RUN python -m venv /venv \
    && /venv/bin/pip install -U pip wheel \
    && /venv/bin/pip install -e '/app/pipecat[websocket,tracing]' uvloop \
    && chmod +x /app/run-sweep.sh
ENV PATH="/venv/bin:${PATH}"

# Runs the full Rust-vs-pipecat sweep and prints the comparison tables.
ENTRYPOINT ["/app/run-sweep.sh"]
