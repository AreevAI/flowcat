# Flowcat benchmark — reproducible (Docker + one Azure VM)

Does a Rust voice-pipeline runtime ("flowcat") beat the Python pipecat runtime on
**concurrent-call density and tail latency** for a real speech-to-speech
topology? This kit answers it reproducibly: one command on a documented VM.

## What it measures

A single WebSocket + μ-law load generator drives **N concurrent full-duplex
"calls"** (each: a 160-byte μ-law frame every 20 ms = 50 fps, full-duplex echo)
against two systems under test on the **same box**, measuring per-frame
round-trip latency (p50/p99/max) and achieved-vs-target throughput:

| SUT | Deployment | Per-frame work |
| --- | --- | --- |
| **flowcat-rust** | **1 process**, all SUT cores (tokio, no GIL) | WS read → μ-law decode → pipeline → μ-law encode → WS write |
| **pipecat-python** | **1 process per SUT core**, `SO_REUSEPORT` (kernel-balanced — the real multiprocess deployment) | same, through a real 5-stage pipecat pipeline (≈22 asyncio tasks/call) |

The load generator is pinned to its own cores so it never steals cycles from the
SUT. Both SUTs do identical wire + codec work; the only difference measured is
**framework cost under concurrency** (Rust threads vs Python GIL + asyncio + GC).

This is deliberately **conservative toward pipecat**: it isolates the pipeline
(echo, no real STT/LLM/TTS) and gives Python its full multiprocess deployment.

## Quick start (Azure)

```bash
# 1) Provision a clean, documented VM (16 vCPU; pick an available region/SKU).
az group create -n flowcat-bench -l eastus
az vm create -g flowcat-bench -n flowcat-bench \
  --image Ubuntu2404 --size Standard_D16s_v5 \
  --admin-username azureuser --generate-ssh-keys --public-ip-sku Standard

# 2) Install Docker on it.
az vm run-command invoke -g flowcat-bench -n flowcat-bench \
  --command-id RunShellScript --scripts "curl -fsSL https://get.docker.com | sh"

# 3) Get the code onto the VM and run the sweep (ssh in first):
ssh azureuser@<public-ip>
git clone <this-repo> flowcat && cd flowcat
git clone https://github.com/pipecat-ai/pipecat pipecat   # the pipecat you compare against (pin a ref for reproducibility)
sudo docker compose -f bench/compose.yml up --build

# 4) Tear down (a full run is minutes; cost is a few dollars).
az group delete -n flowcat-bench --yes --no-wait
```

The sweep prints two blocks of tables (flowcat-rust, then pipecat-python), one
row per caller count. Save them: `... up --build 2>&1 | tee bench/vm-results.txt`.

### Tuning knobs (env on the compose service or `-e`)

- `SWEEP="10 25 50 100 250 500 1000 2000"` — caller counts
- `SECS=10` `FPS=50` — duration / frames-per-sec per call
- `LOADGEN_CORES=4` — cores reserved for the load generator (rest go to the SUT)

## How to read it

For each caller count compare, across the two SUTs:

- **p99 RTT** — the real-time signal. Conversational voice needs a tight tail;
  a p99 in the hundreds of ms is an audible glitch even if the average is fine.
- **achieved_fps_total vs target** — <100% (or a climbing p99) means that SUT is
  saturated at that concurrency.

The headline is the **ratio of usable concurrency**: the highest caller count at
which each SUT holds an acceptable p99, normalized to the same core budget.

## Notes, caveats, reproducibility

- **Re-run BOTH stacks here.** Don't mix these Linux numbers with the macOS
  laptop preview in [`RESULTS.md`](./RESULTS.md). This kit runs both on one
  platform so the comparison is apples-to-apples; the VM run is the authoritative,
  citable result. The laptop preview (Rust p99 ≤ 3 ms to 2,000 calls vs pipecat
  p99 81–445 ms at 10–100 calls/process) is directional only.
- **Pinned SKU = reproducibility.** Results scale with SUT core count (Rust grows
  ~linearly; each Python worker is ~1 core). Report the SKU; anyone matching it
  reproduces the ballpark. `Standard_D16s_v5` (16 vCPU) is the reference.
- **Python 3.12** (image): `audioop` is removed in 3.13.
- **`SO_REUSEPORT` + `host` networking** are why pipecat gets a fair multiprocess
  shot and the loop-back load path has no NAT overhead.
- This measures the **framework floor under real I/O**, not a live call with real
  provider latency — provider round-trips add shared cost to both and are
  dominated by exactly the tail-jitter behavior shown here.

## Keeping the published numbers in sync

A benchmark re-run is **not finished** until all three of these reflect the new
numbers — they are NOT auto-synced:

1. **[`RESULTS.md`](./RESULTS.md)** — the source-of-truth percentile tables and the
   verdict.
2. **The root [`README.md`](../README.md)** — the "Benchmark & capacity" tables and the
   headline figures in the "Why Flowcat" section.
3. **The chart images `docs/bench-p99-latency.png` and `docs/bench-setup.png`** — these
   are **hand-designed, not script-generated**. Update the chart data/labels, re-export
   as PNG (2×), and overwrite the two files in `docs/`; re-running the sweep will
   otherwise silently leave them stale. (The docker command rendered in
   `bench-setup.png` shows `flowcat/bench/compose.yml`; the canonical path is
   `bench/compose.yml` — fix it in the chart source on the next re-export.)

## Files

- `Dockerfile` (+ `Dockerfile.dockerignore`) — Rust build + Python/pipecat runtime
- `compose.yml` — one-command build & run
- `run-sweep.sh` — the in-container orchestrator (core-pinning, both SUTs, sweep)
- `pipecat_sut.py` — pipecat SUT (real pipeline per connection; `--reuse-port`)
- `io_harness` (from `../bench-rs`) — the Rust SUT (`serve`) **and** load gen (`load`)
- `pipecat_profile.py` / `../bench-rs` `flowcat-bench` — the phase-1 framework-floor micro-benchmarks
- `RESULTS.md` — laptop preview + interpretation
