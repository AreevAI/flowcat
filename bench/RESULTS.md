# Flowcat benchmark — pipecat (Python) vs flowcat (Rust)

**Verdict: GO.** A Rust reimplementation of the pipecat voice-pipeline runtime wins
decisively on tail latency + concurrent-call density for a real speech-to-speech
topology.

## Result — authoritative (Azure `Standard_FX16mds_v2`, 16 vCPU, 2026-05-31)

One box, identical WebSocket + μ-law load generator, full-duplex echo, 50 fps/call.
pipecat in its real multiprocess deployment (12 workers, `SO_REUSEPORT`);
flowcat-rust = 1 process. p99 round-trip latency:

| concurrent calls | flowcat-rust | pipecat | pipecat throughput |
| --- | --- | --- | --- |
| 50 | 0.39 ms | 1.1 ms | 100% |
| 100 | 0.51 ms | 33 ms | 100% |
| 250 | 0.59 ms | 51 ms | 100% |
| 500 | 0.51 ms | 843 ms | 100% |
| 1000 | 0.47 ms | 5,673 ms | 77% |
| 2000 | 0.61 ms | 5,074 ms | 41% (failing) |

- flowcat-rust: flat p99 ≤ 0.61 ms, 100% throughput, to 2,000 calls.
- pipecat (fairly deployed): usable to ~250 calls; collapses by 1,000; an intrinsic
  ~100–160 ms GC/GIL max-latency floor even at 10 calls.
- **~8× the concurrent calls at a tight tail; 2–4 orders lower tail at matched load.**

Reproduce: `docker compose -f bench/compose.yml up --build` on a 16-vCPU VM
(see `README.md`). Methodology, framework-floor micro-benchmarks, and the laptop /
local-Docker phases that led here are detailed below.

---

## Detail — methodology & phase history

Machine: this laptop (darwin, uvloop). All numbers are the pipecat **framework
floor** for a real speech-to-speech topology

    transport.input() → user_agg → GeminiLive → transport.output() → assistant_agg

with transport + LLM stubbed as no-op passthroughs. Every figure is therefore a
**lower bound** on real per-session cost (a live call adds a real WS transport,
μ-law serialization, and the Gemini WS client on top).

## Pipecat baseline (the number Flowcat must beat)

| Metric | Value | How measured |
| --- | --- | --- |
| Per-frame routing | **~105 µs/frame** (~15 µs × 7 processor hops) | 20k frames through the pipeline, 1 core |
| Frame-hop throughput | **~9,450 frames/s/core** | same |
| **Calls/process ceiling (CPU)** | **~94 concurrent calls** (framework only; real I/O lowers it to ~tens) | 9,450 ÷ ~100 frames/s per call |
| asyncio tasks / idle session | **22** (stable at 200/500/1000) | `len(asyncio.all_tasks())` delta |
| Tasks at 1000 calls | **~22,000** in one event loop | extrapolation |
| RAM / idle session | **≤ ~1 MB, likely ~tens of KB** (allocator-noise dominated) | `ps` RSS, marginal across 1500→2000 |

### The headline

- **GIL-bound.** This relay work pins to ~1 core per process, so a single
  Python process saturates at ~94 calls (optimistic) on frame routing alone.
  Real μ-law + WS socket I/O + Gemini WS client push the realistic ceiling to
  ~tens of calls/process. You scale by running **many processes**.
- **RAM is not the constraint** — CPU/scheduler saturates first by a wide margin.
- **22 tasks/session** ⇒ ~22k asyncio tasks at 1k calls: real scheduler pressure.

This is exactly the profile where a Rust runtime should win: no GIL (one process
uses all cores), ~ns task handoff, no per-frame Python object allocation.

## Flowcat (Rust) — measured

Same 7-stage passthrough pipeline, 160-byte frames, `ps` RSS. (bench-rs)

| Metric | Python (pipecat) | Rust (flowcat) | ratio |
| --- | --- | --- | --- |
| µs / frame (1 core) | 105.8 | **0.20** | **~525×** |
| frames/s / core | 9,450 | 4,946,530 | ~525× |
| RAM / idle session | ≤ ~1 MB (noisy) | **19.6 KB** (clean) | ~50× |
| tasks / session | 22 asyncio | 7 tokio | — |
| core scaling (1→14, concurrent) | n/a (GIL: 1 core/proc) | 4.39M → **36.85M f/s (8.4×)** | — |

Box-level **framework floor**: Python ~1,300 calls (≈94/proc × 14 procs) vs
Rust ~368,000 calls (one process, 14 cores).

### What this proves — and what it does NOT

PROVES (measured):
- Pipecat's per-frame framework overhead (~106 µs, 22 tasks/session) is **~525×
  larger** than Rust's. In Rust the framework will **never** be the bottleneck;
  in Python it's a real, GIL-serialized tax.
- **No-GIL core scaling works**: one Rust process uses all cores (8.4× / 14).
  Python pins one process to ~1 core, so it needs ~14 processes to fill the box
  — each with its own ~234 MB baseline + connection pools + its own
  PyO3 engine bridge.
- RAM is a non-issue on both sides; Rust's is ~50× tighter and predictable.

DOES NOT PROVE (the critical caveat):
- **The end-to-end density multiple.** This is the *framework floor* with no-op
  stages. A real call also does μ-law encode/decode, WebSocket recv/send
  (syscalls), and audio buffer handling — **work that is ~identical in both
  languages and will dominate per-frame time.** Once that shared I/O cost is
  added, the 525× framework ratio compresses hard. Realistic end-to-end density
  is plausibly **single-digit to low-double-digit ×**, NOT 525×. The true number
  is unmeasured until the real WS+μ-law transport is benchmarked on both sides.
- **Tail latency / jitter under load** — the actual real-time audio concern —
  is not measured here.

### Verdict for the gate

The pre-registered gate was "substantial density win **OR** the FFI-removal +
single-binary simplification is worth it on its own." Phase-1 evidence:

- The **architecture half of the gate is already met**: one Rust process replaces
  ~14 Python processes *and* removes the entire reason the PyO3 FFI
  seam exists. That stands regardless of the exact density multiple.
- The **density half is promising but unproven** — gated on the real-I/O harness.

→ Recommendation: **GO to the next phase — build the real WS+μ-law transport on
both sides and measure end-to-end density + jitter.** Do NOT greenlight a full
rewrite on the 525× floor alone; that number will shrink once real I/O is added.

## Phase 2 — end-to-end (real WebSocket + μ-law, full-duplex echo)

Each caller streams a 160-byte μ-law frame every 20 ms (50 fps) over a real WS;
the SUT decodes → pipeline → encodes → echoes. This is the real-I/O number, no
longer the framework floor. (Laptop, macOS/uvloop, load-gen co-located — which
*penalizes* Rust and is generous to pipecat, since the cheap client leaves Python
its cores.)

| callers | flowcat-rust p50 / p99 | pipecat (1 proc) p50 / p99 | throughput |
| --- | --- | --- | --- |
| 10 | ≤~1 / ~1 ms | 0.61 / **81 ms** | both 100% |
| 25 | ≤~1 / ~1 ms | 2.11 / **198 ms** | both 100% |
| 50 | ≤~1 / ~1 ms | 0.85 / **299 ms** | both 100% |
| 100 | 0.72 / **1.45 ms** | 3.21 / **445 ms** | both 100% |
| 500 | 0.71 / 1.58 ms | — past usable — | Rust 100% |
| 1000 | 1.15 / 2.94 ms | — past usable — | Rust 100% |
| 2000 | 0.81 / **2.22 ms** | — past usable — | Rust 100% |

**Tail latency, not throughput, is the story.** pipecat keeps up on average (low
p50, 100% throughput) but its **p99 is 81 ms at 10 calls and 445 ms at 100** —
GC + GIL stalls across hundreds of asyncio tasks. For real-time voice a 300 ms
tail is an audible glitch. Rust holds **p99 ≤ 3 ms from 10 to 2,000 calls** — at
matched 100 calls that's **~300× lower tail**, and it scales 20× further.

### Docker validation (the reproducible kit) — pipecat given its FAIR deployment

Run via the `bench/` Docker image (arm64 local smoke, 3 s). Here pipecat runs
**multiprocess** (one worker per core, `SO_REUSEPORT`) — its real production shape:

| callers | flowcat-rust (1 proc) p99 | pipecat (8 procs) p99 |
| --- | --- | --- |
| 5 | 0.57 ms | 29.8 ms |
| 50 | 1.67 ms | 248 ms |

Even at 5 callers (<1 per worker) pipecat shows a ~30 ms tail → the GC/GIL jitter
is **intrinsic to each pipeline**, not a saturation effect. Multiprocess doesn't
fix it; it just spreads it.

## Verdict — both halves of the gate met

- **Density/jitter:** large and real end-to-end (not the inflated 525× floor):
  ~300× lower tail at matched concurrency, 20× more concurrent calls per box.
- **Architecture:** one Rust process replaces ~N Python processes and removes the
  PyO3 FFI seam.

→ **GO** — build Flowcat for real. The authoritative, citable numbers come from
the reproducible Azure run below; the laptop + local-Docker numbers above are
directional.

## Azure VM run — AUTHORITATIVE (2026-05-31)

Hardware: Azure **`Standard_FX16mds_v2`** (16 vCPU, compute-optimized), Ubuntu
24.04, one box. (The personal subscription had `D16s_v5`/standard-Dsv5 restricted
`NotAvailableForSubscription`; FX16mds_v2 was the available unrestricted 16-vCPU
SKU — and compute-optimized cores are ideal here.) 12 cores → SUT, 4 → load-gen,
pinned. Both SUTs on the same box, identical Rust load generator, 10 s/point,
50 fps/call. **pipecat in its real multiprocess deployment: 12 workers,
`SO_REUSEPORT`, one per SUT core** (not single-process — Python given every
advantage).

Full RTT distribution per caller count (all values in **ms**). The load
generator records p50 / p90 / p99 / p99.9 / max / mean every run.

**flowcat-rust** (1 process, 12 cores):

| callers | p50 | p90 | p99 | p99.9 | max | mean | throughput |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 0.23 | 0.27 | 0.38 | 0.40 | 0.42 | 0.21 | 100% |
| 25 | 0.24 | 0.30 | 0.41 | 0.45 | 0.49 | 0.24 | 100% |
| 50 | 0.14 | 0.29 | 0.39 | 0.45 | 0.50 | 0.17 | 100% |
| 100 | 0.31 | 0.43 | 0.51 | 0.55 | 0.59 | 0.31 | 100% |
| 250 | 0.27 | 0.45 | 0.59 | 0.73 | 0.82 | 0.29 | 100% |
| 500 | 0.20 | 0.36 | 0.51 | 0.64 | 3.87 | 0.22 | 100% |
| 1000 | 0.19 | 0.31 | 0.47 | 0.66 | 4.36 | 0.20 | 100% |
| 2000 | 0.21 | 0.39 | 0.61 | 1.00 | 35.03 | 0.24 | 100% |

**pipecat-python** (12 processes, `SO_REUSEPORT`, 12 cores):

| callers | p50 | p90 | p99 | p99.9 | max | mean | throughput |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 0.23 | 0.55 | 0.70 | 101.7 | 162.5 | 0.77 | 100% |
| 25 | 0.27 | 0.47 | 0.65 | 73.1 | 149.8 | 0.62 | 100% |
| 50 | 0.46 | 0.79 | 1.13 | 115.0 | 292.3 | 0.97 | 100% |
| 100 | 1.43 | 1.99 | 33.0 | 297.4 | 487.2 | 2.76 | 100% |
| 250 | 3.87 | 5.37 | 50.8 | 841.4 | 1157.6 | 7.98 | 100% |
| 500 | 9.53 | 227.2 | 843.5 | 1777.6 | 2629.4 | 67.4 | 100% |
| 1000 | 1982.5 | 3839.7 | 5673.3 | 6901.6 | 7659.1 | 2137.3 | 77% |
| 2000 | 1911.7 | 3483.4 | 5073.6 | 6036.8 | 6514.6 | 1942.5 | 41% (982 conns failed) |

What the wider percentiles reveal that p99 alone hides:

- **pipecat's tail is bimodal from the very start.** At 10 calls its p50/p90/p99 are
  all sub-ms (0.23/0.55/0.70) — but **p99.9 = 102 ms and max = 163 ms**. ~1 frame in
  1,000 eats a GC/GIL stall. p50 looking healthy hides a tail that's already an
  audible glitch for real-time voice.
- **flowcat-rust stays tight across the whole distribution** — p99.9 ≤ 1 ms through
  1,000 calls; only at 2,000 does a single max of 35 ms appear (p99.9 still 1 ms).
- By 1,000 calls pipecat's **p50 itself is ~2 s** — the whole pipeline is underwater,
  not just the tail.

- **flowcat-rust:** p99 ≤ 0.61 ms and 100% throughput from 10 to 2,000 calls on a
  single process. Flat.
- **pipecat (fair multiprocess):** usable to ~250 calls (p99 ~51 ms); degrades
  hard at 500 (p99 843 ms) and collapses at 1,000+ (p99 5.7 s, throughput < 80%).
- **Intrinsic GC tail:** even at 10 calls pipecat's max/p999 is ~100–160 ms — the
  per-pipeline GC/GIL jitter floor — vs Rust's sub-ms. Multiprocess spreads it,
  doesn't remove it.
- **Net:** on identical hardware, with Python fairly deployed, Rust sustains
  ~8× the concurrent calls at a tight tail (≥2,000 vs ~250) and at matched load
  the tail is 2–4 orders of magnitude lower (e.g. 500 calls: 0.51 ms vs 843 ms).

Reproduce: `bench/README.md` → `docker compose -f bench/compose.yml up --build`
on a 16-vCPU VM. (`run-sweep.sh` staggers pipecat worker launches and waits for
the port to accept before sweeping — a fixed sleep raced the multi-worker import
and measured zero on the first attempt.)

