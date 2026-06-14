#!/usr/bin/env python3
"""Flowcat spike — pipecat per-session cost profiler.

Measures the pipecat *framework floor* for a realistic speech-to-speech
topology:

    transport.input() -> user_agg -> GeminiLive -> transport.output() -> assistant_agg

The transport + LLM are stubbed as no-op PassThrough processors so we isolate
the runtime overhead that actually drives concurrent-session density:
per-processor asyncio tasks + queues, plus the real LLMContext aggregators.

So every number here is a LOWER BOUND on real per-session cost — a live call
adds a real WS transport (sockets, serializer) and a Gemini WS client on top.
If the framework floor alone is already the density ceiling, that's the
strongest possible signal that a Rust runtime (cheap tokio tasks, no GIL) wins.

Modes (run each in a FRESH process so ru_maxrss == resident set):
  memory     --sessions N   resident RAM + asyncio-task count per idle session
  throughput --frames K     per-frame CPU through the 5-processor pipeline (1 core)

Run inside a venv with pipecat installed:
  .venv/bin/python bench/pipecat_profile.py memory --sessions 200
  .venv/bin/python bench/pipecat_profile.py throughput --frames 20000
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import os
import resource
import subprocess
import sys
import time

# Silence pipecat's loguru banner + logs before importing pipecat.
from loguru import logger

logger.remove()

# Match a production deployment: run on uvloop. Fall back to stdlib asyncio.
try:
    import uvloop

    uvloop.install()
    LOOP = "uvloop"
except Exception:  # noqa: BLE001
    LOOP = "asyncio"

from pipecat.frames.frames import InputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.pipeline.base_task import PipelineTaskParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair


def maxrss_bytes() -> int:
    """Peak RSS. macOS reports bytes; Linux reports kilobytes."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024


def cur_rss_bytes() -> int:
    """Current (resident) RSS via ps — ps reports KB on both macOS and Linux.

    Unlike ru_maxrss (peak), this reflects steady-state resident memory, so the
    delta isn't contaminated by the transient spike of starting many sessions.
    """
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])
    return int(out.strip()) * 1024


class PassThrough(FrameProcessor):
    """No-op stand-in for transport.input()/output() and the GeminiLive service."""

    async def process_frame(self, frame, direction: FrameDirection):  # noqa: ANN001
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


def make_session() -> PipelineTask:
    """Build one faithful 5-processor speech-to-speech PipelineTask (idle-able)."""
    ctx = LLMContext()
    user_agg, assistant_agg = LLMContextAggregatorPair(ctx)
    pipeline = Pipeline(
        [
            PassThrough(),  # transport.input()
            user_agg,
            PassThrough(),  # GeminiLive (server-side STT+LLM+TTS) — stubbed
            PassThrough(),  # transport.output()
            assistant_agg,
        ]
    )
    return PipelineTask(
        pipeline,
        cancel_on_idle_timeout=False,  # idle live call must NOT self-terminate
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )


async def _start(task: PipelineTask, loop) -> asyncio.Task:  # noqa: ANN001
    return asyncio.create_task(task.run(PipelineTaskParams(loop=loop)))


async def _teardown(sessions, bgs) -> None:  # noqa: ANN001
    for s in sessions:
        try:
            await s.cancel()
        except Exception:  # noqa: BLE001
            pass
    try:
        await asyncio.wait_for(asyncio.gather(*bgs, return_exceptions=True), timeout=30)
    except asyncio.TimeoutError:
        pass


async def mode_memory(n: int) -> None:
    loop = asyncio.get_running_loop()

    # Warm up one full session so lazy imports / one-time allocations don't
    # land in the measured delta.
    warm = make_session()
    warm_bg = await _start(warm, loop)
    await asyncio.sleep(1.0)
    await _teardown([warm], [warm_bg])
    gc.collect()
    await asyncio.sleep(0.3)

    tasks_before = len(asyncio.all_tasks())
    cur_before = cur_rss_bytes()

    sessions = [make_session() for _ in range(n)]
    bgs = [await _start(s, loop) for s in sessions]
    await asyncio.sleep(3.0)  # let StartFrames propagate; processors spin up + idle
    gc.collect()
    await asyncio.sleep(0.5)  # let GC + transient startup allocations drain
    gc.collect()

    cur_after = cur_rss_bytes()
    peak = maxrss_bytes()
    tasks_after = len(asyncio.all_tasks())

    rss_per = (cur_after - cur_before) / n
    tasks_per = (tasks_after - tasks_before) / n

    print(f"[memory] loop={LOOP} sessions={n}")
    print(f"  rss_before_mb      = {cur_before / 1e6:.1f}  (current/resident)")
    print(f"  rss_after_mb       = {cur_after / 1e6:.1f}  (current/resident)")
    print(f"  peak_rss_mb        = {peak / 1e6:.1f}  (ru_maxrss, for reference)")
    print(f"  rss_per_session_kb = {rss_per / 1024:.1f}")
    print(f"  tasks_before       = {tasks_before}")
    print(f"  tasks_after        = {tasks_after}")
    print(f"  tasks_per_session  = {tasks_per:.1f}")
    print(
        f"  -> 1000 sessions ~ {rss_per * 1000 / 1e9:.2f} GB RAM, "
        f"~{tasks_per * 1000:.0f} asyncio tasks (framework floor)"
    )

    await _teardown(sessions, bgs)


async def mode_throughput(k: int) -> None:
    from pipecat.tests.utils import run_test

    ctx = LLMContext()
    user_agg, assistant_agg = LLMContextAggregatorPair(ctx)
    inner = Pipeline(
        [
            PassThrough(),  # transport.input()
            user_agg,
            PassThrough(),  # GeminiLive — stubbed
            PassThrough(),  # transport.output()
            assistant_agg,
        ]
    )

    # 160 bytes = one 20ms frame of 8kHz/16-bit mono — a telephony audio frame.
    audio = b"\xff" * 160
    frames = [
        InputAudioRawFrame(audio=audio, sample_rate=8000, num_channels=1)
        for _ in range(k)
    ]

    t0 = time.perf_counter()
    await run_test(inner, frames_to_send=frames, send_end_frame=True)
    dt = time.perf_counter() - t0

    # run_test wraps the pipeline as [source, inner(5), sink] => 7 processor hops/frame.
    hops = 7
    print(f"[throughput] loop={LOOP} frames={k}")
    print(f"  elapsed_s          = {dt:.3f}")
    print(f"  frames_per_sec     = {k / dt:,.0f}")
    print(f"  us_per_frame       = {dt / k * 1e6:.2f}")
    print(f"  us_per_proc_hop    = {dt / k / hops * 1e6:.2f}")
    print(
        f"  -> a live call is ~50 frames/s each way (~100/s). One core sustains "
        f"~{int(k / dt / 100):,} such calls before CPU-bound on framework alone."
    )


async def mode_scale(checkpoints: list[int]) -> None:
    """Marginal per-session RAM: grow sessions in one process and measure the
    incremental RSS between checkpoints. The slope cancels the fixed startup
    high-water (pymalloc arenas), giving the true per-session cost."""
    loop = asyncio.get_running_loop()

    warm = make_session()
    warm_bg = await _start(warm, loop)
    await asyncio.sleep(1.0)
    await _teardown([warm], [warm_bg])
    gc.collect()
    await asyncio.sleep(0.3)

    sessions: list[PipelineTask] = []
    bgs: list[asyncio.Task] = []
    prev_n = 0
    base = cur_rss_bytes()
    prev_rss = base
    print(f"[scale] loop={LOOP} baseline_mb={base / 1e6:.1f}")
    for target in checkpoints:
        while len(sessions) < target:
            s = make_session()
            sessions.append(s)
            bgs.append(await _start(s, loop))
        await asyncio.sleep(2.5)
        gc.collect()
        await asyncio.sleep(0.4)
        gc.collect()
        rss = cur_rss_bytes()
        marg = (rss - prev_rss) / (target - prev_n) if target > prev_n else 0
        cum = (rss - base) / target
        print(
            f"  n={target:5d}  rss_mb={rss / 1e6:7.1f}  "
            f"marginal_kb/sess={marg / 1024:6.1f}  cum_kb/sess={cum / 1024:6.1f}  "
            f"tasks={len(asyncio.all_tasks())}"
        )
        prev_rss, prev_n = rss, target

    await _teardown(sessions, bgs)


def main() -> None:
    ap = argparse.ArgumentParser(description="pipecat per-session cost profiler")
    sub = ap.add_subparsers(dest="mode", required=True)
    m = sub.add_parser("memory")
    m.add_argument("--sessions", type=int, default=200)
    t = sub.add_parser("throughput")
    t.add_argument("--frames", type=int, default=20000)
    sc = sub.add_parser("scale")
    sc.add_argument("--checkpoints", type=str, default="200,500,1000,1500")
    args = ap.parse_args()

    if args.mode == "memory":
        asyncio.run(mode_memory(args.sessions))
    elif args.mode == "throughput":
        asyncio.run(mode_throughput(args.frames))
    else:
        asyncio.run(mode_scale([int(x) for x in args.checkpoints.split(",")]))


if __name__ == "__main__":
    main()
