#!/usr/bin/env python3
"""Flowcat spike — pipecat SUT for the real-I/O harness (driven by io_harness load).

A websockets server that, per connection, runs a REAL pipecat pipeline:

    transport.in (stub) → user_agg → echo-LLM → transport.out (WS sink) → assistant_agg

doing μ-law decode/encode + full pipeline processing + ASAP echo — the same work
the Rust io_harness SUT does. Driven by the SAME Rust load generator, so the only
thing the comparison measures is framework cost under concurrency (GIL vs no-GIL).

We deliberately bypass pipecat's WebsocketServerTransport: it is single-client and
its output transport paces audio to real-time (`_write_audio_sleep`), which would
mask framework latency. Using the websockets lib for multi-connection accept +
manual inject/capture is CONSERVATIVE toward pipecat — it strips transport
overhead and isolates the (real) pipeline cost.

  .venv/bin/python bench/pipecat_sut.py --host 127.0.0.1 --port 9098
"""
from __future__ import annotations

import argparse
import asyncio
import audioop  # G.711 μ-law codec (C; deprecated in 3.13, present in this 3.11 venv)
import os

from loguru import logger

logger.remove()

try:
    import uvloop

    uvloop.install()
    LOOP = "uvloop"
except Exception:  # noqa: BLE001
    LOOP = "asyncio"

from websockets.asyncio.server import serve

from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameProcessor


class PassThrough(FrameProcessor):
    """Stand-in for transport.input()."""

    async def process_frame(self, frame, direction):  # noqa: ANN001
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class EchoLLM(FrameProcessor):
    """Stand-in for GeminiLive: turn inbound audio into outbound audio (echo)."""

    async def process_frame(self, frame, direction):  # noqa: ANN001
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            await self.push_frame(
                OutputAudioRawFrame(
                    audio=frame.audio,
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                )
            )
        else:
            await self.push_frame(frame, direction)


class WSSink(FrameProcessor):
    """Stand-in for transport.output(): μ-law encode + send on the WS, ASAP."""

    def __init__(self, send_cb, **kw):
        super().__init__(**kw)
        self._send_cb = send_cb

    async def process_frame(self, frame, direction):  # noqa: ANN001
        await super().process_frame(frame, direction)
        if isinstance(frame, OutputAudioRawFrame):
            await self._send_cb(frame.audio)
        await self.push_frame(frame, direction)


async def handle(ws):
    ctx = LLMContext()
    user_agg, assistant_agg = LLMContextAggregatorPair(ctx)

    async def send_cb(pcm: bytes):
        try:
            await ws.send(audioop.lin2ulaw(pcm, 2))
        except Exception:  # noqa: BLE001
            pass

    pipeline = Pipeline([PassThrough(), user_agg, EchoLLM(), WSSink(send_cb), assistant_agg])
    task = PipelineTask(pipeline, cancel_on_idle_timeout=False, params=PipelineParams())
    runner = PipelineRunner(handle_sigint=False)
    run = asyncio.create_task(runner.run(task))
    await asyncio.sleep(0.05)  # let StartFrame propagate before injecting audio

    try:
        async for msg in ws:
            if isinstance(msg, (bytes, bytearray)):
                pcm = audioop.ulaw2lin(bytes(msg), 2)
                await task.queue_frame(
                    InputAudioRawFrame(audio=pcm, sample_rate=8000, num_channels=1)
                )
    except Exception:  # noqa: BLE001
        pass
    finally:
        try:
            await task.cancel()
        except Exception:  # noqa: BLE001
            pass
        try:
            await asyncio.wait_for(run, timeout=10)
        except Exception:  # noqa: BLE001
            pass


async def main(host: str, port: int, reuse_port: bool) -> None:
    async with serve(handle, host, port, reuse_port=reuse_port):
        print(
            f"[serve] pipecat SUT ({LOOP}) pid={os.getpid()} on ws://{host}:{port} "
            f"reuse_port={reuse_port}",
            flush=True,
        )
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9098)
    ap.add_argument(
        "--reuse-port",
        action="store_true",
        help="SO_REUSEPORT: run one process per core, kernel load-balances connections",
    )
    args = ap.parse_args()
    asyncio.run(main(args.host, args.port, args.reuse_port))
