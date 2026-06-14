<!-- SPDX-License-Identifier: Apache-2.0 -->
# Flowcat examples

Companion examples for using Flowcat — including from Python, without writing Rust
or using in-process bindings. The Rust demo binary lives separately in
[`../flowcat-cli`](../flowcat-cli) (`flowcat pipeline`, `flowcat ws-echo`).

| Example | What it shows |
| --- | --- |
| [`python-remote-brain/`](python-remote-brain) | Drive the **conversation policy** from a Python HTTP service via the `RemoteBrain` adapter (`brain-http` feature). Pure-stdlib reference server. |
| [`python-mcp-tools/`](python-mcp-tools) | Expose **Python functions as agent tools** over MCP, consumed by Flowcat's `mcp` client. |

Both keep your Python at **turn granularity** (between turns) — the latency-
critical media loop stays in Rust, so Flowcat's tail-latency profile is preserved.
See the [README](../README.md) "Using Flowcat from Python" section for the model.
