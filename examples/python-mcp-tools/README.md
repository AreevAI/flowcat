<!-- SPDX-License-Identifier: Apache-2.0 -->
# MCP tools (Python) example

Plug your **Python functions** into a Flowcat agent as callable tools — no Rust,
no bindings. You expose functions from an [MCP](https://modelcontextprotocol.io)
server; Flowcat's `mcp` feature connects as a client, lists the tools, exposes
them to the model, and dispatches the model's calls back to your server.

This is the right path when you want the LLM to drive the conversation (via the
system prompt) while your Python code handles the side-effects / business logic.
For driving the conversation *policy* itself from Python, see
[`../python-remote-brain`](../python-remote-brain).

## Run the reference MCP server

```bash
pip install "mcp[cli]"
python3 mcp_server.py        # serves MCP over HTTP at http://127.0.0.1:8000/mcp
```

It exposes one tool, `lookup_order`. Replace it with your own functions.

## Point Flowcat at it (Rust embedder)

Enable the `mcp` feature, then wire the client into the pipeline as a processor:

```rust
use std::sync::Arc;
use flowcat_services::mcp::{HttpMcpTransport, McpClient, McpProcessor}; // feature `mcp`

let transport = HttpMcpTransport::new("http://127.0.0.1:8000/mcp");
let client = McpClient::new(Arc::new(transport));
// Optionally restrict which tools are exposed: .with_tools_filter(["lookup_order"])
let mcp = McpProcessor::new(client);

// Insert `mcp` into the FrameProcessor pipeline; its tools are merged into the
// model's tool set and the model's calls to them are dispatched to your server.
```

> Note: `HttpMcpTransport` blocks private/loopback URLs by default (SSRF guard).
> For this local example, call `.allow_private_url()` on the transport. Do **not**
> allow private URLs for untrusted/remote MCP endpoints in production.
