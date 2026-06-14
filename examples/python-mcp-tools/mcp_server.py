# SPDX-License-Identifier: Apache-2.0
"""Minimal MCP server exposing a Python function as an agent tool.

Flowcat's `mcp` feature is an MCP (Model Context Protocol) *client*: it connects
to an MCP server, lists its tools, exposes them to the model as callable
functions, and dispatches the model's calls to the server. That makes MCP the
clean way to plug your **Python business functions** into a Flowcat agent as
tools — no Rust, no bindings.

This server exposes one tool, `lookup_order`, over streamable HTTP. It uses the
official `mcp` Python package:

    pip install "mcp[cli]"
    python3 mcp_server.py        # serves MCP over HTTP on 127.0.0.1:8000/mcp

Then enable Flowcat's `mcp` feature and point its HttpMcpTransport at the URL
(see README.md). Replace `lookup_order` with your own functions.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("flowcat-demo-tools", host="127.0.0.1", port=8000)


@mcp.tool()
def lookup_order(order_id: str) -> dict:
    """Look up the status of an order by its id."""
    # A real implementation would hit your database / API here.
    return {"order_id": order_id, "status": "shipped", "eta_days": 2}


if __name__ == "__main__":
    # Streamable HTTP transport (served at /mcp).
    mcp.run(transport="streamable-http")
