# SPDX-License-Identifier: Apache-2.0
"""Reference "remote brain" server for Flowcat — pure Python standard library.

Flowcat's `RemoteBrain` adapter (flowcat-services, feature `brain-http`) drives a
call's *conversation policy* by calling two JSON endpoints on a service you run.
This is the credential-free, no-Rust, no-bindings way to put your Python logic in
charge of what the agent says and does, while the latency-critical media loop
stays in Rust. Your code is consulted at *turn granularity* (between turns), never
on the per-audio-frame path.

Run it:

    python3 brain_server.py            # listens on 127.0.0.1:8080

Then point Flowcat's RemoteBrain at it:

    RemoteBrain::connect("http://127.0.0.1:8080", brain_config, "gemini", None)

This demo implements a tiny receptionist flow (greeting -> confirm -> end) to show
all three actions (`transition`, `stay`, `end`) and how collected variables
accumulate. Replace the `decide()` logic with your own — an LLM call, a database
lookup, a state machine, whatever — keeping the request/response shapes.

No third-party packages required (only the Python standard library).
"""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST, PORT = "127.0.0.1", 8080


# --- Tool sets per conversation node (returned to Flowcat as `tools`). ----------
def tool(name, description, params=None):
    return {"name": name, "description": description, "params": params or {"type": "object"}}


GREETING_TOOLS = [
    tool("book_appointment", "Caller wants to book an appointment.",
         {"type": "object", "properties": {"day": {"type": "string"}}}),
    tool("end_call", "Caller is done; end the call."),
]
CONFIRM_TOOLS = [
    tool("confirm", "Caller confirms the appointment."),
    tool("end_call", "Caller is done; end the call."),
]


def session():
    """POST /session -> seed the initial conversation state."""
    return {
        "system_prompt": "You are a friendly receptionist. Greet the caller and "
                         "ask how you can help.",
        "tools": GREETING_TOOLS,
        "node_id": "greeting",
        "collected_vars": {},
    }


def decide(req):
    """POST /tool-call -> interpret the model's tool call into the next action.

    `req` = { "node_id", "tool": {"name", "args"}, "collected_vars" }.
    Returns the documented response shape.
    """
    name = req["tool"]["name"]
    args = req["tool"].get("args") or {}
    vars_ = dict(req.get("collected_vars") or {})

    if name == "book_appointment":
        vars_["requested_day"] = args.get("day", "unspecified")
        return {
            "action": "transition",
            "system_prompt": "Confirm the appointment day with the caller, then ask "
                            "them to say 'confirm'.",
            "tools": CONFIRM_TOOLS,
            "say": f"Sure — booking you for {vars_['requested_day']}. Shall I confirm?",
            "node_id": "confirm",
            "collected_vars": vars_,
            "finished": False,
        }
    if name == "confirm":
        vars_["confirmed"] = True
        return {
            "action": "end",
            "disposition": "appointment_booked",
            "node_id": "wrapup",
            "collected_vars": vars_,
            "finished": True,
        }
    if name == "end_call":
        return {
            "action": "end",
            "disposition": "caller_hung_up",
            "node_id": req["node_id"],
            "collected_vars": vars_,
            "finished": True,
        }
    # Unknown tool: stay put, change nothing.
    return {
        "action": "stay",
        "node_id": req["node_id"],
        "collected_vars": vars_,
        "finished": False,
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            req = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return self._send({"error": "invalid json"}, 400)

        # (A real server would authenticate the `Authorization: Bearer` header here.)
        if self.path == "/session":
            self._send(session())
        elif self.path == "/tool-call":
            self._send(decide(req))
        else:
            self._send({"error": "not found"}, 404)

    def log_message(self, fmt, *args):  # quieter logging
        print(f"[brain] {self.path} {fmt % args}")


if __name__ == "__main__":
    print(f"remote-brain reference server on http://{HOST}:{PORT}  (Ctrl-C to stop)")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
