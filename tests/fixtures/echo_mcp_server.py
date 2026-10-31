#!/usr/bin/env python3
"""Minimal stdio MCP server for testing lean-coder's MCP client. Raw JSON-RPC over
stdin/stdout, newline-delimited. One tool: echo."""
import sys, json

TOOLS = [{"name": "echo",
          "description": "Echo back the given text.",
          "inputSchema": {"type": "object",
                          "properties": {"text": {"type": "string"}},
                          "required": ["text"]}}]


def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = msg.get("method")
        mid = msg.get("id")
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                "serverInfo": {"name": "echo", "version": "1.0"}}})
        elif method == "notifications/initialized":
            pass  # no response
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            params = msg.get("params", {})
            text = (params.get("arguments") or {}).get("text", "")
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "content": [{"type": "text", "text": f"echo: {text}"}]}})
        elif mid is not None:
            send({"jsonrpc": "2.0", "id": mid,
                  "error": {"code": -32601, "message": f"method not found: {method}"}})


if __name__ == "__main__":
    main()
