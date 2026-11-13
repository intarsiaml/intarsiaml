#!/usr/bin/env python3
"""Tiny fake OpenAI-compatible streaming server for testing replay2.py locally (no GPU). Not for benchmarking.
  python3 _fake_server.py [port]   # default 8765; /v1/models, /v1/chat/completions (SSE with usage + cached_tokens)
Prefix cache is faked: cached_tokens = longest previously seen prompt prefix length (chars/4)."""
import json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

seen = []

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _json(self, o):
        b = json.dumps(o).encode(); self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        self._json({"data": [{"id": "guidance"}]})
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = json.dumps(req["messages"]); ptok = len(prompt) // 4
        cached = 0
        for s in seen:
            n = 0
            while n < min(len(s), len(prompt)) and s[n] == prompt[n]: n += 1
            cached = max(cached, n // 4)
        seen.append(prompt)
        n_out = min(req.get("max_tokens", 16), 24)
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Connection", "close"); self.end_headers()
        time.sleep(0.2)  # fake prefill
        def send(o): self.wfile.write(b"data: " + json.dumps(o).encode() + b"\n\n"); self.wfile.flush()
        for i in range(n_out):
            send({"choices": [{"delta": {"content": f"w{i} "}}]}); time.sleep(0.02)
        send({"choices": [{"delta": {}, "finish_reason": "length"}]})
        send({"choices": [], "usage": {"prompt_tokens": ptok, "completion_tokens": n_out, "prompt_tokens_details": {"cached_tokens": cached}}})
        self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush(); self.close_connection = True

ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 8765), H).serve_forever()
