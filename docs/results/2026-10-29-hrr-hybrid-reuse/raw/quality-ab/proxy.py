#!/usr/bin/env python3
"""Logging proxy: PORT UPSTREAM_PORT LOG.jsonl. Forwards POSTs, logs prompt_tokens and cached_tokens per request."""
import sys, json, http.client, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
port, up, log = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
lk = threading.Lock()
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def _fwd(self):
        n = int(self.headers.get("Content-Length") or 0); body = self.rfile.read(n) if n else b""
        c = http.client.HTTPConnection("127.0.0.1", up, timeout=900)
        c.request(self.command, self.path, body, {k: v for k, v in self.headers.items() if k.lower() not in ("host", "connection")})
        r = c.getresponse(); data = r.read()
        self.send_response(r.status)
        for k, v in r.getheaders():
            if k.lower() not in ("transfer-encoding", "connection", "content-length"): self.send_header(k, v)
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        if self.command == "POST" and r.status == 200 and ("completions" in self.path):
            u = None
            try:
                t = data.decode()
                if t.lstrip().startswith("{"): u = json.loads(t).get("usage")
                else:
                    for ln in t.splitlines():
                        if ln.startswith("data:") and "usage" in ln and not ln.endswith("[DONE]"):
                            j = json.loads(ln[5:]); u = j.get("usage") or u
            except Exception: pass
            if u:
                rec = {"t": time.time(), "prompt_tokens": u.get("prompt_tokens"), "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"), "completion_tokens": u.get("completion_tokens")}
                with lk, open(log, "a") as f: f.write(json.dumps(rec) + "\n")
    do_GET = do_POST = _fwd
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
