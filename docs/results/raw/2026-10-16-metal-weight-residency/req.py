#!/usr/bin/env python3
"""One streamed completion against `intarsia serve`: prints a JSON line with ttft_ms (request sent -> first token chunk),
itl_ms (gaps between the following chunks), total_ms, text, usage. Prompt = prompt-4062.txt with a nonce line in front
(so no request shares a prefix with an earlier one), `req.py <port> <nonce> <max_tokens>`."""
import http.client, json, sys, time
port, nonce, max_tokens = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
R = "/opt/code/intarsiaml-cbgap/docs/results/raw/2026-10-16-metal-prefill-cbgap"
prompt = f"Reference {nonce}.\n" + open(f"{R}/prompt-4062.txt").read()
body = json.dumps({"model": "qwen3", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "stream": True})
c = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
t0 = time.perf_counter()
c.request("POST", "/v1/completions", body, {"content-type": "application/json"})
r = c.getresponse()
stamps, text, usage = [], [], None
buf = b""
while True:
    chunk = r.read1(65536)
    if not chunk:
        break
    buf += chunk
    while b"\n\n" in buf:
        ev, buf = buf.split(b"\n\n", 1)
        for line in ev.split(b"\n"):
            if not line.startswith(b"data:"):
                continue
            d = line[5:].strip()
            if d == b"[DONE]":
                continue
            j = json.loads(d)
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices", []):
                if ch.get("text"):
                    stamps.append(time.perf_counter())
                    text.append(ch["text"])
t1 = time.perf_counter()
print(json.dumps({"status": r.status, "ttft_ms": (stamps[0] - t0) * 1e3 if stamps else None,
                  "itl_ms": [(b - a) * 1e3 for a, b in zip(stamps, stamps[1:])], "total_ms": (t1 - t0) * 1e3,
                  "chunks": len(stamps), "text": "".join(text), "usage": usage}))
