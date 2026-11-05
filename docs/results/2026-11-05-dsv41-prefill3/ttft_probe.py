#!/usr/bin/env python3
"""Lane dsv41/prefill3: time to the first stream event and the first content token of a streamed chat request, for combinations of
ignore_eos / min_tokens / max_tokens (localeval's speed sweep sends ignore_eos + min_tokens = max_tokens when it asks for more than
one token), on one filler prompt with a fresh first line each time.

Usage: ttft_probe.py BASE_URL PROMPT_FILE [REPS]
Prints one line per request: variant, prompt_tokens, ms to the first SSE data event, ms to the first non-empty content, total ms.
"""
import http.client
import json
import random
import sys
import time
import urllib.parse

base, path = sys.argv[1], sys.argv[2]
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
u = urllib.parse.urlparse(base)
text = open(path).read()
VARIANTS = [
    ("plain mt=1", dict(max_tokens=1)),
    ("plain mt=8", dict(max_tokens=8)),
    ("plain mt=128", dict(max_tokens=128)),
    ("ignore_eos mt=128", dict(max_tokens=128, ignore_eos=True)),
    ("min_tokens=128 mt=128", dict(max_tokens=128, min_tokens=128)),
    ("ignore_eos+min_tokens=128 mt=128 (localeval)", dict(max_tokens=128, ignore_eos=True, min_tokens=128)),
]
for rep in range(reps):
    for name, extra in VARIANTS:
        body = {"model": "dsv41-intarsia", "messages": [{"role": "user", "content": f"Run id {random.random()}.\n" + text}],
                "temperature": 0, "stream": True, "stream_options": {"include_usage": True}, **extra}
        c = http.client.HTTPConnection(u.hostname, u.port, timeout=300)
        c.request("POST", u.path.rstrip("/") + "/chat/completions", json.dumps(body), {"content-type": "application/json"})
        t0 = time.perf_counter()
        r = c.getresponse()
        first = content = None
        ptoks = None
        buf = b""
        while True:
            chunk = r.read1(65536)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                ev, buf = buf.split(b"\n\n", 1)
                if not ev.startswith(b"data:") or ev.strip() == b"data: [DONE]":
                    continue
                now = time.perf_counter() - t0
                if first is None:
                    first = now
                try:
                    d = json.loads(ev[5:])
                except ValueError:
                    continue
                if d.get("usage"):
                    ptoks = d["usage"].get("prompt_tokens")
                for ch in d.get("choices", []):
                    delta = ch.get("delta", {})
                    if (delta.get("content") or delta.get("reasoning_content")) and content is None:
                        content = now
        total = time.perf_counter() - t0
        f = lambda x: "-" if x is None else f"{1000 * x:7.0f}"
        print(f"ttft {name:46s} prompt {ptoks} first_event {f(first)} ms first_content {f(content)} ms total {f(total)} ms", flush=True)
        c.close()
