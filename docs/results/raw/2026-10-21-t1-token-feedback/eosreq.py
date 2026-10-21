#!/usr/bin/env python3
"""eosreq.py <base-url> <model> <n>: n short chat requests back to back that end on EOS (no max_tokens hit), greedy.
With token feedback each one leaves a discarded step in flight; it lands in the next request's prefill. Prints the
wall time of each request and the median, plus completion tokens (identical across arms means the same text)."""
import json, statistics, sys, time, urllib.request

url, model, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
qs = ["Name one primary color. One word.", "What is 2+2? Answer with a number.", "Say hello in French. One word.",
      "Name a planet. One word."]
walls, toks = [], []
for i in range(n):
    body = {"model": model, "temperature": 0, "max_tokens": 64, "messages": [{"role": "user", "content": qs[i % len(qs)]}],
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    r = json.load(urllib.request.urlopen(req))
    w = (time.perf_counter() - t0) * 1e3
    c = r["usage"]["completion_tokens"]
    walls.append(w); toks.append(c)
    print(f"req {i} wall {w:.1f} ms completion {c} finish {r['choices'][0]['finish_reason']} text {r['choices'][0]['message']['content']!r}")
print(f"median wall {statistics.median(walls):.1f} ms (after the first: {statistics.median(walls[1:]):.1f}); completion tokens {toks}")
