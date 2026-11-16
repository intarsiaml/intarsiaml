#!/usr/bin/env python3
"""Time to first streamed token and inter-token gaps of one /v1/completions call. usage: ttft.py URL PROMPTFILE NONCE [max_tokens]"""
import json, sys, time, urllib.request
url, pf, nonce = sys.argv[1], sys.argv[2], sys.argv[3]
mt = int(sys.argv[4]) if len(sys.argv) > 4 else 48
body = json.dumps({"model": "intarsia", "prompt": f"[{nonce}] " + open(pf).read(), "max_tokens": mt, "temperature": 0, "stream": True,
                   "stream_options": {"include_usage": True}}).encode()
t0 = time.perf_counter(); ts = []; usage = None
with urllib.request.urlopen(urllib.request.Request(url + "/completions", body, {"Content-Type": "application/json"})) as r:
    for line in r:
        if not line.startswith(b"data: ") or b"[DONE]" in line: continue
        d = json.loads(line[6:]); usage = d.get("usage") or usage
        if d.get("choices") and d["choices"][0].get("text"): ts.append(time.perf_counter() - t0)
gaps = [b - a for a, b in zip(ts, ts[1:])]
print(f"nonce {nonce}: ttft {ts[0]*1e3:.0f} ms, events {len(ts)}, first gaps ms {[round(g*1e3,1) for g in gaps[:6]]}, total {ts[-1]*1e3:.0f} ms, usage {usage}")
