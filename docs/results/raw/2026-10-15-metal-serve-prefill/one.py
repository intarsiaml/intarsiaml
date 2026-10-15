#!/usr/bin/env python3
# usage: one.py PORT MODEL PROMPT : one nonce-prefixed 4k request (max_tokens 1, non-stream); prints
# "ttft_ms srv_prefill_ms t0_epoch" (server prefill from /metrics last_prefill_tokens_per_second).
import sys, json, time, urllib.request, secrets, re
port, model, pf = sys.argv[1:4]
txt = open(pf).read()
base = f"http://127.0.0.1:{port}"
body = json.dumps({"model": model, "messages": [{"role": "user", "content": secrets.token_hex(6) + " " + txt}],
                   "max_tokens": 1, "temperature": 0}).encode()
t0e = time.time(); t0 = time.perf_counter()
r = json.load(urllib.request.urlopen(urllib.request.Request(base + "/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=300))
dt = (time.perf_counter() - t0) * 1e3
pt = r["usage"]["prompt_tokens"]
m = urllib.request.urlopen(base + "/metrics", timeout=10).read().decode()
tps = float(re.search(r"^intarsia_last_prefill_tokens_per_second\s+([0-9.eE+-]+)", m, re.M).group(1))
print("%.0f %.0f %.3f" % (dt, pt / tps * 1e3, t0e))
