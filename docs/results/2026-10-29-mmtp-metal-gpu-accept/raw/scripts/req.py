#!/usr/bin/env python3
"""MMTP diagnostic request: req.py PORT PROMPT_FILE MIN_TOKENS (greedy, 256 tokens, ignore_eos)."""
import json, sys, time, urllib.request
port, pf, mt = sys.argv[1], sys.argv[2], int(sys.argv[3])
body = {"model": "intarsia", "prompt": "nonce 8812. " + open(pf).read(), "max_tokens": 256, "temperature": 0, "ignore_eos": True, "min_tokens": mt}
t = time.time()
r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600)
j = json.load(r)
print("min_tokens", mt, j.get("usage"), f"{time.time() - t:.2f}s")
