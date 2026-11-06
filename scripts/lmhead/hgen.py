#!/usr/bin/env python3
"""usage: hgen.py BASE_URL MODEL PROMPT_DIR OUT_DIR name... : 128 greedy tokens per prompt (raw completion, no chat template) from a running server."""
import json, sys, urllib.request, os
base, model, d, out, names = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
os.makedirs(out, exist_ok=True)
for n in names:
    body = json.dumps({"model": model, "prompt": open(f"{d}/{n}.txt").read(), "max_tokens": 128, "min_tokens": 128, "ignore_eos": True, "temperature": 0}).encode()
    req = urllib.request.Request(base + "/v1/completions", body, {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        j = json.load(r)
    t = j["choices"][0]["text"]
    open(f"{out}/{n}.txt", "w").write(t)
    print(n, len(t), "chars", j.get("usage"), flush=True)
