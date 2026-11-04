#!/usr/bin/env python3
"""One request to a CPU reference server: prompt ids, greedy with logprobs. Saves ids, top20 rows, wall."""
import json, sys, time, urllib.request
port, outp, key, mt = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
P = json.load(open(".scratch/logs/oracle/prompts.json"))[key]["ids"]
if len(sys.argv) > 5:  # teacher-force generated ids from a metal run: file, cell, k
    f, cell, k = sys.argv[5], sys.argv[6], int(sys.argv[7]); P = P + json.load(open(f))[key][cell]["ids"][:k]
t0 = time.time()
b = {"model": "intarsia", "prompt": P, "max_tokens": mt, "temperature": 0, "ignore_eos": True, "return_token_ids": True, "logprobs": 20}
r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(b).encode(), {"Content-Type": "application/json"})
d = json.load(urllib.request.urlopen(r, timeout=float(__import__("os").environ.get("PROBE_TIMEOUT","1200")))); c = d["choices"][0]
try: res = json.load(open(outp))
except Exception: res = {}
tag = key + ("" if len(sys.argv) <= 5 else f"@{sys.argv[6]}{sys.argv[7]}")
res[tag] = {"ids": c["token_ids"], "top": c["logprobs"]["top_logprobs_ids"], "wall": time.time() - t0, "plen": d["usage"]["prompt_tokens"]}
json.dump(res, open(outp, "w")); print(tag, "wall", round(time.time() - t0, 1), flush=True)
