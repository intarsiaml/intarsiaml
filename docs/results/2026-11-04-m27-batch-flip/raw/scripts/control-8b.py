#!/usr/bin/env python3
import json, sys, time, urllib.request
port, outp, keys = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
P = json.load(open(".scratch/logs/gates/prompts-text.json"))
res = {}
for k in keys:
    t0 = time.time()
    b = {"model": "intarsia", "prompt": P[k], "max_tokens": 200, "temperature": 0, "ignore_eos": True, "return_token_ids": True, "logprobs": 20}
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(b).encode(), {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(r, timeout=2400)); c = d["choices"][0]
    res[k] = {"ids": c["token_ids"], "top": c["logprobs"]["top_logprobs_ids"], "plen": d["usage"]["prompt_tokens"], "wall": time.time() - t0}
    json.dump(res, open(outp, "w")); print(k, "wall", round(time.time() - t0), flush=True)
