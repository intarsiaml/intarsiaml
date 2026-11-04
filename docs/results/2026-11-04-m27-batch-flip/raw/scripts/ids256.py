#!/usr/bin/env python3
"""256 greedy ids per prompt length, solo and behind a decoding row, against one server. ids only, the serving path (LastArgmax)."""
import json, sys, threading, time, urllib.request
port, outp, tag, lens = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
P = json.load(open(".scratch/logs/gates/prompts-text.json"))
text = open("docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-16384.txt").read()
BG = "Background row.\n" + text[:150]
def call(p, mt):
    b = {"model": "intarsia", "prompt": p, "max_tokens": mt, "temperature": 0, "ignore_eos": True, "return_token_ids": True}
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(b).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=900))
try: res = json.load(open(outp))
except Exception: res = {}
for k in lens.split(","):
    for m in ("solo", "behind"):
        t = None
        if m == "behind":
            t = threading.Thread(target=lambda: call(BG, 300)); t.start(); time.sleep(3)
        d = call(P[k], 256)
        res[f"{tag}/{k}/{m}"] = {"ids": d["choices"][0]["token_ids"], "plen": d["usage"]["prompt_tokens"]}
        if t: t.join()
        json.dump(res, open(outp, "w"))
