#!/usr/bin/env python3
"""Metal paths on the length-probe prompts: solo or behind a decoding row; "plen@row" prefills the solo stream's ids up to that row in one pass (teacher forced)."""
import json, sys, threading, time, urllib.request
port, outp, mode, tag = sys.argv[1:5]; keys = sys.argv[5:]
PR = json.load(open(".scratch/logs/oracle/prompts.json"))
LEN = json.load(open(".scratch/logs/exp3/len.json")); LEN.update(json.load(open(".scratch/logs/exp4/nopeel-len.json")))
text = open("docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-16384.txt").read()
BG = "Background row.\n" + text[:150]
def call(p, mt):
    b = {"model": "intarsia", "prompt": p, "max_tokens": mt, "temperature": 0, "ignore_eos": True, "return_token_ids": True, "logprobs": 20}
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(b).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=900))
try: res = json.load(open(outp))
except Exception: res = {}
for k in keys:
    plen, _, row = k.partition("@")
    p = PR[plen]["ids"] + (LEN[plen]["solo"]["ids"][:int(row)] if row else [])
    t = None
    if mode == "behind":
        t = threading.Thread(target=lambda: call(BG, 200)); t.start(); time.sleep(3)
    d = call(p, 3 if row else 200); c = d["choices"][0]
    res[f"{tag}/{k}"] = {"ids": c["token_ids"], "top": c["logprobs"]["top_logprobs_ids"], "plen": d["usage"]["prompt_tokens"]}
    if t: t.join()
    json.dump(res, open(outp, "w"))
