#!/usr/bin/env python3
"""Real concurrent bursts at max-batch 2 with q4's queue position set by small arrival gaps. Trace showed idle-engine admissions are the 1st and 3rd of a burst
(the first two rows finish one iteration apart). Prediction: q4 at position 1 or 3 runs C=2048 and gives 557 at token 184; position 2, 4 or 5 runs C=256 and gives 14663."""
import json, sys, threading, time, urllib.request
port, pr, outp = sys.argv[1:4]
text = open(pr).read()
lens = [150, 400, 900, 1800, 3000]
P = {k: f"Request {k}, {lens[k]} characters of context follows.\n" + text[:lens[k]] for k in range(5)}
def call(k, res, tag):
    b = {"model": "intarsia", "prompt": P[k], "max_tokens": 200, "temperature": 0, "ignore_eos": True, "return_token_ids": True}
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(b).encode(), {"Content-Type": "application/json"})
    res[tag] = json.load(urllib.request.urlopen(r, timeout=900))["choices"][0]["token_ids"]
out = {}
for rep in range(3):
    for order in ([4, 0, 1, 2], [0, 4, 1, 2], [0, 1, 4, 2], [0, 1, 2, 4], [0, 1, 2, 3, 4]):
        res = {}; ts = []
        for k in order:
            t = threading.Thread(target=call, args=(k, res, f"q{k}")); t.start(); ts.append(t); time.sleep(0.15)
        [t.join() for t in ts]
        out[f"{len(out):02d}-pos{order.index(4) + 1}"] = {"ids": res["q4"], "order": order}
        json.dump(out, open(outp, "w"))
