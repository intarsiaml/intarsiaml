#!/usr/bin/env python3
"""All prompts: cold solo vs cold behind a decoding row. Same prompts as ident-rt.sh. Saves ids and top-20 logprobs."""
import json, sys, threading, time, urllib.request
port, pr, outp = sys.argv[1:4]; qs = [int(x) for x in sys.argv[4].split(",")]; modes = sys.argv[5].split(",")
text = open(pr).read()
lens = [150, 400, 900, 1800, 3000, 5000, 8000, 11000, 15700, 22000, 31000, 40000]
P = [f"Request {k}, {n} characters of context follows.\n" + text[:n] for k, n in enumerate(lens)]
def call(p, mt, lp):
    b = {"model": "intarsia", "prompt": p, "max_tokens": mt, "temperature": 0, "ignore_eos": True, "return_token_ids": True}
    if lp: b["logprobs"] = 20
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(b).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=900))
res = {}
for q in qs:
    for m in modes:
        print(time.strftime("%X"), "q", q, m, file=sys.stderr, flush=True)
        t = None
        if m == "behind":
            bgp = P[2] if q == 0 else P[0]
            t = threading.Thread(target=lambda: call(bgp, 200, False)); t.start(); time.sleep(3)
        d = call(P[q], 200, True); c = d["choices"][0]; l = c["logprobs"]
        res[f"q{q}-{m}"] = {"ids": c["token_ids"], "top": l["top_logprobs_ids"], "plen": d["usage"]["prompt_tokens"]}
        if t: t.join()
        json.dump(res, open(outp, "w"))
