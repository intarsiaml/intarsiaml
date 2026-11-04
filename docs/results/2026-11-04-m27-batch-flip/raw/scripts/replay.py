#!/usr/bin/env python3
"""q4 replay driver: runs named cells against a running server; each cell: optional background request, then q4. Saves ids and top-20 logprobs."""
import json, sys, threading, time, urllib.request
port, pr, outp = sys.argv[1:4]; cells = sys.argv[4:]
text = open(pr).read()
def prompt(k, n): return f"Request {k}, {n} characters of context follows.\n" + text[:n]
Q4 = prompt(4, 3000)
BGS = {"q0": prompt(0, 150), "q3": prompt(3, 1800), "q2": prompt(2, 900)}
def call(p, mt, lp):
    b = {"model": "intarsia", "prompt": p, "max_tokens": mt, "temperature": 0, "ignore_eos": True, "return_token_ids": True}
    if lp: b["logprobs"] = 20
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(b).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=900))
res = {}
try: res = json.load(open(outp))
except Exception: pass
def rec(d):
    c = d["choices"][0]; l = c.get("logprobs") or {}
    return {"ids": c.get("token_ids"), "token_logprobs": l.get("token_logprobs"), "top": l.get("top_logprobs_ids"), "n": d["usage"]["completion_tokens"]}
# cell syntax: name:mode[:lp]  mode = solo | behind-<bg> (q4 starts while bg decodes) | ahead-<bg> (bg starts while q4 decodes)
for i, cell in enumerate(cells):
    name, mode, *rest = cell.split(":"); lp = rest != ["nolp"]
    print(time.strftime("%X"), "CELL", name, mode, "lp" if lp else "nolp", flush=True, file=sys.stderr)
    bgt = None
    if mode.startswith("behind-") or mode.startswith("ahead-"):
        order, bgn = mode.split("-")
        bgres = {}
        def bg(): bgres["d"] = call(BGS[bgn], 500, False)
        bgt = threading.Thread(target=bg)
    if mode.startswith("behind-"):
        bgt.start(); time.sleep(4 if bgn == "q3" else 3)
    if mode.startswith("ahead-"):
        qres = {}
        def q(): qres["d"] = call(Q4, 200, lp)
        qt = threading.Thread(target=q); qt.start(); time.sleep(3); bgt.start(); qt.join(); d = qres["d"]
    else:
        d = call(Q4, 200, lp)
    res[f"{i:02d}-{name}"] = rec(d)
    if bgt: bgt.join()
    json.dump(res, open(outp, "w"))
    time.sleep(0.5)
