#!/usr/bin/env python3
"""usage: lf_client.py PORT OUTJSON PROMPTFILE [bad]. Greedy 256-token /v1/completions requests with token ids: short prompt and 4K prompt one at a time
(single stream), then the same two plus two more concurrently (informational). With 'bad': after that POST {"prompt":[999999999]} (expect 400), then rerun the
short request (expect 200 and the same ids)."""
import json, sys, threading, urllib.request, urllib.error
port, out, pf = sys.argv[1:4]; bad = len(sys.argv) > 4
text = open(pf).read()
P = {"short": "Write a Rust function that parses a CSV line into fields, handling quoted commas, with unit tests.", "4k": text,
     "c900": "Request 3.\n" + text[:900], "c1800": "Request 4.\n" + text[:1800]}
def post(body):
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(r, timeout=900); return resp.status, json.load(resp)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]
def go(k):
    s, d = post({"model": "intarsia", "prompt": P[k], "max_tokens": 256, "temperature": 0, "ignore_eos": True, "return_token_ids": True})
    return {"status": s, "ids": d["choices"][0].get("token_ids") if s == 200 else None, "text": d["choices"][0]["text"] if s == 200 else d}
res = {"seq": {k: go(k) for k in ("short", "4k")}}
conc = {}
ts = [threading.Thread(target=lambda k=k: conc.__setitem__(k, go(k))) for k in P]
[t.start() for t in ts]; [t.join() for t in ts]; res["conc"] = conc
if bad:
    s, d = post({"prompt": [999999999]}); res["oov"] = {"status": s, "body": d if s != 200 else "200"}
    res["after"] = go("short")
json.dump(res, open(out, "w"), indent=1)
print({k: (v["status"], len(v["ids"] or [])) for k, v in res["seq"].items()}, "conc", {k: (v["status"], len(v["ids"] or [])) for k, v in conc.items()})
if bad: print("oov", res["oov"]["status"], res["oov"]["body"][:200], "| after", res["after"]["status"], "same ids as before:", res["after"]["ids"] == res["seq"]["short"]["ids"])
