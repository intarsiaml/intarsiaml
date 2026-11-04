#!/usr/bin/env python3
"""Real concurrent bursts, no fixed composition: q4 with three short prompts, thread start order shuffled per round (seeded). ids only, as ident-rt.sh."""
import json, random, sys, threading, urllib.request
port, pr, outp, rounds = sys.argv[1:5]
text = open(pr).read()
lens = [150, 400, 900, 1800, 3000]
P = {k: f"Request {k}, {lens[k]} characters of context follows.\n" + text[:lens[k]] for k in (0, 1, 2, 4)}
res = {}; rng = random.Random(20261104)
def call(k, rnd):
    b = json.dumps({"model": "intarsia", "prompt": P[k], "max_tokens": 200, "temperature": 0, "ignore_eos": True, "return_token_ids": True}).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", b, {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(r, timeout=900))
    res[f"r{rnd}-q{k}"] = {"ids": d["choices"][0]["token_ids"]}
order = {}
for rnd in range(1, int(rounds) + 1):
    o = list(P); rng.shuffle(o); order[rnd] = o
    ts = [threading.Thread(target=call, args=(k, rnd)) for k in o]
    [t.start() for t in ts]; [t.join() for t in ts]
    json.dump({"res": res, "order": order}, open(outp, "w"))
