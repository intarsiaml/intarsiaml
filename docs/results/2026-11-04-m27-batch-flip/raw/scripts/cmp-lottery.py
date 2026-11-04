#!/usr/bin/env python3
import json, re
from collections import Counter
O = ".scratch/logs/lottery/"
ref = json.load(open("docs/results/2026-11-03-m27-hs-rowtable/raw/ident/def-mb2-ids.json"))["r1-q4"]["ids"]
D = json.load(open(O + "ids.json")); res, order = D["res"], D["order"]
ev = {}
for l in open(O + "server.err"):
    m = re.search(r"(Admitted|PrefillStart|Chunk) \{ iter: (\d+), id: (\d+), (.*) \}", l)
    if not m: continue
    e = ev.setdefault(int(m[3]), {"chunks": []}); kv = dict(x.split(": ") for x in m[4].split(", "))
    if m[1] == "Admitted": e["cached"] = int(kv["cached"])
    elif m[1] == "PrefillStart": e["C"], e["dec"] = int(kv["chunk"]), int(kv["decoding"])
    else: e["chunks"].append(int(kv["len"]))
q4 = [e for i, e in sorted(ev.items()) if sum(e["chunks"]) == 772]
tab = Counter()
for rnd, e in enumerate(q4, 1):
    v = res.get(f"r{rnd}-q4", {}).get("ids")
    if not v: continue
    flip = v != ref; d = next((k for k, (a, b) in enumerate(zip(v, ref)) if a != b), None)
    tab[(e["C"], e["chunks"].__str__(), flip)] += 1
    print(f"r{rnd:2d} start-order {order[str(rnd)]} C={e['C']} decoding={e['dec']} cached={e.get('cached')} chunks={e['chunks']} differs={flip} first_diff={d} id184={v[184]}")
print("table (C, chunks, differs from ref): count")
for k, n in sorted(tab.items()): print(k, n)
others = sum(1 for k, v in res.items() if not k.endswith("q4") and v["ids"] != res["r1-" + k.split("-")[1]]["ids"])
print("q0/q1/q2 requests differing from their own r1:", others, "of", sum(1 for k in res if not k.endswith("q4")))
