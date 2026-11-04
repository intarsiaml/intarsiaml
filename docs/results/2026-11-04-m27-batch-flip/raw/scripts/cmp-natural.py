#!/usr/bin/env python3
"""Per request: outcome vs reference, and the schedule the trace recorded. Ids follow submit order, so requests are matched by prompt length; rounds by order of appearance."""
import json, re, sys
tag = sys.argv[1]; O = ".scratch/logs/natural/"
ref = json.load(open("docs/results/2026-11-03-m27-hs-rowtable/raw/ident/def-mb2-ids.json"))
d = json.load(open(O + tag + "-ids.json"))
ev = {}
for l in open(O + tag + "-server.err"):
    m = re.search(r"(Admitted|PrefillStart|Chunk) \{ iter: (\d+), id: (\d+), (.*) \}", l)
    if not m: continue
    e = ev.setdefault(int(m[3]), {"chunks": []})
    kv = dict(x.split(": ") for x in m[4].split(", "))
    if m[1] == "Admitted": e["cached"] = int(kv["cached"]); e["slot"] = int(kv["slot"])
    elif m[1] == "PrefillStart": e["C"] = int(kv["chunk"]); e["dec"] = int(kv["decoding"])
    else: e["chunks"].append(int(kv["len"]))
for i, e in ev.items(): e["plen"] = sum(e["chunks"])
plens = sorted({e["plen"] for e in ev.values()})
print("plens", plens)
q4 = [(i, e) for i, e in sorted(ev.items()) if e["plen"] == 772]
for rnd, (i, e) in enumerate(q4, 1):
    v = d[f"r{rnd}-q4"]["ids"]; r = ref["r1-q4"]["ids"]
    diff = next((k for k, (a, b) in enumerate(zip(v, r)) if a != b), None)
    print(f"r{rnd} q4 id={i} C={e['C']} decoding={e['dec']} cached={e.get('cached')} chunks={e['chunks']} ids==ref {v == r} first_diff={diff} id184={v[184]}")
bad = [k for k, v in d.items() if v["ids"] != ref["r1-" + k.split("-")[1]]["ids"]]
print("requests differing from ref:", bad, "of", len(d))
from collections import Counter
print("C=2048 per plen:", Counter(e["plen"] for e in ev.values() if e.get("C") == 2048))
print("cached>0:", sum(1 for e in ev.values() if e.get("cached")))
