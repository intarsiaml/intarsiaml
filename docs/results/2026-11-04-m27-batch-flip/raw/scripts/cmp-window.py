#!/usr/bin/env python3
import json, re
O = ".scratch/logs/window/"
ref = json.load(open("docs/results/2026-11-03-m27-hs-rowtable/raw/ident/def-mb2-ids.json"))["r1-q4"]["ids"]
d = json.load(open(O + "ids.json")); ev = {}
for l in open(O + "server.err"):
    m = re.search(r"(PrefillStart|Chunk) \{ iter: (\d+), id: (\d+), (.*) \}", l)
    if not m: continue
    e = ev.setdefault(int(m[3]), {"chunks": []}); kv = dict(x.split(": ") for x in m[4].split(", "))
    if m[1] == "PrefillStart": e["C"], e["dec"] = int(kv["chunk"]), int(kv["decoding"])
    else: e["chunks"].append(int(kv["len"]))
q4 = [e for i, e in sorted(ev.items()) if sum(e["chunks"]) == 772]
for (k, v), e in zip(sorted(d.items()), q4):
    fd = next((i for i, (a, b) in enumerate(zip(v["ids"], ref)) if a != b), None)
    print(f"{k} order {v['order']} C={e['C']} decoding={e['dec']} chunks={e['chunks']} differs={v['ids'] != ref} first_diff={fd} id184={v['ids'][184]}")
