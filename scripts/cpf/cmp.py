#!/usr/bin/env python3
"""Lane pf: one ids run against the solo spec-off reference: per sequence identical or the first differing token, the server's
pack counters, and the Pack trace events (which requests' bodies, of how many rows, ran in one packed pass).
cmp.py <solo.json> <run.json> <run.metrics> <trace.txt>"""
import json, re, sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
name = sys.argv[2].split("/")[-1]
res = []
for x, y in zip(a, b):
    res.append("=" if x == y else f"@{next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), min(len(x), len(y)))}")
met = dict(l.split()[:2] for l in open(sys.argv[3]) if l.strip())
packs = [l.strip() for l in open(sys.argv[4]) if "Pack" in l]
ok = all(r == "=" for r in res) and len(a) == len(b)
print(f"{name}: {'ALL' if ok else 'MISMATCH'} per sequence {res} pack_passes {met.get('intarsia_batch_pack_passes_total', '?')} "
      f"packed_chunks {met.get('intarsia_batch_packed_chunks_total', '?')}")
for p in packs:
    print(f"  {name} {p}")
