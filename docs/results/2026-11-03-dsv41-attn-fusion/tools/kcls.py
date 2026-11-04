#!/usr/bin/env python3
"""per-kernel excl time for classes, median over requests of prompt index PI (alternating prompts)."""
import sys, re, statistics as st
from collections import defaultdict
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prefill_table import load, split, excl_times, short
from class_table import OURS, TF, classify
path, eng, pi, np_ = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
want = sys.argv[5].split(",") if len(sys.argv) > 5 else ["attention", "hyper-connection"]
rules = OURS if eng.startswith("ours") else TF
reqs = [q for q in split(load(path), 1_000_000_000) if (max(r[0]+r[1] for r in q)-q[0][0]) > 50e6]
reqs = [q for i, q in enumerate(reqs) if i % np_ == pi]
per = []
for q in reqs:
    ex = excl_times(q); d = defaultdict(lambda: [0, 0.0, 0.0])
    for i, r in enumerate(q):
        c = classify(rules, r[2])
        if any(c.startswith(w) for w in want):
            k = (c.split()[0], short(r[2])); d[k][0] += 1; d[k][1] += r[1]/1e6; d[k][2] += ex.get(i, 0)/1e6
    per.append(d)
keys = {k for d in per for k in d}
rows = [(k, st.median(d.get(k, [0,0,0])[0] for d in per), st.median(d.get(k, [0,0,0])[1] for d in per), st.median(d.get(k, [0,0,0])[2] for d in per)) for k in keys]
tot = 0
for k, n, s, e in sorted(rows, key=lambda x: -x[3]):
    tot += e
    print(f"{k[0]:<18}{k[1]:<80}{int(n):>6} calls  sum {s:8.1f} ms  excl {e:8.1f} ms")
print("total excl", round(tot, 1), "nreq", len(reqs))
