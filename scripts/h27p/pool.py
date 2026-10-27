#!/usr/bin/env python3
"""Pool the localeval speed sweeps of lane H27P (labels h27p-<W>-<arm>-r<n>) from /opt/code/deepeval/results: per arm and prompt size the
median of the per-round medians (prefill tok/s, decode tok/s), the rounds listed, and ratios against the base arm.
usage: pool.py [W ...]   (default w5 w7)"""
import glob
import re
import statistics as st
import sys

ws = sys.argv[1:] or ["w5", "w7"]
rows = {}
for w in ws:
    for d in sorted(glob.glob(f"/opt/code/deepeval/results/*h27p-{w}-*")):
        m = re.search(rf"h27p-{w}-(\w+)-r(\d)$", d)
        if not m:
            continue
        arm, r = m.group(1), int(m.group(2))
        for l in open(d + "/summary.md"):
            if not re.match(r"\| \d+ \|", l):
                continue
            c = [x.strip() for x in l.strip().strip("|").split("|")]
            n = int(c[0])
            size = 1024 if n < 2000 else 4096 if n < 8000 else 16384
            rows.setdefault((arm, size), []).append((w, r, float(c[2].split()[0]), float(c[4].split()[0])))
arms = sorted({a for a, _ in rows})
print("arm size n | prefill tok/s median [rounds] | decode tok/s median [rounds]")
for size in (1024, 4096, 16384):
    for arm in arms:
        v = rows.get((arm, size))
        if not v:
            continue
        print(f"{arm:5} {size:5} {len(v)} | {st.median(x[2] for x in v):7.0f} {[int(x[2]) for x in v]} | {st.median(x[3] for x in v):6.1f} {[x[3] for x in v]}")
    b = rows.get(("base", size))
    for arm in arms:
        v = rows.get((arm, size))
        if arm != "base" and v and b:
            print(f"  {arm}/base {size}: prefill x{st.median(x[2] for x in v) / st.median(x[2] for x in b):.3f}  decode x{st.median(x[3] for x in v) / st.median(x[3] for x in b):.3f}")
