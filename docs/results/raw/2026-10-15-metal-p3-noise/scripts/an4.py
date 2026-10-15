#!/usr/bin/env python3
"""an4.py RUNDIR [minMHz]: engine-side prefill wall per (build, chunk, tail class) from the trace lines, GPU-clock filtered."""
import sys, json, re, glob, statistics as st, collections
from datetime import datetime
D = sys.argv[1]; MIN = float(sys.argv[2]) if len(sys.argv) > 2 else 1450
mm = []
for l in open(f"{D}/macmon.jsonl"):
    try: d = json.loads(l)
    except Exception: continue
    mm.append((datetime.fromisoformat(d["timestamp"]).timestamp(), d["gpu_usage"][0], d["gpu_power"]))
mm.sort()
rows = []
for f in sorted(glob.glob(f"{D}/p*-server.err")):
    m0 = re.search(r"(p\d+)-(\w+)-c(\d+)-server", f); build, chunk = m0[2], int(m0[3])
    for l in open(f):
        m = re.match(r"([\d.]+) \[prefill-trace\] plen (\d+) pos0 (\d+) wall (\d+) ms", l)
        if not m: continue
        t, plen, pos0, wall = float(m[1]), int(m[2]), int(m[3]), int(m[4])
        n = plen - pos0
        if not (4070 <= n <= 4200): continue
        k = (n - 1) // chunk; last = n - k * chunk
        s = [x for x in mm if t - wall/1000 < x[0] <= t + 0.1]
        rows.append(dict(build=build, chunk=chunk, n=n, plen=plen, last=last, wall=wall, mhz=st.mean(x[1] for x in s), pass_=m0[1]))
json.dump(rows, open(f"{D}/rows.json", "w"))
print(f"filter: request-mean GPU clock >= {MIN} MHz")
g = collections.defaultdict(list)
for r in rows: g[(r["chunk"], r["n"] if r["chunk"] else 0, r["build"])].append(r)
print("chunk  n(last chunk)   build   n_all n_fast  median wall ms (min-max) | MHz")
keys = sorted({(c, n) for (c, n, b) in g})
for c, n in keys:
    for b in ("cand", "fix"):
        rs = g.get((c, n, b), [])
        if not rs: continue
        good = [r for r in rs if r["mhz"] >= MIN]; w = [r["wall"] for r in good]
        last = rs[0]["last"]
        print(f"{c:5d}  {n} ({last:4d})  {b:5s}  {len(rs):3d} {len(good):3d}   " + (f"{st.median(w):.0f} ({min(w)}-{max(w)}) | {st.mean(r['mhz'] for r in good):.0f}" if w else "-"))
