#!/usr/bin/env python3
"""an2.py RUNDIR [minMHz] : engine prefill wall per prompt length and chunk config, clock-filtered."""
import sys, json, re, glob, statistics as st, collections
from datetime import datetime
D = sys.argv[1]; MIN = float(sys.argv[2]) if len(sys.argv) > 2 else 1500
mm = []
for l in open(f"{D}/macmon.jsonl"):
    try: d = json.loads(l)
    except Exception: continue
    mm.append((datetime.fromisoformat(d["timestamp"]).timestamp(), d["gpu_usage"][0], d["gpu_power"]))
mm.sort()
rows = []
for f in sorted(glob.glob(f"{D}/p*-server.err")):
    label = re.search(r"(p\d+-c\d+)-server", f)[1]; chunk = int(label.split("-c")[1])
    for l in open(f):
        m = re.match(r"([\d.]+) \[prefill-trace\] plen (\d+) pos0 (\d+) wall (\d+) ms, engine-thread cpu (\d+) ms; per-chunk wall/cpu ms: (.*)", l)
        if not m: continue
        t, plen, wall = float(m[1]), int(m[2]), int(m[4])
        if wall < 700: continue
        s = [x for x in mm if t - wall/1000 < x[0] <= t + 0.1]
        mhz = st.mean(x[1] for x in s) if s else 0
        rows.append((chunk, plen, wall, mhz, label, [int(x.split('/')[0]) for x in m[6].split()]))
print(f"clock filter >= {MIN} MHz mean over the request")
by = collections.defaultdict(list)
for r in rows: by[(r[0], r[1])].append(r)
for (c, p) in sorted(by):
    rs = by[(c, p)]; good = [r for r in rs if r[3] >= MIN]
    w = [r[2] for r in good]
    print(f"chunk {c:5d} plen {p}  n={len(rs)} fast={len(good)}  wall ms median {st.median(w) if w else float('nan'):.0f} (min {min(w) if w else 0} max {max(w) if w else 0})  MHz {st.mean(r[3] for r in good) if good else 0:.0f}")
