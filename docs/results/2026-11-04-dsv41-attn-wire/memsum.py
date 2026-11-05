#!/usr/bin/env python3
"""Lowest MemAvailable (GiB) per rank and config from a run-full.sh window: memsum.py WINDOW_DIR RUN_LOG  (RUN_LOG: the job's stdout, which has every "config" line; WINDOW_DIR:  spark1-memfine-*.log, spark2-memfine-*.log).
A config runs from its "config abN:" line to "config abN done"; 'after load' starts at the "serve up" line of that config (the weight load, which has
the lowest values, comes before it). Times in steps.log are local (endor), the memfine lines are unix seconds."""
import sys, glob, re, datetime as dt
d = sys.argv[1]
mf = {h: [(int(l.split()[0]), float(re.search(r"MemAvailable=([\d.]+)", l).group(1))) for l in open(glob.glob(f"{d}/{h}-memfine-*.log")[0]) if "MemAvailable=" in l] for h in ("spark1", "spark2")}
day = dt.datetime.fromtimestamp(mf["spark1"][0][0]).date()
ts = lambda s: int(dt.datetime.combine(day, dt.time.fromisoformat(s)).timestamp())
cfg = {}; cur = None
for l in open(sys.argv[2]):
    m = re.match(r"(\d\d:\d\d:\d\d) config (ab\d+):", l)
    if m: cur = m[2]; cfg[cur] = {"start": ts(m[1])}; continue
    m = re.match(r"(\d\d:\d\d:\d\d) serve up=1", l)
    if m and cur: cfg[cur]["up"] = ts(m[1])
    m = re.match(r"(\d\d:\d\d:\d\d) config (ab\d+) done", l)
    if m: cfg[m[2]]["end"] = ts(m[1])
print("config   spark1 min (after load)   spark2 min (after load)   [GiB; kill line 5]")
for c, t in cfg.items():
    out = []
    for h in ("spark1", "spark2"):
        allv = [v for s, v in mf[h] if t["start"] <= s <= t["end"] + 1]
        up = [v for s, v in mf[h] if t["up"] <= s <= t["end"] + 1]
        out.append(f"{min(allv):6.2f} ({min(up):.2f})")
    print(f"{c:6}   {out[0]:>22}   {out[1]:>22}")
