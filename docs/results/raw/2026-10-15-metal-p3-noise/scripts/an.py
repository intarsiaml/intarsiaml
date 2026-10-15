#!/usr/bin/env python3
"""an.py RUNDIR : per-request engine prefill (trace lines) joined with macmon GPU clock/power over the prefill window."""
import sys, json, re, glob, statistics as st
from datetime import datetime
D = sys.argv[1]
mm = []
for l in open(f"{D}/macmon.jsonl"):
    try: d = json.loads(l)
    except Exception: continue
    mm.append((datetime.fromisoformat(d["timestamp"]).timestamp(), d["gpu_usage"][0], d["gpu_power"], d["gpu_usage"][1]))
mm.sort()
def win(a, b): return [s for s in mm if a < s[0] <= b + 0.1]
rows = []
for f in sorted(glob.glob(f"{D}/p*-server.err")):
    label = re.search(r"(p\d+-c\d+)-server", f)[1]
    for l in open(f):
        m = re.match(r"([\d.]+) \[prefill-trace\] wall (\d+) ms, engine-thread cpu (\d+) ms; per-chunk wall/cpu ms: (.*)", l)
        if not m: continue
        t, wall = float(m[1]), int(m[2])
        ch = [tuple(map(int, x.split("/"))) for x in m[4].split()]
        s = win(t - wall / 1000, t)
        rows.append(dict(label=label, t=t, wall=wall, chunks=[c[0] for c in ch],
                         mhz=st.mean(x[1] for x in s) if s else None, w=st.mean(x[2] for x in s) if s else None,
                         util=st.mean(x[3] for x in s) if s else None, n=len(s)))
big = [r for r in rows if r["wall"] > 700]
print("label wall chunks(ms) lastchunk mhz W util")
for r in rows:
    if r["wall"] < 700: continue
    last = r["wall"] - sum(r["chunks"])
    print(f'{r["label"]:10s} {r["wall"]:5d} {r["chunks"]} last={last:4d} {r["mhz"]:.0f}MHz {r["w"]:.0f}W util={r["util"]*100:.0f}% n={r["n"]}')
