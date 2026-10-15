#!/usr/bin/env python3
"""analyze.py <passes dir> <macmon.jsonl>: per pass and prompt size: decode tok/s (median of 3 reps, each rep), prefill tok/s,
GPU clock (macmon gpu_usage[0] MHz, median over the rep's decode window) and utilisation."""
import json, re, glob, os, sys, statistics as st
from datetime import datetime
P, MM = sys.argv[1], sys.argv[2]
mm = []
for line in open(MM):
    try: d = json.loads(line)
    except Exception: continue
    mm.append((datetime.fromisoformat(d["timestamp"]).timestamp(), d["gpu_usage"][0], d["gpu_usage"][1], d["gpu_power"]))
mm.sort()
med = lambda x: st.median(x) if x else float("nan")
out = {}
for f in sorted(glob.glob(P + "/*-localeval.ts")):
    label = os.path.basename(f).replace("-localeval.ts", "")
    for line in open(f):
        ts, _, body = line.partition("\t")
        m = re.search(r"\]\s+(\d+) tok rep (\d)/3: prompt=(\d+) ttft=([\d.]+)s prefill=(\d+) tok/s decode=([\d.]+) tok/s.*out=(\d+)", body)
        if not m: continue
        size, ttft, pf, dec, n = int(m[1]), float(m[4]), int(m[5]), float(m[6]), int(m[7])
        tend = float(ts); tdec = n / dec
        w = [s for s in mm if tend - tdec < s[0] <= tend]
        out.setdefault((label, size), []).append((dec, pf, med([s[1] for s in w]), med([s[2] for s in w]) * 100))
print("| pass | size | decode tok/s (3 reps) | median | prefill tok/s | GPU MHz in decode | GPU util % |")
print("|---|---|---|---|---|---|---|")
for (label, size), r in sorted(out.items(), key=lambda k: (k[0][1], k[0][0])):
    print(f"| {label} | {size} | {' / '.join(f'{x[0]:.1f}' for x in r)} | {med([x[0] for x in r]):.2f} | {med([x[1] for x in r]):.0f} | {med([x[2] for x in r]):.0f} | {med([x[3] for x in r]):.0f} |")
