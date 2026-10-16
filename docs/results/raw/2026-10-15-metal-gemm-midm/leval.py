#!/usr/bin/env python3
"""localeval speed ABBA summary: per prompt size the TTFT of each arm (all reps of both servers), median and [min..max].
python3 leval.py <chunk>"""
import re, glob, statistics, collections, sys
chunk = int(sys.argv[1])
d = collections.defaultdict(lambda: collections.defaultdict(list))
for f in sorted(glob.glob(f"localeval-c{chunk}/gm-c{chunk}-*-localeval.out")):
    arm = re.search(r"-(base|new)-(\d)", f).group(1)
    for l in open(f):
        m = re.search(r"(\d+) tok rep \d/\d: prompt=(\d+) ttft=([\d.]+)s prefill=(\d+) tok/s", l)
        if m: d[int(m.group(1))][arm].append((float(m.group(3)), int(m.group(4)), int(m.group(2))))
print("size (prompt tok) | base TTFT s median [min..max] n | new TTFT s | new vs base | prefill tok/s base / new (median)")
for s, v in sorted(d.items()):
    b = [x[0] for x in v["base"]]; n = [x[0] for x in v["new"]]
    pb = statistics.median(x[1] for x in v["base"]); pn = statistics.median(x[1] for x in v["new"])
    pt = statistics.median(x[2] for x in v["base"] + v["new"])
    print(f"{s} ({pt:.0f}) | {statistics.median(b):.3f} [{min(b):.3f}..{max(b):.3f}] n={len(b)} | {statistics.median(n):.3f} [{min(n):.3f}..{max(n):.3f}] n={len(n)} | {100*(statistics.median(n)/statistics.median(b)-1):+.1f}% | {pb:.0f} / {pn:.0f}")
print("\nper server (TTFT s median over reps), servers in launch order:")
srv = collections.defaultdict(lambda: collections.defaultdict(list))
for f in sorted(glob.glob(f"localeval-c{chunk}/gm-c{chunk}-*-localeval.out")):
    name = re.search(r"gm-c\d+-(\w+-\d)", f).group(1)
    for l in open(f):
        m = re.search(r"(\d+) tok rep \d/\d: prompt=(\d+) ttft=([\d.]+)s", l)
        if m: srv[name][int(m.group(1))].append(float(m.group(3)))
import os
order = sorted(srv, key=lambda n: os.path.getmtime(glob.glob(f"localeval-c{chunk}/gm-c{chunk}-{n}-localeval.out")[0]))
sizes = sorted({s for n in srv for s in srv[n]})
print("server | " + " | ".join(str(s) for s in sizes))
for n in order:
    print(f"{n} | " + " | ".join(f"{statistics.median(srv[n][s]):.3f}" for s in sizes))
