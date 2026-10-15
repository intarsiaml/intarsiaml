#!/usr/bin/env python3
"""Summarise sweep/*.err: plain prefill ms and peak memory footprint per (prompt, chunk): median (min-max) over reps."""
import glob, re, statistics, sys, os

d = sys.argv[1]
rows = {}
for f in sorted(glob.glob(os.path.join(d, "*-c*-r*.err"))):
    m = re.match(r"(\w+)-c(\d+)-r(\d+)\.err", os.path.basename(f))
    if not m:
        continue
    p, c, r = m.group(1), int(m.group(2)), int(m.group(3))
    t = open(f).read()
    pm = re.findall(r"prefill (\d+) ms \(([\d.]+) tok/s\)", t)
    ntok = re.findall(r"prompt: (\d+) tokens", t)
    fp = re.search(r"(\d+)\s+peak memory footprint", t)
    if not pm:
        continue
    rows.setdefault((p, c), []).append((int(pm[-1][0]), int(ntok[-1]), int(fp.group(1)) if fp else 0, r))

for p in ["1k", "4k", "16k"]:
    base = None
    print(f"## {p}")
    print("| chunk | n | prefill ms median (min-max) | tok/s median | vs 256 | peak footprint MiB median (min-max) |")
    print("|---:|---:|---|---:|---:|---|")
    for c in [256, 512, 1024, 2048]:
        v = rows.get((p, c))
        if not v:
            continue
        ms = [x[0] for x in v]
        n = v[0][1]
        med = statistics.median(ms)
        if base is None:
            base = med
        fps = [x[2] / 2**20 for x in v]
        print(f"| {c} | {len(v)} | {med:.0f} ({min(ms)}-{max(ms)}) | {n / med * 1000:.0f} | {base / med:.2f}x | {statistics.median(fps):.0f} ({min(fps):.0f}-{max(fps):.0f}) |")
    print()
