#!/usr/bin/env python3
"""absum.py <ab.log>...: per variant step period (ms, mean of the pairs; first number after the dispatch-count-free idle field)
and the delta against `base`/`off`, from exp.sh output lines."""
import re, sys, collections, statistics
for f in sys.argv[1:]:
    per = collections.defaultdict(list)
    for l in open(f):
        m = re.match(r"(\S+) (short|4k) pair(\d+) idle_util=\S+ ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)", l)
        if m:
            per[m.group(1)].append((int(m.group(3)), float(m.group(4)), float(m.group(5))))
    print(f"== {f}")
    ref = None
    for k in ("base", "off"):
        if k in per:
            ref = k
            break
    base = {p: v for p, v, _ in per[ref]} if ref else {}
    for k, v in per.items():
        mean = statistics.mean(x for _, x, _ in v)
        d = [x - base[p] for p, x, _ in v if p in base]
        print(f"{k:6s} n={len(v)} period mean {mean:7.3f} ms  vs {ref}: mean {statistics.mean(d):+.3f} (min {min(d):+.3f} max {max(d):+.3f}) tok/s {1000 / mean:.2f}")
