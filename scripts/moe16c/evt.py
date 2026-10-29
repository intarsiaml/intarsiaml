#!/usr/bin/env python3
"""Lane moe16c: base vs dev from micro.sh output (EVT lines after `== base` / `== dev`), mean us per (kind, shape, form, M)."""
import re, sys, collections
forms = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else {"unset", "split=unset"}
cur, d = None, collections.defaultdict(list)
for line in open(sys.argv[1]):
    if line.startswith("== "):
        cur = line.split()[1]
    m = re.match(r"EVT (\w+) (\d+x\d+) +(\S+) +M=(\d+) +([\d.]+) us", line)
    if m and cur:
        kind, shape, form, mm, us = m.groups()
        if form == "split=": continue
        d[(kind, shape, form, int(mm), cur)].append(float(us))
keys = sorted({k[:4] for k in d}, key=lambda k: (k[0], k[1], k[2], k[3]))
print("| kernel | shape | form | M | base us | dev us | dev vs base |\n|---|---|---|---:|---:|---:|---:|")
for k in keys:
    if k[2] not in forms: continue
    b, v = d.get(k + ("base",)), d.get(k + ("dev",))
    if not b or not v: continue
    bm, vm = sum(b) / len(b), sum(v) / len(v)
    print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | {bm:.2f} | {vm:.2f} | {100 * (vm / bm - 1):+.1f} % |")
