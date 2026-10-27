#!/usr/bin/env python3
"""Lane GLR: tables from the TIME a8tp2 lines of timing runs: per variant the microseconds per shape at one M and the sum over the five
shapes at every M. usage: ttab.py <dir> <M> [variant ...]"""
import re, sys, glob, os
d, M = sys.argv[1], int(sys.argv[2])
vs = sys.argv[3:] or sorted(os.path.basename(f)[2:-4] for f in glob.glob(f"{d}/t-*.txt"))
data = {}
for v in vs:
    f = f"{d}/t-{v}.txt"
    if not os.path.exists(f): continue
    rows = {}
    for l in open(f):
        m = re.match(r"TIME a8tp2\s+(\S+)\s+(\d+x\d+)\s+M=(\d+)\s+([\d.]+) us", l)
        if m: rows[(m.group(1), int(m.group(3)))] = float(m.group(4))
    data[v] = rows
shapes = ["gate+up", "down", "q+gate", "z", "o"]
print(f"M={M} us per shape".ljust(16) + "".join(s.rjust(10) for s in shapes) + "  sum".rjust(10))
for v, r in data.items():
    vals = [r.get((s, M)) for s in shapes]
    tot = sum(x for x in vals if x) if all(vals) else float("nan")
    print(v.ljust(16) + "".join((f"{x:10.1f}" if x else "         -") for x in vals) + f"{tot:10.1f}")
Ms = sorted({m for r in data.values() for (_, m) in r})
print("sum over shapes by M (us)".ljust(26) + "".join(str(m).rjust(9) for m in Ms))
for v, r in data.items():
    print(v.ljust(26) + "".join((f"{sum(r.get((s, m), float('nan')) for s in shapes):9.0f}") for m in Ms))
