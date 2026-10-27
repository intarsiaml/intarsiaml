#!/usr/bin/env python3
"""Lane LDF: timing runs of one window ordered base, V1..Vn, base2, Vn-r..V1-r, base3: each variant's sum of microseconds at every M divided by the
base interpolated linearly in run order between the surrounding base runs (the card's clock drifts a few percent over a window); fwd and -r averaged.
usage: drift.py <dir> <base tags comma> <variant order comma> (variant tags without -r)"""
import re, sys, os
d = sys.argv[1]; bases = sys.argv[2].split(","); vs = sys.argv[3].split(",")
order = [bases[0]] + vs + [bases[1]] + [v + "-r" for v in reversed(vs)] + [bases[2]]
shapes = ["gate+up", "down", "q+gate", "z", "o"]
def load(tag):
    r = {}
    for l in open(f"{d}/t-{tag}.txt"):
        m = re.match(r"TIME a8tp2\s+(\S+)\s+\d+x\d+\s+M=(\d+)\s+([\d.]+) us", l)
        if m: r[(m.group(1), int(m.group(2)))] = float(m.group(3))
    return r
data = {t: load(t) for t in order if os.path.exists(f"{d}/t-{t}.txt")}
Ms = sorted({m for r in data.values() for (_, m) in r})
def tot(r, m): return sum(r[(s, m)] for s in shapes)
pos = {t: i for i, t in enumerate(order)}
nb = len(order) - 1
def base_at(i, m):
    lo, mid, hi = 0, pos[bases[1]], nb
    a, b, ia, ib = (lo, mid, 0, 1) if i <= mid else (mid, hi, 1, 2)
    fa, fb = tot(data[bases[ia]], m), tot(data[bases[ib]], m)
    return fa + (fb - fa) * (i - a) / (b - a)
print("variant".ljust(14) + "".join(f"M={m}".rjust(9) for m in Ms) + "   (ratio to the interpolated base; < 1 is faster)")
for v in vs:
    row = []
    for m in Ms:
        rs = [tot(data[t], m) / base_at(pos[t], m) for t in (v, v + "-r") if t in data]
        row.append(sum(rs) / len(rs))
    print(v.ljust(14) + "".join(f"{x:9.3f}" for x in row))
