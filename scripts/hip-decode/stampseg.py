#!/usr/bin/env python3
"""stampseg.py <a stamps.txt> <b stamps.txt>: the same prefill stamped twice (e.g. split on = a, off = b). For every pair
of consecutive stamped ops, the unstamped stretch between them (end of op i -> start of op i+1: the kernels in between and
their boundaries) in both runs, summed by (op i -> op i+1); and the stamped ops themselves. Shows where b loses time."""
import sys
from collections import defaultdict


def ops(path):
    with open(path) as f:
        khz = float(f.readline().split()[2])
        st = [(int(a) / (khz / 1000.0), t, int(p)) for a, t, p in (l.split() for l in f if l.strip())]
    out = []
    for i, (ts, tag, piece) in enumerate(st):
        if piece < 0:
            out.append([tag, ts, ts])
        else:
            out[-1][2] = ts
    return out


a, b = ops(sys.argv[1]), ops(sys.argv[2])
assert [x[0] for x in a] == [x[0] for x in b], "different op sequences"
seg, inop = defaultdict(lambda: [0, 0.0, 0.0]), defaultdict(lambda: [0, 0.0, 0.0])
for i in range(len(a)):
    r = inop[a[i][0]]
    r[0] += 1; r[1] += a[i][2] - a[i][1]; r[2] += b[i][2] - b[i][1]
    if i + 1 < len(a):
        k = f"{a[i][0]} -> {a[i + 1][0]}"
        r = seg[k]
        r[0] += 1; r[1] += a[i + 1][1] - a[i][2]; r[2] += b[i + 1][1] - b[i][2]
print(f"{'stretch':<28} {'n':>4} {'a ms':>9} {'b ms':>9} {'b - a ms':>9}")
for name, d in (("inside op", inop), ("between ops", seg)):
    print(f"-- {name}")
    for k, (n, x, y) in sorted(d.items(), key=lambda kv: -(kv[1][2] - kv[1][1])):
        print(f"{k:<28} {n:>4} {x / 1e3:>9.2f} {y / 1e3:>9.2f} {(y - x) / 1e3:>+9.2f}")
