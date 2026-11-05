#!/usr/bin/env python3
"""usage: pick-sizes.py SCOUT_DIR -> prints space separated sizes for the speed arms. Cells: E1 (p mod 4096 in 2050..2057, p < 4096), E2 (same, p > 4096), C (peeled, not exposed, ~1030),
N controls (2100 and 4096 as sizes, never peeled). A size qualifies when every scout rep lands in the cell's window."""
import glob, re, sys
P = {}
for f in glob.glob(sys.argv[1] + "/scout-*-localeval.*"):
    for l in open(f):
        m = re.search(r"\] +(\d+) tok rep (\d)/\d: prompt=(\d+)", l)
        if m: P.setdefault(int(m[1]), set()).add(int(m[3]))
def ok(ps, lo, hi): return all(lo <= p <= hi for p in ps)
pick = []
def take(cond, k):
    c = [s for s, ps in sorted(P.items()) if cond(s, ps)]
    mid = sorted(c, key=lambda s: abs(sum(P[s]) / len(P[s]) - (cond.mid if hasattr(cond, "mid") else 0)))
    return mid[:k]
E1 = [s for s, ps in sorted(P.items()) if ok(ps, 2050, 2057)]
E2 = [s for s, ps in sorted(P.items()) if ok(ps, 6146, 6153)]
C = [s for s, ps in sorted(P.items()) if all(1026 <= p <= 1033 for p in ps) or all(1282 <= p <= 1289 for p in ps)]
print("E1", E1, "E2", E2, "C", C, file=__import__("sys").stderr)
print(" ".join(str(s) for s in E1[:2] + E2[:2] + C[:1] + [2100, 4096]))
