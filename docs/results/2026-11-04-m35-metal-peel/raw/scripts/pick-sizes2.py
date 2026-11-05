#!/usr/bin/env python3
"""usage: pick-sizes2.py SCOUT_DIR -> sizes whose scout reps land in the exposed window 2050..2057 (E1) or the collateral window 1026..1033 (C); falls back to the best single-rep hits."""
import glob, re, sys
P = {}
for f in glob.glob(sys.argv[1] + "/scout2-*-localeval.*"):
    for l in open(f):
        m = re.search(r"\] +(\d+) tok rep (\d)/\d: prompt=(\d+)", l)
        if m: P.setdefault(int(m[1]), []).append(int(m[3]))
def pick(lo, hi, k):
    both = [s for s, ps in sorted(P.items()) if all(lo <= p <= hi for p in ps)]
    one = sorted((s for s, ps in P.items() if any(lo <= p <= hi for p in ps) and s not in both), key=lambda s: -sum(lo <= p <= hi for p in P[s]))
    return (both + one)[:k]
E1, C = pick(2050, 2057, 2), pick(1026, 1033, 2)
print("E1", E1, [P[s] for s in E1], "C", C, [P[s] for s in C], file=sys.stderr)
print(" ".join(str(s) for s in E1 + C))
