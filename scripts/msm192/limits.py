#!/usr/bin/env python3
"""Lane msm192: per-shape naxd limit from naxd_bench output (ROUNDS lines). Per (shape, M): the ratio scratch_round / form0_round
for every round (the rounds rotate the arm order, so a round's arms ran back to back); limit = the largest M of the sweep such that
every M' <= M in the sweep (from 128 up) has min-over-rounds ratio >= THRESH, capped at CAP.
usage: limits.py [--thresh 1.03] [--cap 192] bench.txt...   (several files: their rounds are pooled, so the rule holds in every round of every file)"""
import argparse, re, collections
ap = argparse.ArgumentParser(); ap.add_argument("--thresh", type=float, default=1.03); ap.add_argument("--cap", type=int, default=192); ap.add_argument("files", nargs="+")
a = ap.parse_args(); thresh, cap = a.thresh, a.cap
d = collections.defaultdict(dict)
hashdiff = 0
for f in a.files:
  for l in open(f, errors="replace"):
    if "HASH DIFFERS" in l: hashdiff += 1
    m = re.match(r"ROUNDS n=(\d+) k=(\d+) m=(\d+) arm=(.+?) us ([\d. ]+)$", l.strip())
    if m:
        n, k, mm = int(m[1]), int(m[2]), int(m[3])
        d[(n, k, mm)].setdefault(m[4].strip(), []).extend(float(x) for x in m[5].split())
shapes = sorted({(n, k) for n, k, _ in d}, key=lambda s: (s[0] * s[1]), reverse=True)
print(f"hash differs lines: {hashdiff}")
res = {}
for (n, k) in sorted({(n, k) for n, k, _ in d}):
    ms = sorted(m for (nn, kk, m) in d if (nn, kk) == (n, k))
    row = []
    lim = 0; ok = True
    for m in ms:
        arms = d[(n, k, m)]
        sc = arms["scratch (3 dispatches)"]; fm = next(v for a, v in arms.items() if "t64x64k64w4" in a)
        r = [s / x for s, x in zip(sc, fm)]
        med = sorted(r)[len(r) // 2]
        row.append((m, min(r), med, max(r)))
        if m >= 128:
            ok = ok and min(r) >= thresh
            if ok and m <= cap: lim = m
    res[(n, k)] = lim
    print(f"{n}x{k}: limit {max(lim,128) if lim else 128}  " + "  ".join(f"M{m}: {lo:.2f}/{md:.2f}/{hi:.2f}" for m, lo, md, hi in row))
