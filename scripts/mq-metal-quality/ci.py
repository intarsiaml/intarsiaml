#!/usr/bin/env python3
"""Scores table with Wilson 95% CIs from localeval result.json files: ci.py raw/*-result.json"""
import json, sys, math
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h
for f in sorted(sys.argv[1:]):
    d = json.load(open(f))
    for b in d["benchmarks"]:
        n = b["n_predictions"]; s = b["overall_score"]; k = round(s * n); lo, hi = wilson(k, n)
        extra = {kk: round(v, 3) for kk, v in (b.get("scores") or {}).items()}
        print(f"{f.split('/')[-1].split('-result')[0]:11s} {b['benchmark']:6s} n={n:5d} k={k:4d} score={s:.3f} CI95=[{lo:.3f},{hi:.3f}] {extra}")
