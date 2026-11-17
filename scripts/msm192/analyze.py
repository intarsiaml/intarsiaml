#!/usr/bin/env python3
"""Lane msm192: TTFT ABBA table. usage: analyze.py <outdir>...  (several dirs pool their visits; A = main, B = this branch; per cell: mean of the arm's visits' TTFT p50 / mean, spread, B / A)."""
import collections, glob, json, os, statistics, sys
c = collections.defaultdict(lambda: collections.defaultdict(list))
for d in sorted(d for O in sys.argv[1:] for d in glob.glob(f"{O}/[AB]*")):
    arm = os.path.basename(d)[0]
    for f in glob.glob(f"{d}/*.json"):
        t = os.path.basename(f)[:-5]
        if t != "warm": c[t][arm].append((json.load(open(f))["summary"], os.path.basename(d)))
mean = lambda v: sum(v) / len(v)
spread = lambda v: (max(v) - min(v)) / mean(v) * 100
print("| cell | tokens | A TTFT p50 ms | B TTFT p50 ms | B / A (p50) | B / A (mean) | spread A / B (%) | per visit p50 ms (A | B) |")
print("|---|---:|---:|---:|---:|---:|---|---|")
for t in sorted(c, key=lambda t: (t == "null4k", t)):
    A, B = c[t]["A"], c[t]["B"]
    if not A or not B: continue
    g = lambda X, k: [x[0][k] * 1e3 for x in X]
    pa, pb = g(A, "ttft_p50"), g(B, "ttft_p50")
    print(f"| {t} | {A[0][0]['prompt_tokens_p50']:.0f} | {mean(pa):.1f} | {mean(pb):.1f} | {mean(pb)/mean(pa):.3f} | {mean(g(B,'ttft_mean'))/mean(g(A,'ttft_mean')):.3f} | "
          f"{spread(pa):.1f} / {spread(pb):.1f} | {' '.join(f'{x:.1f}' for x in pa)} | {' '.join(f'{x:.1f}' for x in pb)} |")
