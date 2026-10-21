#!/usr/bin/env python3
"""kop.py <kernel-trace.jsonl.gz> <steps>: per-op GPU time per decode step (kernel mode, serialised; last <steps> steps)."""
import gzip, json, sys, collections
r = [json.loads(l) for l in gzip.open(sys.argv[1], "rt")]
n = int(sys.argv[2]); r = r[-364 * n:]
agg = collections.defaultdict(lambda: [0, 0.0, 0])
for x in r:
    k = (x["op"], x["n"], x["k"]); a = agg[k]; a[0] += 1; a[1] += (x["e"] - x["s"]) * 1e3; a[2] = x["bytes"]
tot = sum(a[1] for a in agg.values()) / n
print(f"{sys.argv[1]}: kernel-mode sum {tot:.2f} ms/step")
for k, a in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    ms = a[1] / n; us = a[1] / a[0] * 1e3
    gbs = f"{a[2] / (us * 1e-6) / 1e9:.0f} GB/s" if a[2] else ""
    print(f"{k[0]:28s} {k[1]:>6}x{k[2]:<6} {a[0]//n:>3}/step {us:8.1f} us {ms:7.3f} ms {100*ms/tot:5.1f}% {gbs}")
