#!/usr/bin/env python3
"""period.py <cb-trace.jsonl> <dispatches-per-step> <steps>: step period, GPU busy and idle (ms/step) from a cb trace."""
import json, sys

f, d, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
c = [json.loads(l) for l in open(f)]
acc, j = 0, len(c)
while acc < d * n and j > 0:
    j -= 1
    acc += c[j]["nd"]
s = c[j:]
span = s[-1]["e"] - s[0]["s"]
busy = sum(r["e"] - r["s"] for r in s)
print(f"{span / n * 1e3:.3f} {busy / n * 1e3:.3f} {(span - busy) / n * 1e3:.3f} {acc / n:.1f}")
