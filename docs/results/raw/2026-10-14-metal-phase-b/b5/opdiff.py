#!/usr/bin/env python3
"""opdiff.py <kernel-trace-A> <kernel-trace-B> [steps]: per-op median us/call and us/step (last N decode steps), A vs B.

A step starts at an embedding record with t == 1. Ops are keyed by (op, n, k) so GEMVs of different shapes stay apart."""
import json, sys, collections

N = int(sys.argv[3]) if len(sys.argv) > 3 else 64


def steps(path):
    recs = [json.loads(l) for l in open(path)]
    idx = [i for i, r in enumerate(recs) if r["op"] == "embedding_bf16" and r["t"] == 1]
    segs = [recs[a:b] for a, b in zip(idx, idx[1:] + [len(recs)])]
    return segs[-N:]


def table(path):
    segs = steps(path)
    d = collections.defaultdict(list)
    per_step = collections.defaultdict(float)
    for s in segs:
        for r in s:
            k = (r["op"], r["n"], r["k"])
            us = (r["e"] - r["s"]) * 1e6
            d[k].append(us)
            per_step[k] += us
    out = {}
    for k, v in d.items():
        v.sort()
        out[k] = (v[len(v) // 2], len(v) / len(segs), per_step[k] / len(segs))
    return out, sum(per_step.values()) / len(segs)


a, ta = table(sys.argv[1])
b, tb = table(sys.argv[2])
print(f"total us/step (sum of per-launch kernel-mode durations): A {ta:.0f}  B {tb:.0f}  diff {tb - ta:+.0f}")
print(f"{'op':<34}{'n':>7}{'k':>7}{'calls':>7}{'A us':>9}{'B us':>9}{'A ms/st':>9}{'B ms/st':>9}{'diff ms':>9}")
keys = sorted(set(a) | set(b), key=lambda k: -(abs(b.get(k, (0, 0, 0))[2] - a.get(k, (0, 0, 0))[2])))
for k in keys:
    x = a.get(k, (0, 0, 0))
    y = b.get(k, (0, 0, 0))
    print(f"{k[0]:<34}{k[1]:>7}{k[2]:>7}{y[1]:>7.0f}{x[0]:>9.1f}{y[0]:>9.1f}{x[2] / 1e3:>9.3f}{y[2] / 1e3:>9.3f}{(y[2] - x[2]) / 1e3:>+9.3f}")
