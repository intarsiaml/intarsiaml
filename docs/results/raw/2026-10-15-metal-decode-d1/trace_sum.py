#!/usr/bin/env python3
"""trace_sum.py <kernel-trace.jsonl>: ops per decode step (t == 1 records after the prefill), counts and mean kernel-mode us."""
import json, sys, collections
recs = [json.loads(l) for l in open(sys.argv[1])]
dec = [r for r in recs if r["t"] == 1 or r["op"] in ("argmax_f32", "argmax_f32_v4")]
# a step ends at the argmax record
steps, cur = [], []
for r in recs:
    cur.append(r)
    if r["op"].startswith("argmax"):
        steps.append(cur); cur = []
steps = [s for s in steps if len(s) > 100]
n = len(steps)
print(f"{n} decode steps; dispatches per step: {sorted(set(len(s) for s in steps))}")
s = steps[n // 2]
seq = [r["op"] for r in s]
print("one step, first 2 layers + tail:")
print(" ", seq[:24])
print(" ", seq[-6:])
cnt = collections.Counter(); tm = collections.defaultdict(float)
for st in steps[1:]:
    for r in st:
        cnt[r["op"]] += 1; tm[r["op"]] += (r["e"] - r["s"]) * 1e6
print(f"{'op':44s} {'calls/step':>10s} {'mean us':>9s} {'ms/step':>8s}")
for op, c in sorted(cnt.items(), key=lambda kv: -tm[kv[0]]):
    k = n - 1
    print(f"{op:44s} {c / k:10.1f} {tm[op] / c:9.1f} {tm[op] / k / 1000:8.3f}")
