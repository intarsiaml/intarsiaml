#!/usr/bin/env python3
"""usage: an_micro.py FILE.jsonl -> per T phase, mean exclusive GPU us per record op (moe_experts_bf16_rows, #1 = down, #2 = gather)."""
import json, sys, collections
R = sorted((json.loads(l) for l in open(sys.argv[1])), key=lambda r: r['s'])
n = len(R) // 4
ph = [R[i * n:(i + 1) * n] for i in range(4)]  # 4 equal T phases (same call count each)
TS = [8, 9, 12, 16]
if len(ph) != len(TS): print('phases', len(ph), [len(p) for p in ph])
for T, P in zip(TS, ph):
    pe = P[0]['s']; acc = collections.defaultdict(list)
    for r in P:
        d = max(0.0, r['e'] - max(r['s'], pe)); pe = max(pe, r['e'])
        acc[r['op']].append(d)
    print(f"T={T}: " + "  ".join(f"{k}: {1e6*sum(v)/len(v):.1f}us(n={len(v)})" for k, v in sorted(acc.items())))
