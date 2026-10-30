#!/usr/bin/env python3
"""an_served_experts.py TRACE : distinct experts per MoE layer per decode step by rows in the served `kernel` trace (labels n = distinct,
k = pairs), for decode steps with t rows (t <= 16)."""
import sys, collections, statistics
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import mp
R = mp.load(sys.argv[1])
S = []
for r in R:
    if r['op'] == 'embedding_bf16': S.append([])
    if S: S[-1].append(r)
by = collections.defaultdict(list)
for s in S:
    t = s[0]['t']
    if t > 16: continue
    m = [x for x in s if x['op'].startswith('moe_experts') and '#' not in x['op']]
    if len(m) != 40: continue
    by[t].append([x['n'] for x in m])
print('| rows in step | steps | pairs / layer | mean distinct / layer | min | max | formula | measured / formula | expert MB / step |')
print('|---|---|---|---|---|---|---|---|---|')
for t in sorted(by):
    if len(by[t]) < 20: continue
    ns = [n for st in by[t] for n in st]
    form = 256 * (1 - (1 - 8 / 256) ** t)
    print(f'| {t} | {len(by[t])} | {8*t} | {statistics.mean(ns):.1f} | {min(ns)} | {max(ns)} | {form:.1f} | {statistics.mean(ns)/form:.2f} | {statistics.mean(ns)*40*1.77:.0f} |')
