#!/usr/bin/env python3
"""an_experts.py TRACE.jsonl.gz ORDER : distinct routed experts per layer per step from the `kernel` trace labels
(MoE launch: n = distinct experts, k = valid (token, slot) pairs)."""
import sys, statistics
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import mp
f = sys.argv[1]
order = [int(x) for x in sys.argv[2].split(',')]
D = mp.by_b(f, 'kernel', order)
E_BYTES = 1.77e6   # per expert: gate-up 1024x2048 + down 2048x512 nvfp4 incl. block scales (MDEC: 8 x 1.77 MB x 40 layers = 0.57 GB at B=1)
print('| B | pairs / layer | mean distinct / layer | min | max | p10 | p90 | formula 256(1-(1-8/256)^B) | measured / formula | pairs per distinct expert | expert MB / step (measured) | expert MB / step (formula) |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|')
for b in sorted(order):
    ns = []; ks = []
    perlayer = [[] for _ in range(40)]
    for s in D[b]:
        m = [r for r in s if r['op'].startswith('moe_experts') and '#' not in r['op']]
        assert len(m) == 40, len(m)
        for i, r in enumerate(m):
            ns.append(r['n']); ks.append(r['k']); perlayer[i].append(r['n'])
    ns_s = sorted(ns)
    form = 256 * (1 - (1 - 8 / 256) ** b)
    mean = statistics.mean(ns)
    print(f'| {b} | {statistics.mean(ks):.1f} | {mean:.1f} | {ns_s[0]} | {ns_s[-1]} | {ns_s[len(ns)//10]} | {ns_s[len(ns)*9//10]} | {form:.1f} | {mean/form:.2f} | {statistics.mean(ks)/mean:.2f} | {mean*40*E_BYTES/1e6:.0f} | {form*40*E_BYTES/1e6:.0f} |')
    if b == 16:
        pl = [statistics.mean(x) for x in perlayer]
        print('\nper-layer mean distinct experts at B=16 (MoE layer 0..39):', ' '.join(f'{x:.0f}' for x in pl))
