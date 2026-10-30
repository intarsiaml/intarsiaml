#!/usr/bin/env python3
"""an_cb.py TRACE.jsonl.gz ORDER : normal-stream (`cb` profile) host / GPU timeline per decode step: command buffers per step,
dispatches, GPU busy, GPU idle gaps inside the step, host encode time (ho -> hc), commit time (hc -> hr), gap before the step."""
import sys, statistics
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import mp
f = sys.argv[1]
order = [int(x) for x in sys.argv[2].split(',')]
R = mp.load(f)
ph = mp.phases(R)
assert len(ph) == 1 + len(order), [len(p) for p in ph]
print('| B | cmd buffers / step | dispatches / step | step wall (first cb host-open -> last cb GPU end) ms | GPU busy ms | GPU idle inside step ms | host encode (sum ho->hc) ms | commit calls (sum hc->hr) ms | idle between steps (last cb end -> next first cb start) ms | first-cb GPU start minus its host-open ms |')
print('|---|---|---|---|---|---|---|---|---|---|')
for b, P in zip(order, ph[1:]):
    S = []
    for r in P:
        if r['op'] == 'embedding_bf16':
            S.append([])
        if S: S[-1].append(r)
    S = S[mp.WARM:]
    rows = []
    for i, s in enumerate(S):
        busy = sum(r['e'] - r['s'] for r in s)   # command buffers of one stream run back to back, so no overlap
        span = s[-1]['e'] - s[0]['s']
        wall = s[-1]['e'] - s[0]['ho']
        enc = sum(r['hc'] - r['ho'] for r in s)
        com = sum(r['hr'] - r['hc'] for r in s)
        gap = (S[i + 1][0]['s'] - s[-1]['e']) if i + 1 < len(S) else None
        rows.append((len(s), sum(r['nd'] for r in s), wall, busy, span - busy, enc, com, gap, s[0]['s'] - s[0]['ho']))
    m = lambda j: statistics.mean(x[j] for x in rows if x[j] is not None)
    print(f'| {b} | {m(0):.1f} | {m(1):.0f} | {m(2)*1e3:.2f} | {m(3)*1e3:.2f} | {m(4)*1e3:.3f} | {m(5)*1e3:.2f} | {m(6)*1e3:.2f} | {m(7)*1e3:.3f} | {m(8)*1e3:.3f} |')
