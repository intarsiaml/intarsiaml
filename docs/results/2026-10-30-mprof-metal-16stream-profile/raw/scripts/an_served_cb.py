#!/usr/bin/env python3
"""an_served_cb.py TRACE(cb) : served decode steps (embedding t = rows) in the normal-stream `cb` profile: command buffers / dispatches
per step, GPU busy, GPU idle inside and between steps, host encode / commit time. Steady steps only (rows == the run's full batch)."""
import sys, statistics, collections
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import mp
R = mp.load(sys.argv[1])
S = []
for r in R:
    if r['op'] == 'embedding_bf16': S.append([])
    if S: S[-1].append(r)
print('| rows | steps | cmd buffers / step | dispatches / step | GPU busy ms | GPU idle inside step ms | GPU idle before next step ms | step period ms (first cb start -> next first cb start) | host encode ms (sum ho->hc) | commit ms (sum hc->hr) |')
print('|---|---|---|---|---|---|---|---|---|---|')
for rows in (1, 8, 16):
    idx = [i for i, s in enumerate(S) if s[0]['t'] == rows]
    # steady: consecutive decode steps with the same rows, drop 8 at both ends
    runs = []; cur = []
    for i in idx:
        if cur and i != cur[-1] + 1: runs.append(cur); cur = []
        cur.append(i)
    runs.append(cur)
    run = max(runs, key=len)[8:-8]
    rec = []
    for i in run:
        s = S[i]; nxt = S[i + 1][0]['s']
        busy = sum(r['e'] - r['s'] for r in s)
        rec.append((len(s), sum(r['nd'] for r in s), busy, (s[-1]['e'] - s[0]['s']) - busy, nxt - s[-1]['e'], nxt - s[0]['s'],
                    sum(r['hc'] - r['ho'] for r in s), sum(r['hr'] - r['hc'] for r in s)))
    m = lambda j: statistics.mean(x[j] for x in rec)
    print(f'| {rows} | {len(rec)} | {m(0):.1f} | {m(1):.0f} | {m(2)*1e3:.2f} | {m(3)*1e3:.3f} | {m(4)*1e3:.3f} | {m(5)*1e3:.2f} | {m(6)*1e3:.2f} | {m(7)*1e3:.2f} |')
