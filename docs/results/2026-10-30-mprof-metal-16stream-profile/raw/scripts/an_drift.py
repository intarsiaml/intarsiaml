#!/usr/bin/env python3
"""an_drift.py TRACE(ce1 served) : lm_head (data-independent, nvfp4 248320x2048) and fp8 8192x2048 time per decode step, binned over
the steady D1 / D8 / D16 runs, to see whether kernel times ramp up with sustained load (clock / power state)."""
import sys, statistics
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import mp
R = mp.load(sys.argv[1]); mp.exclusive(R)
S = []
for r in R:
    if r['op'] == 'embedding_bf16': S.append([])
    if S: S[-1].append(r)
t0 = R[0]['s']
def runs(rows):
    idx = [i for i, s in enumerate(S) if s[0]['t'] == rows]
    out = []; cur = []
    for i in idx:
        if cur and i != cur[-1] + 1: out.append(cur); cur = []
        cur.append(i)
    out.append(cur)
    return [r for r in out if len(r) >= 100]
def lm(s): return sum(r['dur'] for r in s if r['n'] == 248320) * 1e3
def f8(s): return sum(r['dur'] for r in s if r['op'].startswith('matmul_fp8')) * 1e3
def step(s): return sum(r['dur'] for r in s) * 1e3
print('| run | steps | elapsed s since trace start | lm_head ms (mean of 50-step bin) | fp8 ms | whole step ms (sum of records) |')
print('|---|---|---|---|---|---|')
for rows in (1, 8, 16):
    for run in runs(rows):
        for a in range(0, len(run), 50):
            b = run[a:a + 50]
            if len(b) < 50: continue
            ss = [S[i] for i in b]
            print(f'| D{rows} | {a}-{a + 49} | {ss[0][0]["s"] - t0:.1f} | {statistics.mean(lm(s) for s in ss):.3f} | {statistics.mean(f8(s) for s in ss):.2f} | {statistics.mean(step(s) for s in ss):.2f} |')
