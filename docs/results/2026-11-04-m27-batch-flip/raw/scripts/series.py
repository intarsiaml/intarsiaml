#!/usr/bin/env python3
"""Per-row KL(solo||busy) over rows sharing context; bucketed, plus spikes."""
import json, math, statistics as st
def rm(r): return {int(x[0]): x[1] for x in r}
L = json.load(open('.scratch/logs/exp3/len.json')); L.update(json.load(open('.scratch/logs/exp4/nopeel-len.json')))
A = json.load(open('.scratch/logs/sweep/all.json')); L['772'] = {'solo': A['q4-solo'], 'behind': A['q4-behind']}
for k in ['258','260','262','265','516','772','1028','1540']:
    a, b = L[k]['solo'], L[k]['behind']
    d = next((i for i, (x, y) in enumerate(zip(a['ids'], b['ids'])) if x != y), 200); n = d
    kl = []; dmax = []
    for t in range(n):
        x, y = rm(a['top'][t]), rm(b['top'][t])
        kl.append(sum(math.exp(x[i]) * (x[i] - y[i]) for i in x.keys() & y.keys()))
        top = sorted(x, key=x.get, reverse=True)[:3]
        dmax.append(max(abs(x[i] - y[i]) for i in top if i in y))
    B = [(lo, min(lo + 25, n)) for lo in range(0, n, 25)]
    med = [f"{st.median(dmax[l:h]):.1e}" for l, h in B if h > l]
    print(k, 'rows', n, 'row0 top3 dlp', f"{dmax[0]:.2e}", 'median top3 dlp per 25 rows', ' '.join(med))
    sp = sorted(range(n), key=lambda t: -kl[t])[:3]
    print('   spikes (row, KL, top3 dlp, top lp solo):', [(t, round(kl[t], 4), round(dmax[t], 3), round(max(rm(a['top'][t]).values()), 3)) for t in sp])
