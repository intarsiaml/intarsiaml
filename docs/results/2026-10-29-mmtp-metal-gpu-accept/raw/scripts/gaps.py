#!/usr/bin/env python3
"""MMTP: per-record gap dump of a cb trace (decode region). gaps.py FILE [first_rec] [count]"""
import gzip, json, sys
f = sys.argv[1]
op = gzip.open if f.endswith('.gz') else open
R = sorted((json.loads(l) for l in op(f, 'rt')), key=lambda r: r['s'])
a = int(sys.argv[2]) if len(sys.argv) > 2 else 0
n = int(sys.argv[3]) if len(sys.argv) > 3 else 60
pe = None
for r in R[a:a + n]:
    gap = (r['s'] - pe) * 1e6 if pe else 0
    ho = (r['ho'] - pe) * 1e6 if pe else 0
    hc = (r['hc'] - pe) * 1e6 if pe else 0
    print(f"{r['i']:4d} {r['op'][:34]:34s} t{r['t']:<5d} nd{r['nd']:4d} dur{(r['e']-r['s'])*1e6:8.0f} gap{gap:7.0f} ho{ho:8.0f} hc{hc:8.0f}")
    pe = max(pe or 0, r['e'])
