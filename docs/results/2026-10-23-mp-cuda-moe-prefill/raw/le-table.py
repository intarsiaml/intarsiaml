#!/usr/bin/env python3
"""Per-arm prefill/decode medians from a le-window.sh output dir: le-table.py <dir>  (arms vllm, def, pf; 3 rounds; sizes 1k/4k/16k)."""
import re, glob, sys, statistics as st
d = sys.argv[1]
R = {}
for f in sorted(glob.glob(d + '/mp-*.out')):
    arm = f.split('/')[-1].split('-')[1]
    by = {}
    for l in open(f):
        m = re.search(r'\] +(\d+) tok rep \d/3: prompt=(\d+) ttft=([\d.]+)s prefill=(\d+) tok/s decode=([\d.]+)', l)
        if m: by.setdefault(int(m.group(1)), []).append((int(m.group(4)), float(m.group(5))))
    for s, v in by.items():
        R.setdefault(arm, {}).setdefault(s, []).append((st.median(x[0] for x in v), st.median(x[1] for x in v), min(x[0] for x in v), max(x[0] for x in v)))
for arm in R:
    print(arm, 'prefill tok/s per round (median of 3 reps):', {s: [int(x[0]) for x in v] for s, v in R[arm].items()})
    print('   median over rounds:', {s: int(st.median(x[0] for x in v)) for s, v in R[arm].items()}, ' decode:', {s: round(st.median(x[1] for x in v), 1) for s, v in R[arm].items()})
for s in (1024, 4096, 16384):
    v, dd, p = (st.median(x[0] for x in R[a][s]) for a in ('vllm', 'def', 'pf'))
    print('%5d: def vs vLLM %+.1f %%, pf vs vLLM %+.1f %%, pf vs def %+.1f %%' % (s, (dd / v - 1) * 100, (p / v - 1) * 100, (p / dd - 1) * 100))
