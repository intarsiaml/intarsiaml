#!/usr/bin/env python3
"""Tables from .lane/ab/*-localeval.{out,err}: median over rounds of the per-round median (decode, prefill) and burst aggregates."""
import re, glob, statistics as st, sys, os
D = sys.argv[1] if len(sys.argv) > 1 else '.lane/ab'
rows = {}  # (mode, arm, metric, size) -> [per round values]
for f in sorted(glob.glob(f'{D}/*-localeval.out')):
    tag = os.path.basename(f)[:-len('-localeval.out')]
    mode, arm, rnd = tag.split('-')
    txt = open(f).read()
    per = {}
    for m in re.finditer(r'^\[localeval\]\s+(\d+) tok rep \d+/\d+: prompt=(\d+) .*?prefill=(\d+) tok/s decode=([\d.]+) tok/s', txt, re.M):
        per.setdefault(int(m[1]), []).append((float(m[3]), float(m[4])))
    for size, v in per.items():
        rows.setdefault((mode, arm, 'prefill', size), []).append(st.median(x[0] for x in v))
        rows.setdefault((mode, arm, 'decode', size), []).append(st.median(x[1] for x in v))
    err = open(f.replace('.out', '.err')).read() if os.path.exists(f.replace('.out', '.err')) else ''
    for m in re.finditer(r'(\d+) streams: aggregate ([\d.]+) tok/s', txt + err):
        rows.setdefault((mode, arm, 'agg', int(m[1])), []).append(float(m[2]))
keys = sorted({(k[0], k[2], k[3]) for k in rows}, key=lambda k: (k[0], k[1], k[2]))
print('| mode | metric | size/streams | main rounds | dev rounds | main median | dev median | change |')
print('|---|---|---:|---|---|---:|---:|---:|')
for mode, metric, size in keys:
    a, b = rows.get((mode, 'main', metric, size), []), rows.get((mode, 'dev', metric, size), [])
    if not a or not b: continue
    ma, mb = st.median(a), st.median(b)
    print(f'| {mode} | {metric} | {size} | {" / ".join(f"{x:.1f}" for x in a)} | {" / ".join(f"{x:.1f}" for x in b)} | {ma:.1f} | {mb:.1f} | {100*(mb/ma-1):+.1f} % |')
