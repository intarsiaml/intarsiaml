#!/usr/bin/env python3
"""Collect localeval speed rows from w3/*-localeval.out: per arm, per round, prefill and decode medians at 1k/4k/16k."""
import glob, re, sys, statistics, collections, os
D = sys.argv[1]
rows = collections.defaultdict(lambda: collections.defaultdict(list))  # (model, arm) -> size -> [(round, pf, dec)]
for f in sorted(glob.glob(f'{D}/*-localeval.out')):
    r, m, arm = os.path.basename(f).split('-')[:3]
    for line in open(f):
        c = [x.strip() for x in line.split('|')]
        if len(c) < 7 or not re.match(r'^\d+$', c[1] or ''):
            continue
        size = int(c[1])
        if size < 900:
            continue
        pf = float(c[3].split()[0])
        dec = float(c[5].split()[0])
        key = '1k' if size < 2000 else '4k' if size < 8000 else '16k'
        rows[(m, arm)][key].append((int(r), pf, dec))
for (m, arm), d in sorted(rows.items()):
    out = []
    for key in ('1k', '4k', '16k'):
        v = sorted(d.get(key, []))
        if not v:
            continue
        pfs = [x[1] for x in v]; decs = [x[2] for x in v]
        out.append(f"{key}: pf {statistics.median(pfs):.0f} ({'/'.join(f'{x:.0f}' for x in pfs)}) dec {statistics.median(decs):.1f} ({'/'.join(f'{x:.1f}' for x in decs)})")
    print(f'{m}B {arm:5s} ' + ' | '.join(out))
