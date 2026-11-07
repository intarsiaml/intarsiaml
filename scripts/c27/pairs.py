#!/usr/bin/env python3
"""Lane c27: paired ratios from an ab.sh log. pairs.py <speed.log> <base arm> [arm ...]
Lines: `r<round>-<size>-<arm>: prefill .. | decode N ms (X tok/s) | spec: ... (acceptance A%, T tokens per pass)`; prints per size and arm the decode tok/s per round, the paired ratio to the base arm per round and its median,
and the acceptance."""
import re, sys, statistics as st, collections
log, base, *arms = sys.argv[1:]
d = collections.defaultdict(dict); acc = {}
for l in open(log, errors='replace'):
    m = re.match(r'r(\d+)-(\w+)-(\w+): .*decode \d+ ms \(([\d.]+) tok/s\)(?:.*acceptance ([\d.]+)%, ([\d.]+) tokens)?', l)
    if m:
        r, sz, arm, v, a, tp = m.groups(); d[(sz, arm)][int(r)] = float(v)
        if a: acc[(sz, arm)] = (a, tp)
for sz in sorted({k[0] for k in d}):
    b = d.get((sz, base), {})
    for arm in [base] + arms:
        v = d.get((sz, arm), {})
        if not v: continue
        rs = sorted(set(v) & set(b))
        ratios = [v[r] / b[r] for r in rs]
        a = acc.get((sz, arm), ('', ''))
        print(f"{sz:4s} {arm:8s} tok/s " + " ".join(f"{v[r]:6.2f}" for r in sorted(v)) + (f" | ratio to {base}: " + " ".join(f"{x:.3f}" for x in ratios) + f" median {st.median(ratios):.3f}" if arm != base and ratios else "") + (f" | acceptance {a[0]}%, {a[1]} tok/pass" if a[0] else ""))
