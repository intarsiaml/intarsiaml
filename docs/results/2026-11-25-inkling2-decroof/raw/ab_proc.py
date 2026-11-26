#!/usr/bin/env python3
"""Cross-process ABBA summary (lane ink-decroof): DIR/q<i>_<arm>.err, each process runs the 143 / 1021 / 4108 prompts once."""
import glob, os, re, statistics as st, sys
O = sys.argv[1]; R = {}
for f in sorted(glob.glob(O + '/q*_*.err')):
    i, a = re.match(r'q(\d+)_(\w+)\.err', os.path.basename(f)).groups()
    for line in open(f):
        m = re.match(r'decode\[(\d+)\]: \d+ tokens, \d+ ms \(([\d.]+) tok/s\)', line)
        if m: R.setdefault((a, int(m.group(1))), []).append((int(i), float(m.group(2))))
for li, L in enumerate(('143', '1k', '4k')):
    line = []
    for a in ('base', 'cand'):
        xs = [v for _, v in sorted(R.get((a, li), []))]
        if xs: line.append(f'{a} median {st.median(xs):.2f} [{min(xs):.2f}-{max(xs):.2f}] n={len(xs)} ({" ".join(f"{x:.2f}" for x in xs)})')
    b, c = R.get(('base', li)), R.get(('cand', li))
    if b and c: line.append(f'cand/base {st.median(v for _, v in c) / st.median(v for _, v in b):.4f}')
    print(f'{L}: ' + ' | '.join(line))
