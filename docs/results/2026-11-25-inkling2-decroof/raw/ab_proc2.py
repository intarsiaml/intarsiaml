#!/usr/bin/env python3
"""Cross-binary ABBA (lane ink-decroof job X): DIR/q<i>_<arm>.err, each process runs prompts6.json (143 / 1021 / 4108 twice).
Per process: the mean of its two runs per length. Fast / slow mode: a process is fast if its 143 rate is within 4 % of the
best process of its arm."""
import glob, os, re, statistics as st, sys
O = sys.argv[1]; P = {}
for f in sorted(glob.glob(O + '/q*_*.err'), key=lambda f: int(re.search(r'q(\d+)_', f).group(1))):
    i, a = re.match(r'q(\d+)_(\w+)\.err', os.path.basename(f)).groups()
    d = {}
    for line in open(f):
        m = re.match(r'decode\[(\d+)\]: \d+ tokens, \d+ ms \(([\d.]+) tok/s\)', line)
        if m: d.setdefault(int(m.group(1)) % 3, []).append(float(m.group(2)))
    if len(d) == 3: P[(int(i), a)] = [st.fmean(d[k]) for k in range(3)]
for (i, a), v in sorted(P.items()): print(f'q{i} {a}: ' + ' '.join(f'{x:.2f}' for x in v))
best = {a: max(v[0] for (i, b), v in P.items() if b == a) for a in ('base', 'tip')}
fast = {k: v[0] >= 0.96 * best[k[1]] for k, v in P.items()}
for li, L in enumerate(('143', '1k', '4k')):
    out = []
    for sel, name in ((lambda k: True, 'all'), (lambda k: fast[k], 'fast')):
        b = [v[li] for k, v in P.items() if k[1] == 'base' and sel(k)]; t = [v[li] for k, v in P.items() if k[1] == 'tip' and sel(k)]
        if b and t: out.append(f'{name}: base {st.median(b):.2f} [{min(b):.2f}-{max(b):.2f}] n={len(b)}, tip {st.median(t):.2f} [{min(t):.2f}-{max(t):.2f}] n={len(t)}, tip/base {st.median(t) / st.median(b):.4f}')
    print(f'{L}: ' + ' | '.join(out))
