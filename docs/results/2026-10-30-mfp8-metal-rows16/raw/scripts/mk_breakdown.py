#!/usr/bin/env python3
"""mk_breakdown.py : per-op breakdown table (ms / step and share) at B = 1, 4, 8, 16 (+ 9, 12 in ms) from the two ce1 real-traffic runs."""
import sys, collections
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import mp
specs = [('w1/ce1.jsonl.gz', [1, 4, 8, 9, 12, 16]), ('w1b/ce1.jsonl.gz', [16, 12, 9, 8, 4, 1])]
Bs = [1, 4, 8, 9, 12, 16]
tab = collections.defaultdict(lambda: collections.defaultdict(float)); cnt = collections.defaultdict(lambda: collections.defaultdict(float)); tot = collections.defaultdict(float)
for f, order in specs:
    D = mp.by_b(f, 'ce1', order)
    for b in Bs:
        n = len(D[b]) * len(specs)
        for s in D[b]:
            for r in s:
                k = mp.kind(r, b); tab[k][b] += r['dur'] * 1e3 / n; cnt[k][b] += 1 / n; tot[b] += r['dur'] * 1e3 / n
gb = collections.defaultdict(lambda: collections.defaultdict(float))
for (g, sub), v in tab.items():
    for b, x in v.items(): gb[g][b] += x
print('| op group | B=1 | B=4 | B=8 | B=9 | B=12 | B=16 | 8 -> 16 (ms, x) |')
print('|---|---|---|---|---|---|---|---|')
for g in sorted(gb, key=lambda g: -gb[g][16]):
    cells = [f'{gb[g][b]:.2f} ({gb[g][b] / tot[b] * 100:.0f}%)' for b in Bs]
    print(f'| **{g}** | ' + ' | '.join(cells) + f' | {gb[g][16] - gb[g][8]:+.2f}, x{gb[g][16] / gb[g][8]:.2f} |')
    for (g2, sub), v in sorted(tab.items(), key=lambda kv: -kv[1][16]):
        if g2 != g or v[16] < 0.05 and v[8] < 0.05: continue
        print(f'| &emsp;{sub} | ' + ' | '.join(f'{v[b]:.2f}' for b in Bs) + f' | {v[16] - v[8]:+.2f}, x{v[16] / v[8] if v[8] > 0.01 else 0:.2f} |')
print('| **total (sum of in-stream record times)** | ' + ' | '.join(f'{tot[b]:.2f}' for b in Bs) + f' | {tot[16] - tot[8]:+.2f}, x{tot[16] / tot[8]:.2f} |')
