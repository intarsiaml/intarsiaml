#!/usr/bin/env python3
"""an_groups.py TRACE.jsonl.gz MODE(ce1|kernel) [ORDER e.g. 1,4,8,9,12,16]: per-op-group ms / step at each B, share at the
largest B, 8 -> 16 delta. ce1: exclusive durations (continuous stream); kernel: serialized per-launch durations."""
import sys, collections
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import mp
# usage: an_groups.py MODE FILE:ORDER [FILE:ORDER ...]  (several traces of the same B set are averaged)
mode = sys.argv[1]
specs = [a.split(':') for a in sys.argv[2:]]
f = ' + '.join(a[0] for a in specs)
Bs = sorted(int(x) for x in specs[0][1].split(','))
Ds = [mp.by_b(a[0], mode, [int(x) for x in a[1].split(',')]) for a in specs]
class _M:  # merged view: per B the concatenation of the step lists, weights equalised by averaging per trace below
    pass
order = Bs
tab = collections.defaultdict(lambda: collections.defaultdict(float))   # (group, sub) -> B -> ms/step
cnt = collections.defaultdict(lambda: collections.defaultdict(float))
tot = {}
for b in Bs:
    for D in Ds:
        st = D[b]; n = len(st) * len(Ds)
        tot[b] = tot.get(b, 0) + sum(r['dur'] for s in st for r in s) / n * 1e3
        for s in st:
            for r in s:
                k = mp.kind(r, b)
                tab[k][b] += r['dur'] * 1e3 / n
                cnt[k][b] += 1 / n
print(f'# {f} mode={mode}: ms per decode step by op group (mean over steps after {mp.WARM} warm-up)')
hdr = '| group / op | ' + ' | '.join(f'B={b}' for b in Bs) + ' | share B=16 | launches B=1 | launches B=16 | 8->16 delta ms | x |'
print(hdr); print('|' + '---|' * (len(Bs) + 6))
grp = collections.defaultdict(float)
gb = collections.defaultdict(lambda: collections.defaultdict(float))
for (g, sub), v in tab.items():
    for b, x in v.items(): gb[g][b] += x
for g in sorted(gb, key=lambda g: -gb[g][16]):
    d = gb[g][16] - gb[g][8]
    print(f'| **{g}** | ' + ' | '.join(f'{gb[g][b]:.2f}' for b in Bs) + f' | {gb[g][16]/tot[16]*100:.1f}% | | | {d:+.2f} | {gb[g][16]/gb[g][8]:.2f} |')
    for (g2, sub), v in sorted(tab.items(), key=lambda kv: -kv[1][16]):
        if g2 != g: continue
        d = v[16] - v[8]
        print(f'| &nbsp;&nbsp;{sub} | ' + ' | '.join(f'{v[b]:.3f}' for b in Bs) + f' | {v[16]/tot[16]*100:.1f}% | {cnt[(g2,sub)][1]:.0f} | {cnt[(g2,sub)][16]:.0f} | {d:+.3f} | {v[16]/v[8] if v[8] else 0:.2f} |')
print('| **total (sum of records)** | ' + ' | '.join(f'{tot[b]:.2f}' for b in Bs) + ' | 100% | | | ' + f'{tot[16]-tot[8]:+.2f} | {tot[16]/tot[8]:.2f} |')
