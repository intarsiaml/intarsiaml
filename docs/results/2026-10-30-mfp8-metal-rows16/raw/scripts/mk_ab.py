#!/usr/bin/env python3
"""mk_ab.py DIR : per-op ms/step (ce1, exclusive durations) of the dense fp8 sub-ops and the step total, base vs r16, B = 8, 9, 12, 16,
mean over the rounds in DIR (ce1-<arm>-r<N>.jsonl; odd rounds ascending B 8,9,12,16, even descending)."""
import sys, glob, collections, re
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import mp
d = sys.argv[1]
Bs = [8, 9, 12, 16]
ARMS = sys.argv[2].split(',') if len(sys.argv) > 2 else ['base', 'r16']
tab = collections.defaultdict(lambda: collections.defaultdict(list))  # (arm, key) -> B -> [ms]
for f in sorted(glob.glob(f'{d}/ce1-*-r*.jsonl*')):
    m = re.search(r'ce1-(\w+)-r(\d+)', f); arm, r = m.group(1), int(m.group(2))
    order = Bs if r % 2 else Bs[::-1]
    D = mp.by_b(f, 'ce1', order)
    for b in Bs:
        per = collections.defaultdict(float); n = len(D[b])
        for s in D[b]:
            for rec in s:
                g, sub = mp.kind(rec, b)
                per[sub] += rec['dur'] * 1e3 / n; per['GROUP ' + g] += rec['dur'] * 1e3 / n; per['TOTAL'] += rec['dur'] * 1e3 / n
        for k, v in per.items(): tab[(arm, k)][b].append(v)
keys = sorted({k for (_, k) in tab}, key=lambda k: (not k.startswith('TOTAL'), not k.startswith('GROUP'), k))
print('| op | arm | ' + ' | '.join(f'B={b}' for b in Bs) + ' |'); print('|---|---|' + '---|' * len(Bs))
for k in keys:
    if not (k == 'TOTAL' or 'fp8' in k or k.startswith('GROUP')): continue
    for arm in ARMS:
        v = tab.get((arm, k))
        if v: print(f'| {k} | {arm} | ' + ' | '.join(f'{sum(v[b])/len(v[b]):.2f}' if v[b] else '-' for b in Bs) + ' |')
