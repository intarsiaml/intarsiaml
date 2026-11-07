#!/usr/bin/env python3
"""Lane d2r: decode tok/s per size and arm from an xg.sh steps.log (speed visits r<round>-<size>-<arm>). table.py <steps.log>"""
import re, sys, collections
d = collections.defaultdict(lambda: collections.defaultdict(list)); pf = collections.defaultdict(lambda: collections.defaultdict(list)); acc = collections.defaultdict(lambda: collections.defaultdict(list))
for l in open(sys.argv[1], errors='replace'):
    m = re.match(r'\S+ r(\d+)-(\w+)-(tp\w+): .*prefill \d+ ms \(([\d.]+) tok/s\) \| decode \d+ ms \(([\d.]+) tok/s\)(?:.*acceptance ([\d.]+)%, ([\d.]+) tokens)?', l)
    if m:
        r, sz, a, p, v, ac, tpp = m.groups(); d[sz][a].append((int(r), float(v))); pf[sz][a].append(float(p))
        if ac: acc[sz][a].append((float(ac), float(tpp)))
arms = ['tp1m', 'tp2off', 'tp2m']
print(f"{'size':5s}" + "".join(f"{a:>26s}" for a in arms) + f"{'tp2m/tp2off':>13s}{'tp2m/tp1m':>11s}")
for sz in ['1k', '4k', '16k']:
    if sz not in d: continue
    row = f"{sz:5s}"; mean = {}
    for a in arms:
        v = sorted(d[sz].get(a, []))
        if v:
            mean[a] = sum(x for _, x in v) / len(v)
            row += f"{mean[a]:8.2f} ({'/'.join(f'{x:.1f}' for _, x in v)})".rjust(26)
        else: row += f"{'-':>26s}"
    r1 = f"{mean['tp2m']/mean['tp2off']:.3f}x" if 'tp2m' in mean and 'tp2off' in mean else '-'
    r2 = f"{mean['tp2m']/mean['tp1m']:.3f}x" if 'tp2m' in mean and 'tp1m' in mean else '-'
    print(row + f"{r1:>13s}{r2:>11s}")
print("prefill tok/s (mean):")
for sz in ['1k', '4k', '16k']:
    if sz in pf: print(f"  {sz}: " + "  ".join(f"{a} {sum(pf[sz][a])/len(pf[sz][a]):.0f}" for a in arms if pf[sz].get(a)))
print("acceptance % / tokens per pass (mean):")
for sz in ['1k', '4k', '16k']:
    if sz in acc: print(f"  {sz}: " + "  ".join(f"{a} {sum(x for x,_ in acc[sz][a])/len(acc[sz][a]):.1f}% {sum(y for _,y in acc[sz][a])/len(acc[sz][a]):.2f}" for a in arms if acc[sz].get(a)))
