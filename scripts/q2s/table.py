#!/usr/bin/env python3
"""Lane q2s: decode tok/s per size and arm (speed visits r<round>-<size>-<arm>), sampled acceptance per arm (smp-<prompt>-<arm>) from
xg.sh steps.log files. table.py <steps.log>... ; arms are t<tp>_<quant>_<k> or t<tp>_off."""
import re, sys, collections
d = collections.defaultdict(lambda: collections.defaultdict(list)); acc = collections.defaultdict(lambda: collections.defaultdict(list))
smp = collections.defaultdict(lambda: collections.defaultdict(list))
for f in sys.argv[1:]:
    for l in open(f, errors='replace'):
        m = re.match(r'\S+ r(\d+)-(\w+)-(t\d_\w+): .*prefill \d+ ms \(([\d.]+) tok/s\) \| decode \d+ ms \(([\d.]+) tok/s\)(?:.*acceptance ([\d.]+)%, ([\d.]+) tokens)?', l)
        if m:
            r, sz, a, p, v, ac, tpp = m.groups(); d[sz][a].append((int(r), float(v)))
            if ac: acc[sz][a].append((float(ac), float(tpp)))
        m = re.match(r'\S+ smp-(\S+?)-(t\d_\w+): .*decode \d+ ms \(([\d.]+) tok/s\)(?:.*acceptance ([\d.]+)%, ([\d.]+) tokens)?', l)
        if m:
            p, a, v, ac, tpp = m.groups(); smp[a][p].append((float(v), float(ac) if ac else None, float(tpp) if tpp else None))
arms = sorted({a for s in d.values() for a in s})
print("decode tok/s, mean (rounds):")
print(f"{'size':5s}" + "".join(f"{a:>26s}" for a in arms))
for sz in ['1k', '4k', '16k']:
    if sz not in d: continue
    row = f"{sz:5s}"
    for a in arms:
        v = sorted(d[sz].get(a, []))
        row += (f"{sum(x for _, x in v)/len(v):8.2f} ({'/'.join(f'{x:.1f}' for _, x in v)})" if v else '-').rjust(26)
    print(row)
print("acceptance % / tokens per pass (mean):")
for sz in ['1k', '4k', '16k']:
    if sz in acc: print(f"  {sz}: " + "  ".join(f"{a} {sum(x for x,_ in acc[sz][a])/len(acc[sz][a]):.1f}% {sum(y for _,y in acc[sz][a])/len(acc[sz][a]):.2f}" for a in arms if acc[sz].get(a)))
if smp:
    print("sampled (T 0.7, top-p 0.95), per prompt: decode tok/s, acceptance %, tokens per pass")
    for a in sorted(smp):
        rows = smp[a]; print(f"  {a}: " + "  ".join(f"{p} {v[0][0]:.1f}/{v[0][1]}%/{v[0][2]}" for p, v in rows.items()))
        ac = [v[0][1] for v in rows.values() if v[0][1] is not None]; tp = [v[0][2] for v in rows.values() if v[0][2] is not None]
        if ac: print(f"     mean acceptance {sum(ac)/len(ac):.1f}%, tokens/pass {sum(tp)/len(tp):.2f}, decode {sum(v[0][0] for v in rows.values())/len(rows):.1f} tok/s")
