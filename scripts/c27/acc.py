#!/usr/bin/env python3
"""Lane c27: acceptance / decode tok/s per prompt and arm from an ab.sh ids-mode log. acc.py <ids.log>"""
import re, sys, collections
d = collections.defaultdict(dict); ok = collections.defaultdict(dict)
for l in open(sys.argv[1], errors='replace'):
    m = re.match(r'ids-(\S+?)-(off|base|\w+): .*decode \d+ ms \(([\d.]+) tok/s\)(?:.*acceptance ([\d.]+)%, ([\d.]+) tokens)?', l)
    if m:
        p, a, v, acc, tp = m.groups(); d[p][a] = (float(v), acc, tp)
    m = re.match(r'IDS (\S+): (\w+) (==|DIFFERS)', l)
    if m: ok[m.group(1)][m.group(2)] = m.group(3) == '=='
arms = ['off', 'base'] + sorted({a for p in d.values() for a in p} - {'off', 'base'})
print(f"{'prompt':9s}" + "".join(f"{a:>26s}" for a in arms))
for p in d:
    row = f"{p:9s}"
    for a in arms:
        v = d[p].get(a)
        row += f"{(f'{v[0]:.1f} tok/s ' + (f'{v[1]}% {v[2]}tpp' if v[1] else '')) if v else '-':>26s}"
    print(row + "   ids==off: " + " ".join(f"{a}:{'yes' if ok[p].get(a) else 'NO'}" for a in arms if a != 'off' and a in ok[p]))
