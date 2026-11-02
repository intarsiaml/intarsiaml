#!/usr/bin/env python3
"""plain_ab.py DIR [plain|ce1]: per-arm ms/step per B from DIR/<kind>-<arm>-r<round>.log (REALTRAFFIC lines): per-round values, median."""
import re, sys, glob, collections, statistics
d = sys.argv[1]; kind = sys.argv[2] if len(sys.argv) > 2 else 'plain'
v = collections.defaultdict(lambda: collections.defaultdict(list))
for f in sorted(glob.glob(f'{d}/{kind}-*-r*.log')):
    m = re.match(rf'.*/{kind}-(.+)-r(\d+)\.log', f); arm, r = m.group(1), int(m.group(2))
    for l in open(f):
        mm = re.search(r'REALTRAFFIC B=\s*(\d+): ([\d.]+) ms/step', l)
        if mm: v[arm][int(mm.group(1))].append((r, float(mm.group(2))))
bs = sorted({b for a in v for b in v[a]})
print('| arm | ' + ' | '.join(f'B={b}' for b in bs) + ' |'); print('|---|' + '---:|' * len(bs))
for a in sorted(v):
    print(f'| {a} median | ' + ' | '.join(f'{statistics.median(x for _, x in v[a][b]):.2f}' for b in bs) + ' |')
    print(f'| {a} rounds | ' + ' | '.join('/'.join(f'{x:.2f}' for _, x in sorted(v[a][b])) for b in bs) + ' |')
