#!/usr/bin/env python3
"""Lane ar: A/B table of the INTARSIA_HIP_AR_DRAIN arms from the category harness (localeval quality, MTP k2).
usage: ab_cats.py LABEL_PREFIX [RESULTS_DIR]   e.g. ab_cats.py ar-a2 (labels <prefix>-<arm>-<category>-r<N>)
Per category: mean decode tok/s of the dataset's rows, per arm: mean over rounds (min-max); paired delta vs d3 (same round)."""
import glob, json, os, re, statistics, sys

prefix = sys.argv[1]
root = sys.argv[2] if len(sys.argv) > 2 else '/opt/code/deepeval/results'
data = {}
for f in glob.glob(os.path.join(root, '*', 'result.json')):
    d = json.load(open(f))
    m = re.fullmatch(re.escape(prefix) + r'-(d\d)-([a-z]+-[a-z]+)-r(\d+)', d.get('label') or '')
    if not m or not d.get('datasets'):
        continue
    v = d['datasets'][0]['metrics'].get('Decode (tok/s)', {}).get('mean')
    if v:
        data.setdefault((m.group(1), m.group(2)), {})[int(m.group(3))] = v
arms = sorted({a for a, _ in data}, key=lambda a: (a != 'd3', a))
cats = sorted({c for _, c in data})
print(f'\n{prefix} MTP k2 decode tok/s per category (mean of rows; mean over rounds (min-max, n)); paired delta vs d3 (mean, worst round)')
print('| category | ' + ' | '.join(arms) + ' |'); print('|---|' + '---|' * len(arms))
allpairs = {a: [] for a in arms}
for c in cats + ['all']:
    cells = []
    for a in arms:
        if c == 'all':
            rs = sorted({r for cc in cats for r in data.get((a, cc), {})})
            xs = [statistics.mean([data[(a, cc)][r] for cc in cats if r in data.get((a, cc), {})]) for r in rs]
        else:
            xs = list(data.get((a, c), {}).values())
        if not xs:
            cells.append('-'); continue
        s = f'{statistics.mean(xs):.2f} ({min(xs):.2f}-{max(xs):.2f}, n={len(xs)})'
        if a != 'd3':
            if c == 'all':
                dl = [(statistics.mean([data[(a, cc)][r] for cc in cats]) / statistics.mean([data[('d3', cc)][r] for cc in cats]) - 1) * 100
                      for r in rs if all(r in data.get((a, cc), {}) and r in data.get(('d3', cc), {}) for cc in cats)]
            else:
                b = data.get(('d3', c), {}); v = data.get((a, c), {})
                dl = [(v[r] / b[r] - 1) * 100 for r in v if r in b]
            if dl:
                s += f' **{statistics.mean(dl):+.2f}%** (worst {min(dl):+.2f}%)'
        cells.append(s)
    print(f'| {c} | ' + ' | '.join(cells) + ' |')

# lane ars: recovery of the new arm = (new - d1) / (d3 - d1), per round, over the all-category mean
new = next((a for a in arms if a not in ('d1', 'd3', 'd5')), None)
if new and 'd1' in arms:
    print(f'\nRecovery of {new} vs d1 (1 = as fast as d3, 0 = as slow as d1), per category and over all categories')
    print(f'| category | recovery | {new} vs d1 |'); print('|---|---|---|')
    for c in cats + ['all']:
        rec, dv = [], []
        rs = sorted({r for cc in cats for r in data.get((new, cc), {})}) if c == 'all' else sorted(data.get((new, c), {}))
        for r in rs:
            def val(a):
                if c == 'all':
                    return statistics.mean([data[(a, cc)][r] for cc in cats]) if all(r in data.get((a, cc), {}) for cc in cats) else None
                return data.get((a, c), {}).get(r)
            x, a1, a3 = val(new), val('d1'), val('d3')
            if x and a1 and a3:
                dv.append((x / a1 - 1) * 100)
                if abs(a3 - a1) > 1e-9:
                    rec.append((x - a1) / (a3 - a1))
        if rec:
            print(f'| {c} | {statistics.mean(rec):.2f} (rounds {min(rec):.2f} to {max(rec):.2f}, n={len(rec)}) | {statistics.mean(dv):+.2f}% |')
