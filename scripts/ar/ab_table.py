#!/usr/bin/env python3
"""Lane ar: A/B table of the INTARSIA_HIP_AR_DRAIN arms from localeval speed results.
usage: ab_table.py LABEL_PREFIX [RESULTS_DIR]   e.g. ab_table.py ar-a1-speed27 (labels <prefix>-<arm>-r<N>)
Per prompt size: decode (and prefill) tok/s, mean of the per-round medians with (min-max) over rounds; delta vs d3 = mean of the
per-round paired ratios (same round), and the worst single-round pair."""
import glob, json, os, re, statistics, sys

prefix = sys.argv[1]
root = sys.argv[2] if len(sys.argv) > 2 else '/opt/code/deepeval/results'
data = {}  # (arm, size) -> {round: {'decode':..,'prefill':..}}
for f in glob.glob(os.path.join(root, '*', 'result.json')):
    d = json.load(open(f))
    m = re.fullmatch(re.escape(prefix) + r'-(d\d)-r(\d+)', (d.get('label') or ''))
    if not m or d.get('kind') != 'speed':
        continue
    arm, r = m.group(1), int(m.group(2))
    for row in d['speed']['rows']:
        dec, pre = row.get('decode_tok_s'), row.get('prefill_tok_s')
        data.setdefault((arm, row['target_prompt_tokens']), {})[r] = {
            'decode': dec['median'] if isinstance(dec, dict) else dec, 'prefill': pre['median'] if isinstance(pre, dict) else pre}
arms = sorted({a for a, _ in data}, key=lambda a: (a != 'd3', a))
sizes = sorted({s for _, s in data})
for metric in ('decode', 'prefill'):
    print(f'\n{prefix} {metric} tok/s: mean of per-round medians (min-max over rounds, n rounds); paired delta vs d3 (mean, worst round)')
    print('| size | ' + ' | '.join(arms) + ' |')
    print('|---|' + '---|' * len(arms))
    for s in sizes:
        cells = []
        for a in arms:
            v = data.get((a, s), {})
            xs = [x[metric] for x in v.values() if x[metric]]
            if not xs:
                cells.append('-'); continue
            c = f'{statistics.mean(xs):.2f} ({min(xs):.2f}-{max(xs):.2f}, n={len(xs)})'
            if a != 'd3':
                base = data.get(('d3', s), {})
                dl = [(v[r][metric] / base[r][metric] - 1) * 100 for r in v if r in base and v[r][metric] and base[r][metric]]
                if dl:
                    c += f' **{statistics.mean(dl):+.2f}%** (worst {min(dl):+.2f}%)'
            cells.append(c)
        print(f'| {s} | ' + ' | '.join(cells) + ' |')
