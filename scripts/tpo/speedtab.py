#!/usr/bin/env python3
"""Lane TPO: pool the localeval speed runs of raw/w6d/steps.log by arm (rec = main, best = overlap) and print median-of-run-medians per size.
usage: speedtab.py raw/w6d/steps.log [results dir (default /opt/code/deepeval)]"""
import json, re, statistics as st, sys, collections
log = open(sys.argv[1]).read()
root = (sys.argv[2] if len(sys.argv) > 2 else '/opt/code/deepeval') + '/'
runs = collections.defaultdict(list)
for m in re.finditer(r'item (\d+) (\S+) speed: exit 0 (results/\S+)', log):
    runs['main' if m.group(2) == 'rec' else 'overlap'].append((int(m.group(1)), json.load(open(root + m.group(3)))))
out = {}
for arm, rs in runs.items():
    for _, r in rs:
        for row in r['speed']['rows']:
            n = row['target_prompt_tokens']
            out.setdefault((arm, n), {'pre': [], 'dec': [], 'peak': [], 'items': []})
            o = out[(arm, n)]
            o['pre'].append(row['prefill_tok_s']['median']); o['dec'].append(row['decode_tok_s']['median'])
            o['peak'].append((row.get('peak_decode_tok_s') or {}).get('median'))
print(f"{'size':>6} {'arm':>8} {'runs':>4} {'prefill med':>12} {'(all)':<28} {'decode med':>10} {'(all)'}")
for n in sorted({k[1] for k in out}):
    for arm in ('main', 'overlap'):
        o = out.get((arm, n))
        if o:
            print(f"{n:>6} {arm:>8} {len(o['pre']):>4} {st.median(o['pre']):>12.0f} {str([round(x) for x in o['pre']]):<28} {st.median(o['dec']):>10.1f} {[round(x,1) for x in o['dec']]}")
    a, b = out.get(('main', n)), out.get(('overlap', n))
    if a and b:
        print(f"{n:>6} {'ratio':>8} {'':>4} {st.median(b['pre'])/st.median(a['pre']):>12.3f} {'':<28} {st.median(b['dec'])/st.median(a['dec']):>10.3f}")
