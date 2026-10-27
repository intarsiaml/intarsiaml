#!/usr/bin/env python3
"""Lane HDC: pool the localeval speed runs of a drive.sh steps.log by arm (the ARM[+mods] part of the item) and print, per prompt size,
n, median of run medians, min-max, and the ratio to the base arm.
usage: speedtab2.py steps.log [base-arm] [results root (default /opt/code/deepeval)] [filter substring on the item, e.g. spoff or -spoff]"""
import json, re, statistics as st, sys, collections
log = open(sys.argv[1]).read()
base = sys.argv[2] if len(sys.argv) > 2 else 'base+spoff'
root = (sys.argv[3] if len(sys.argv) > 3 else '/opt/code/deepeval') + '/'
flt = sys.argv[4] if len(sys.argv) > 4 else ''
runs = collections.defaultdict(list)
for m in re.finditer(r'item (\d+) (\S+) speed: exit 0 (results/\S+)', log):
    if flt:
        neg = flt.startswith('-')
        if (flt.lstrip('-') in m.group(2)) == neg: continue
    runs[m.group(2)].append(json.load(open(root + m.group(3))))
out = collections.defaultdict(lambda: collections.defaultdict(lambda: {'pre': [], 'dec': []}))
for arm, rs in runs.items():
    for r in rs:
        for row in r['speed']['rows']:
            n = row['target_prompt_tokens']
            out[arm][n]['pre'].append(row['prefill_tok_s']['median']); out[arm][n]['dec'].append(row['decode_tok_s']['median'])
sizes = sorted({n for a in out.values() for n in a})
print(f"{'size':>6} {'arm':<14} {'n':>2} {'decode med':>10} {'min-max':>13} {'spread%':>8} {'ratio':>7} | {'prefill med':>11} {'ratio':>6}")
for n in sizes:
    b = out.get(base, {}).get(n)
    for arm in sorted(out, key=lambda a: (a != base, a)):
        o = out[arm].get(n)
        if not o: continue
        d, p = o['dec'], o['pre']
        md = st.median(d); sp = 100 * (max(d) - min(d)) / md
        rd = f"{md / st.median(b['dec']):.3f}" if b else '-'
        rp = f"{st.median(p) / st.median(b['pre']):.3f}" if b else '-'
        print(f"{n:>6} {arm:<14} {len(d):>2} {md:>10.2f} {min(d):>6.1f}-{max(d):<6.1f} {sp:>8.1f} {rd:>7} | {st.median(p):>11.0f} {rp:>6}")
