#!/usr/bin/env python3
"""make_doc_table.py <identity dir>: one row per case (base = spike/metal c556008; off / on4 / on6 = this branch)."""
import sys, os, re, glob, collections
d = sys.argv[1]
runs = collections.defaultdict(dict)
for f in glob.glob(os.path.join(d, '*.err')):
    v, case = os.path.basename(f)[:-4].split('-', 1)
    runs[case][v] = f[:-4]
def info(p):
    err = open(p + '.err').read(); out = open(p + '.out', 'rb').read()
    ids = re.search(r'^ids: (.*)$', err, re.M).group(1)
    fin = re.search(r'^finish=(\S+) prompt=(\d+).*completion=(\d+)', err, re.M).groups()
    tps = float(re.search(r'decode \d+ ms \(([\d.]+) tok/s\)', err).group(1))
    sp = re.search(r'spec: verify passes=(\d+) proposed=(\d+) accepted=(\d+)', err)
    fin_tr = re.search(r'^spec: finished at token (\d+) with (\d+) verified tokens unemitted', err, re.M)
    cuts = len(re.findall(r'min_tokens cut', err))
    return dict(key=(out, ids, fin), fin=fin, tps=tps, sp=tuple(map(int, sp.groups())) if sp else None,
                mid=int(fin_tr.group(2)) if fin_tr else 0, cuts=cuts)
order = sys.argv[2].split(',') if len(sys.argv) > 2 else ['short-48', 'short-256', '4k-48', '4k-256', 'grounded-256', 'grounded-512', 'open-256', 'grounded-max102',
         'grounded-max188', '4k-max9', '4k-max44', 'grounded-stop-topp', 'grounded-stop-vecstring', 'grounded-stop-atomic',
         '4k-stop250', 'short-stopkettles', 'grounded-min331', 'grounded-min335', 'grounded-min320-stopfence']
print('| case | prompt tok | finish / tokens | off == base | on4 == base | on6 == base | acceptance on4 / on6 | tokens per pass on4 / on6 | verified tokens dropped at the stop, on4 / on6 | min_tokens cuts on4 / on6 | decode tok/s base / off / on4 / on6 |')
print('|---|---|---|---|---|---|---|---|---|---|---|')
tot = collections.Counter()
for c in order:
    r = {v: info(p) for v, p in runs[c].items()}
    b = r['base']
    eq = lambda v: 'yes' if r[v]['key'] == b['key'] else '**NO**'
    acc = lambda v: f"{100*r[v]['sp'][2]/max(r[v]['sp'][1],1):.0f}%"
    tpp = lambda v: f"{(r[v]['sp'][0]+r[v]['sp'][2])/max(r[v]['sp'][0],1):.2f}"
    for v in ('on4', 'on6'):
        tot[v + 'p'] += r[v]['sp'][1]; tot[v + 'a'] += r[v]['sp'][2]; tot[v + 's'] += r[v]['sp'][0]
    print(f"| {c} | {b['fin'][1]} | {b['fin'][0]} / {b['fin'][2]} | {eq('off')} | {eq('on4')} | {eq('on6')} | {acc('on4')} / {acc('on6')} | {tpp('on4')} / {tpp('on6')} | {r['on4']['mid']} / {r['on6']['mid']} | {r['on4']['cuts']} / {r['on6']['cuts']} | {b['tps']:.1f} / {r['off']['tps']:.1f} / {r['on4']['tps']:.1f} / {r['on6']['tps']:.1f} |")
print()
for v in ('on4', 'on6'):
    print(f"{v}: {tot[v+'s']} verify passes, {tot[v+'p']} proposed, {tot[v+'a']} accepted ({100*tot[v+'a']/tot[v+'p']:.1f}%)")
