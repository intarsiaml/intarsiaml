#!/usr/bin/env python3
"""Lane PF27c (from ARX)2: pool the localeval speed runs of one or more steps.logs by model / spec mode / arm (rec* = main, p1* = push) and print the median of the
per-run medians per size, with the spread, and the push / main ratio.
usage: speedtab2.py steps.log [steps.log ...]   (results are read from /opt/code/deepeval/)"""
import json, re, statistics as st, sys, collections
root = '/opt/code/deepeval/'
out = collections.defaultdict(lambda: {'pre': [], 'dec': []})
runs = collections.Counter()
for path in sys.argv[1:]:
    for m in re.finditer(r'item (\d+) (\S+) speed: exit 0 (results/\S+)', open(path).read()):
        full = m.group(2); arm = 'push' if full.startswith('p1') else 'main'
        grp = ('35B' if 'm35' in full else '27B') + (' spec off' if 'spoff' in full else ' default spec')
        runs[(grp, arm)] += 1
        for row in json.load(open(root + m.group(3)))['speed']['rows']:
            o = out[(grp, row['target_prompt_tokens'], arm)]
            o['pre'].append(row['prefill_tok_s']['median']); o['dec'].append(row['decode_tok_s']['median'])
def f(xs): return f"{st.median(xs):.1f} [{min(xs):.1f}-{max(xs):.1f}]"
for grp in sorted({k[0] for k in out}):
    print(f"## {grp}  (runs: main {runs[(grp,'main')]}, push {runs[(grp,'push')]})")
    for n in sorted({k[1] for k in out if k[0] == grp}):
        a, b = out.get((grp, n, 'main')), out.get((grp, n, 'push'))
        if not (a and b): continue
        print(f"{n:>6} prefill main {f(a['pre']):<22} push {f(b['pre']):<22} {100*(st.median(b['pre'])/st.median(a['pre'])-1):+.1f} %   "
              f"decode main {f(a['dec']):<22} push {f(b['dec']):<22} {100*(st.median(b['dec'])/st.median(a['dec'])-1):+.1f} %")
