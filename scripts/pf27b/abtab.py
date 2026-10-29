#!/usr/bin/env python3
"""Lane PF27b (copy of scripts/pf27/abtab.py, base = pf27b-base-cli; adds TTFT): A/B table from a drive.sh steps.log: localeval speed runs grouped by arm (base / dev binary) and lane (MTP on = recipe
default, MTP off = +spoff); per size the median over runs of each run's median prefill / decode tok/s, every run listed.
usage: abtab.py raw/w2/steps.log [localeval root, default /opt/code/deepeval]"""
import collections, json, re, statistics as st, sys

log = open(sys.argv[1]).read()
root = (sys.argv[2] if len(sys.argv) > 2 else '/opt/code/deepeval') + '/'
runs = collections.defaultdict(list)  # (lane, arm) -> [rows by size]
for m in re.finditer(r'item (\d+) (\S+) speed: exit 0 (results/\S+)', log):
    full = m.group(2)
    arm = 'base' if 'pf27b-base-cli' in full else 'dev'
    lane = 'MTP off' if 'spoff' in full else 'MTP on'
    r = json.load(open(root + m.group(3)))
    runs[(lane, arm)].append({row['target_prompt_tokens']: row for row in r['speed']['rows']})
for lane in ('MTP on', 'MTP off'):
    b, d = runs.get((lane, 'base'), []), runs.get((lane, 'dev'), [])
    if not b or not d:
        continue
    print(f"\n{lane}: {len(b)} base runs, {len(d)} dev runs")
    print(f"| prompt | prefill base | prefill dev | ratio | decode base | decode dev | ratio | prefill runs base / dev |")
    print("|---|---|---|---|---|---|---|---|")
    for n in sorted(b[0]):
        pb = [x[n]['prefill_tok_s']['median'] for x in b]; pd = [x[n]['prefill_tok_s']['median'] for x in d]
        db = [x[n]['decode_tok_s']['median'] for x in b]; dd = [x[n]['decode_tok_s']['median'] for x in d]
        mpb, mpd, mdb, mdd = st.median(pb), st.median(pd), st.median(db), st.median(dd)
        tb = [x[n].get('ttft_s', {}).get('median', float('nan')) for x in b]; td = [x[n].get('ttft_s', {}).get('median', float('nan')) for x in d]
        print(f"| {n} | {mpb:.0f} | {mpd:.0f} | {mpd / mpb:.3f} | {mdb:.1f} | {mdd:.1f} | {mdd / mdb:.3f} | "
              f"{' '.join(f'{v:.0f}' for v in pb)} / {' '.join(f'{v:.0f}' for v in pd)} | TTFT {st.median(tb):.3f} / {st.median(td):.3f} s |")
