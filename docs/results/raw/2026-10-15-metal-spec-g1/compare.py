#!/usr/bin/env python3
"""compare.py <dir> [ref=off]: per case, stdout bytes + ids line + finish/completion of every variant vs the reference
variant; spec counters, decode speed, trace evidence (mid-run stops, min_tokens cuts) of the spec-on runs."""
import sys, os, re, glob, collections
d = sys.argv[1]; ref = sys.argv[2] if len(sys.argv) > 2 else 'off'
runs = collections.defaultdict(dict)
for f in sorted(glob.glob(os.path.join(d, '*.err'))):
    v, case = os.path.basename(f)[:-4].split('-', 1)
    runs[case][v] = f[:-4]
def info(p):
    err = open(p + '.err').read(); out = open(p + '.out', 'rb').read()
    ids = re.search(r'^ids: (.*)$', err, re.M); fin = re.search(r'^finish=(\S+) prompt=(\d+).*completion=(\d+)', err, re.M)
    dec = re.search(r'decode (\d+) ms \(([\d.]+) tok/s\)', err); pre = re.search(r'prefill (\d+) ms', err)
    sp = re.search(r'spec: verify passes=(\d+) proposed=(\d+) accepted=(\d+)', err)
    steps = re.findall(r'^spec: token (\d+) pos (\d+) drafted (\d+) accepted (\d+) emits (\d+)(.*)$', err, re.M)
    fin_tr = re.search(r'^spec: finished at token (\d+) with (\d+) verified tokens unemitted; KV (\d+) -> (\d+)', err, re.M)
    return dict(out=out, ids=ids.group(1) if ids else None, fin=fin.groups() if fin else None,
                dec_ms=int(dec.group(1)) if dec else None, tps=float(dec.group(2)) if dec else None,
                pre_ms=int(pre.group(1)) if pre else None, spec=tuple(map(int, sp.groups())) if sp else None,
                steps=steps, fin_tr=fin_tr.groups() if fin_tr else None, err='error' in err.lower() and not ids)
print(f"| case | variant | stdout == {ref} | ids == {ref} | finish / completion | decode tok/s | verify passes | proposed | accepted | acceptance | tokens/pass | ended mid-run | min_tokens cuts |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
allok = True
for case in sorted(runs):
    vs = runs[case]
    if ref not in vs:
        continue
    r = info(vs[ref])
    for v in sorted(vs, key=lambda x: (x != ref, x)):
        i = info(vs[v])
        so = i['out'] == r['out']; io = i['ids'] is not None and i['ids'] == r['ids']; fo = i['fin'] == r['fin']
        if v != ref:
            allok &= so and io and fo
        s = i['spec']
        acc = f"{100*s[2]/max(s[1],1):.1f}%" if s else '-'
        tpp = f"{(s[0]+s[2])/max(s[0],1):.2f}" if s else '-'
        mid = '-'
        if i['fin_tr']:
            mid = f"yes ({i['fin_tr'][1]} unemitted, KV {i['fin_tr'][2]}->{i['fin_tr'][3]})"
        elif s:
            mid = 'no'
        cuts = sum(1 for st in i['steps'] if 'cut' in st[5]) if s else '-'
        fin = f"{i['fin'][0]} / {i['fin'][2]}" if i['fin'] else 'MISSING'
        print(f"| {case} | {v} | {'ref' if v == ref else ('yes' if so else 'NO')} | {'ref' if v == ref else ('yes' if io else 'NO')} | {fin}{'' if fo or v == ref else ' (DIFF)'} | {i['tps']} | {s[0] if s else '-'} | {s[1] if s else '-'} | {s[2] if s else '-'} | {acc} | {tpp} | {mid} | {cuts} |")
print()
print('ALL IDENTICAL' if allok else 'MISMATCH')
