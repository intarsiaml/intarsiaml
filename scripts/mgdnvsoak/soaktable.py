#!/usr/bin/env python3
"""mgdnvsoak: summarise raw/soak (run 4) and raw/lc-* (run 2). python3 soaktable.py <raw dir>"""
import collections, glob, json, os, statistics, sys
raw = sys.argv[1]
f = os.path.join(raw, "soak/soak.jsonl")
if os.path.exists(f):
    recs = [json.loads(l) for l in open(f)]
    errs = [r for r in recs if "error" in r]; bad = [r for r in recs if r.get("bad")]
    ok = [r for r in recs if "error" not in r]
    by = collections.defaultdict(set)
    for r in ok: by[r["kind"]].add(r["sha"])
    print(f"soak: {len(recs)} requests, {len(errs)} errors, {len(bad)} flagged bad, {sum(r.get('toks',0) for r in ok)} tokens, span {max(r['t'] for r in recs)-min(r['t'] for r in recs):.0f} s")
    print(f"distinct kinds {len(by)}; kinds with more than one reply sha: {[k for k,v in by.items() if len(v)>1]}")
    for r in errs[:5] + bad[:5]: print(" ", r)
    tsv = os.path.join(raw, "soak/mon.tsv")
    if os.path.exists(tsv): print(open(tsv).read())
for d in sorted(glob.glob(os.path.join(raw, "lc-*"))):
    print("==", os.path.basename(d))
    rows = collections.defaultdict(dict)
    for jf in sorted(glob.glob(d + "/*-r*.jsonl")):
        arm, rnd = os.path.basename(jf)[:-6].split("-r")
        for l in open(jf):
            if l.startswith("{"):
                r = json.loads(l); rows[(r["cell"], int(rnd))][arm] = r
    for (cell, rnd), v in sorted(rows.items()):
        o, n = v.get("off"), v.get("on")
        if o and n:
            print(f"{cell} r{rnd}: ptoks {o.get('ptoks')} toks {o['toks']}/{n['toks']} sha {'EQUAL' if o['sha']==n['sha'] else 'DIFF'} {o['sha']}/{n['sha']} dec_tps {o['dec_tps']} -> {n['dec_tps']} ({n['dec_tps']/o['dec_tps']:.3f}x) ttft {o['ttft']} -> {n['ttft']}")
