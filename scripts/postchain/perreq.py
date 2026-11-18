#!/usr/bin/env python3
"""postchain: per request of a serve trace log: steps, mean graph-step ms, mean ms by catch-up rows n (1/2/3), accept rate, first position.
usage: perreq.py LOG[.gz]"""
import re, sys, gzip, statistics as st, collections
G = re.compile(r"^spec: mtp graph step ([\d.]+) ms gap ([\d.]+) ms n (\d+) len (\d+)")
T = re.compile(r"^spec: token \d+ pos (\d+) drafted (\d+) accepted (\d+)")
C = re.compile(r"^spec: copy token")
fn = sys.argv[1]
f = gzip.open(fn, "rt", errors="replace") if fn.endswith(".gz") else open(fn, errors="replace")
reqs, cur, last, acc = [], None, -1, None
for l in f:
    t = T.match(l)
    if t: acc = int(t.group(3)); continue
    if C.match(l):
        if cur is not None: cur["chain"] += 1
        continue
    g = G.match(l)
    if g:
        pos = int(g.group(4))
        if cur is None or pos < last - 50:
            cur = {"ms": [], "n": collections.defaultdict(list), "acc": [], "p0": pos, "chain": 0}; reqs.append(cur)
        last = pos
        ms = float(g.group(1)); n = int(g.group(3))
        if pos >= cur["p0"] + 20:
            cur["ms"].append(ms); cur["n"][n].append(ms)
            if acc is not None: cur["acc"].append(acc)
        acc = None
for i, r in enumerate(reqs):
    if not r["ms"]: continue
    byn = " ".join(f"n{n}:{st.median(v):.2f}({len(v)})" for n, v in sorted(r["n"].items()))
    print(f"req {i:2} p0 {r['p0']:5} steps {len(r['ms']):4} mean {st.mean(r['ms']):.3f} med {st.median(r['ms']):.3f} acc {st.mean(r['acc']) if r['acc'] else 0:.2f} chains {r['chain']} | {byn}")
