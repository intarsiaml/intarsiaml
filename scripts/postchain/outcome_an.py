#!/usr/bin/env python3
"""postchain: split the post-chain graph-step offset by the step's own outcome (accepted 0 / 1 -> rollback from save row 0 / 1; 2 -> no rollback).
Reads serve-<arm>-r<N>.log[.gz] (INTARSIA_SPEC_TRACE=1). For each request (after --warm), steps are binned by (position bin, accepted);
arm minus base of the same round, request, bin and outcome. Steps are classed 'seen' when a chain ran earlier in the process.
usage: outcome_an.py DIR [--arm n20] [--base base] [--bin 100]"""
import re, sys, glob, gzip, argparse, collections, statistics as st
ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--arm", default="n20"); ap.add_argument("--base", default="base")
ap.add_argument("--bin", type=int, default=100); ap.add_argument("--warm", type=int, default=2); ap.add_argument("--minn", type=int, default=4)
ap.add_argument("--lo", type=int, default=100); ap.add_argument("--hi", type=int, default=900)
a = ap.parse_args()
T = re.compile(r"^spec: token \d+ pos (\d+) drafted (\d+) accepted (\d+)")
G = re.compile(r"^spec: mtp graph step ([\d.]+) ms gap ([\d.]+) ms n (\d+) len (\d+)")
C = re.compile(r"^spec: copy token \d+ pos (\d+) drafted (\d+) accepted (\d+)")
def opn(f): return gzip.open(f, "rt", errors="replace") if f.endswith(".gz") else open(f, errors="replace")
def parse(fn):
    reqs, cur, last, pend, chains = [], None, -1, None, 0
    first = None
    for l in opn(fn):
        t = T.match(l)
        if t: pend = int(t.group(3)); continue
        c = C.match(l)
        if c:
            chains += 1
            if first is None: first = int(c.group(3))
            pend = None; continue
        g = G.match(l)
        if g:
            pos = int(g.group(4))
            if cur is None or pos < last - 50:
                cur = []; reqs.append(cur)
            last = pos
            if pend is not None:
                cur.append((pos, float(g.group(1)), pend, chains > 0))
            pend = None
    return reqs, first, chains
runs = {}
for fn in sorted(glob.glob(f"{a.dir}/serve-*-r*.log*")):
    m = re.search(r"serve-(.+)-r(\d+)\.log(?:\.gz)?$", fn)
    if m.group(1) in (a.arm, a.base):
        r, first, n = parse(fn)
        runs[(m.group(1), int(m.group(2)))] = r[a.warm:]
        if m.group(1) == a.arm: print(f"{fn}: chains {n}, first chain accepted {first}")
def bins(steps):
    d = collections.defaultdict(list)
    for pos, ms, acc, seen in steps:
        d[(pos // a.bin, min(acc, 2))].append((ms, seen))
    return d
res = collections.defaultdict(list)
for (arm, rnd), reqs in runs.items():
    if arm != a.arm or (a.base, rnd) not in runs: continue
    for ra, rb in zip(reqs, runs[(a.base, rnd)]):
        A, B = bins(ra), bins(rb)
        for k in A:
            if k not in B or k[0] * a.bin < a.lo or k[0] * a.bin >= a.hi: continue
            if len(A[k]) < a.minn or len(B[k]) < a.minn: continue
            seen = all(s for _, s in A[k]); notseen = not any(s for _, s in A[k])
            if not (seen or notseen): continue
            d = st.mean(m for m, _ in A[k]) - st.mean(m for m, _ in B[k])
            res[(rnd, seen, k[1])].append(d)
print(f"{a.arm} minus {a.base}, ms per graph step, by the step's own accepted count (0,1: rollback from save row 0/1; 2: no rollback)")
for seen in (False, True):
    for acc in (0, 1, 2):
        per = [st.mean(res[(r, seen, acc)]) for r in sorted({k[0] for k in res}) if res[(r, seen, acc)]]
        allv = [v for r in sorted({k[0] for k in res}) for v in res[(r, seen, acc)]]
        if not allv: continue
        print(f"  {'seen' if seen else 'not seen':8} acc {acc}: mean {st.mean(allv):+.3f} bins {len(allv):4}  per round " + " ".join(f"{p:+.3f}" for p in per))
