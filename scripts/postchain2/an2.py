#!/usr/bin/env python3
"""postchain2, the primary measure (fixed before the run; postchain's r2_an.py with two changes): per arm and round,
  (arm - base) over MID1 positions [100, 640)  minus  (arm - base) over PRE1-5 (all positions >= 100; PRE0 left out because with the
  800-token REF an n20 server has a real chain in REF at 828, so its PRE0 is a request right after a chain),
and, against the no-event arm NULL (env NULLARM, default `nc`: copy knob on, probe armed, nothing fires) of the SAME round: arm measure - null measure.
steps matched by (request, position, n) with the same drafts and accepted count. Secondary: the same with POST0 instead of MID1, and with POST1-9.
A round is void when one of the arm's or base's PRE / POST request means is > 1 ms above the median base PRE request (an outside load).
Positive: > 0 in at least 4 of 5 rounds and the mean > 2 SE. usage: r2_an.py DIR"""
import re, glob, gzip, sys, statistics as st, math
T = re.compile(r"^spec: token \d+ pos (\d+) drafted (\d+) accepted (\d+) emits \d+ drafts \[([^\]]*)\]")
G = re.compile(r"^spec: mtp graph step ([\d.]+) ms gap [\d.]+ ms n (\d+) len (\d+)")
C = re.compile(r"^spec: copy token \d+ pos (\d+) drafted (\d+) accepted (\d+)")
def opn(f): return gzip.open(f, "rt", errors="replace") if f.endswith(".gz") else open(f, errors="replace")
def parse(fn):
    reqs, last, pend, chains = [{}], -1, None, []
    for l in opn(fn):
        t = T.match(l)
        if t: pend = (t.group(3), t.group(4)); continue
        c = C.match(l)
        if c: chains.append((len(reqs) - 1, int(c.group(1)), int(c.group(2)), int(c.group(3)))); continue
        g = G.match(l)
        if g:
            pos = int(g.group(3))
            if pos < last - 50: reqs.append({})
            last = pos
            if pend: reqs[-1][(pos, int(g.group(2)))] = (pend, float(g.group(1)))
            pend = None
    return reqs, chains
runs = {}
for fn in sorted(glob.glob(f"{sys.argv[1]}/serve-*-r*.log*")):
    m = re.search(r"serve-(.+)-r(\d+)\.log(?:\.gz)?$", fn); runs[(m.group(1), int(m.group(2)))] = parse(fn)
PRE, MID0, MID1, POST0, POSTR = range(4, 9), 9, 10, 11, range(12, 21)
import os
NULL = os.environ.get('NULLARM', 'nc')
def d(A, B, idx, lo=100, hi=10**6):
    x = [A[i][k][1] - B[i][k][1] for i in idx for k in A[i] if k in B[i] and A[i][k][0] == B[i][k][0] and lo <= k[0] < hi]
    return st.mean(x) if x else float("nan")
rmean = lambda r: st.mean(v[1] for k, v in r.items() if k[0] >= 100)
normal = st.median(rmean(runs[k][0][i]) for k in runs if k[0] == "base" for i in range(3, 9))
def void(R): return any(rmean(R[i]) > normal + 1 for i in list(range(3, 9)) + [POST0] + list(POSTR))
for (arm, rnd), (R, ch) in sorted(runs.items()):
    if ch: print(f"{arm} r{rnd} chains (request, pos, drafted, accepted): {ch}")
res = {}
for (arm, rnd), (A, _) in sorted(runs.items()):
    if arm == "base" or ("base", rnd) not in runs: continue
    B = runs[("base", rnd)][0]
    if len(A) < 21 or len(B) < 21: print(f"{arm} r{rnd}: {len(A)} / {len(B)} requests, skipped"); continue
    pre = d(A, B, PRE)
    res.setdefault(arm, []).append((rnd, d(A, B, [MID1], 100, 640) - pre, d(A, B, [POST0]) - pre, d(A, B, POSTR) - pre, void(A) or void(B)))
def summ(xs):
    xs = [x for x in xs if not math.isnan(x)]
    if len(xs) < 2: return "n/a"
    m, se = st.mean(xs), st.stdev(xs) / math.sqrt(len(xs))
    pos = sum(x > 0 for x in xs)
    return f"{m:+.3f} (SE {se:.3f}, {pos}/{len(xs)} rounds > 0{', POSITIVE' if pos >= len(xs) - 1 and pos >= 4 and m > 2 * se else ''})"
print(f"normal base PRE request {normal:.3f} ms\narm minus base, minus the same over PRE (ms per graph step)")
for arm, v in res.items():
    for rnd, a, b, c, vo in v:
        print(f"  {arm} r{rnd}: MID1 {a:+.3f}  POST0 {b:+.3f}  POST1-9 {c:+.3f}{'  VOID' if vo else ''}")
    ok = [x for x in v if not x[4]]
    print(f"{arm}: MID1 {summ([x[1] for x in ok])} | POST0 {summ([x[2] for x in ok])} | POST1-9 {summ([x[3] for x in ok])}")

nul = {rnd: (a, b, c) for rnd, a, b, c, vo in res.get(NULL, []) if not vo}
print(f"\narm minus the null arm `{NULL}` of the same round (each already minus base and minus PRE): MID1 | POST0 | POST1-9")
for arm, v in res.items():
    if arm == NULL: continue
    rows = [(rnd, a - nul[rnd][0], b - nul[rnd][1], c - nul[rnd][2]) for rnd, a, b, c, vo in v if not vo and rnd in nul]
    for rnd, a, b, c in rows:
        print(f"  {arm} r{rnd}: MID1 {a:+.3f}  POST0 {b:+.3f}  POST1-9 {c:+.3f}")
    print(f"{arm} - {NULL}: MID1 {summ([x[1] for x in rows])} | POST0 {summ([x[2] for x in rows])} | POST1-9 {summ([x[3] for x in rows])}")
