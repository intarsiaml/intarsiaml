#!/usr/bin/env python3
"""postchain: the event request's own steps, arm vs base matched exactly by (position, n) with the same drafts and accepted count:
mean (arm - base) ms before the event position and after it, per round. Also the same for the PRE and POST requests (pooled), so the
in-request offset and the carry-over use one estimator. usage: midmatch.py DIR [--ref 1] [--pre 6] [--mid 2] [--post 10] [--event 525]"""
import re, glob, gzip, argparse, statistics as st, math, collections
ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--base", default="base")
ap.add_argument("--ref", type=int, default=1); ap.add_argument("--pre", type=int, default=6); ap.add_argument("--mid", type=int, default=2); ap.add_argument("--post", type=int, default=10)
ap.add_argument("--event", type=int, default=525); ap.add_argument("--lo", type=int, default=100)
a = ap.parse_args()
T = re.compile(r"^spec: token \d+ pos (\d+) drafted (\d+) accepted (\d+) emits \d+ drafts \[([^\]]*)\]")
G = re.compile(r"^spec: mtp graph step ([\d.]+) ms gap [\d.]+ ms n (\d+) len (\d+)")
def opn(f): return gzip.open(f, "rt", errors="replace") if f.endswith(".gz") else open(f, errors="replace")
def parse(fn):
    reqs, last, pend = [{}], -1, None
    for l in opn(fn):
        t = T.match(l)
        if t: pend = (t.group(3), t.group(4)); continue
        g = G.match(l)
        if g:
            pos = int(g.group(3))
            if pos < last - 50: reqs.append({})
            last = pos
            if pend: reqs[-1][(pos, int(g.group(2)))] = (pend, float(g.group(1)))
            pend = None
    return reqs
runs = {}
for fn in sorted(glob.glob(f"{a.dir}/serve-*-r*.log*")):
    m = re.search(r"serve-(.+)-r(\d+)\.log(?:\.gz)?$", fn); runs[(m.group(1), int(m.group(2)))] = parse(fn)
f0 = 2 + a.ref; I_PRE = range(f0, f0 + a.pre); I_MID = f0 + a.pre; I_POST = range(f0 + a.pre + a.mid, f0 + a.pre + a.mid + a.post)
def diffs(A, B, sel):
    return [A[k][1] - B[k][1] for k in A if k in B and A[k][0] == B[k][0] and sel(k[0])]
out = collections.defaultdict(list)
for (arm, rnd), reqs in sorted(runs.items()):
    if arm == a.base or (a.base, rnd) not in runs: continue
    B = runs[(a.base, rnd)]
    if len(reqs) <= I_POST[-1] or len(B) <= I_POST[-1]: continue
    bef = diffs(reqs[I_MID], B[I_MID], lambda p: a.lo <= p < a.event)
    aft = diffs(reqs[I_MID], B[I_MID], lambda p: p >= a.event + 10)
    pre = [x for i in I_PRE for x in diffs(reqs[i], B[i], lambda p: p >= a.lo)]
    post = [x for i in I_POST for x in diffs(reqs[i], B[i], lambda p: p >= a.lo)]
    mm = lambda x: st.mean(x) if x else float("nan")
    print(f"{arm} r{rnd}: mid before {mm(bef):+.3f} ({len(bef)}) after {mm(aft):+.3f} ({len(aft)}) -> {mm(aft) - mm(bef):+.3f} | pre {mm(pre):+.3f} ({len(pre)}) post {mm(post):+.3f} ({len(post)}) -> {mm(post) - mm(pre):+.3f}")
    out[arm].append((mm(aft) - mm(bef), mm(post) - mm(pre)))
for arm, v in out.items():
    for j, name in ((0, "in-request"), (1, "carry")):
        xs = [x[j] for x in v if not math.isnan(x[j])]
        if len(xs) > 1: print(f"  {arm} {name}: {st.mean(xs):+.3f} (SE {st.stdev(xs) / math.sqrt(len(xs)):.3f}, {len(xs)} rounds)")
