#!/usr/bin/env python3
"""postchain: where in the step does the offset sit? The graph-step trace's split (host before the launch / launch to download done / host after it /
acceptance and rollback, ms) for PRE and POST requests (serve.sh layout), arm minus base of the same round and request, steps matched by (position, n)
with the same drafts; carry = post - pre per segment. usage: split_an.py DIR [--base base]"""
import re, glob, gzip, argparse, statistics as st, math, collections
ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--base", default="base")
ap.add_argument("--ref", type=int, default=1); ap.add_argument("--pre", type=int, default=6); ap.add_argument("--mid", type=int, default=2); ap.add_argument("--post", type=int, default=10); ap.add_argument("--event", type=int, default=525)
a = ap.parse_args()
T = re.compile(r"^spec: token \d+ pos (\d+) drafted (\d+) accepted (\d+) emits \d+ drafts \[([^\]]*)\]")
G = re.compile(r"^spec: mtp graph step ([\d.]+) ms gap [\d.]+ ms n (\d+) len (\d+) captures \d+ at [\d.]+ split ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)(?: launch ([\d.]+))?")
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
            if pend: reqs[-1][(pos, int(g.group(2)))] = (pend, [float(g.group(i) or 'nan') for i in (1, 4, 5, 6, 7, 8)])
            pend = None
    return reqs
runs = {}
for fn in sorted(glob.glob(f"{a.dir}/serve-*-r*.log*")):
    m = re.search(r"serve-(.+)-r(\d+)\.log(?:\.gz)?$", fn); runs[(m.group(1), int(m.group(2)))] = parse(fn)
f0 = 2 + a.ref; I_PRE = range(f0, f0 + a.pre); I_POST = range(f0 + a.pre + a.mid, f0 + a.pre + a.mid + a.post)
names = ["total", "pre-launch", "launch..download", "post-download", "accept", "launch-call"]
agg = collections.defaultdict(list)
for (arm, rnd), reqs in sorted(runs.items()):
    if arm == a.base or (a.base, rnd) not in runs: continue
    B = runs[(a.base, rnd)]
    if len(reqs) <= I_POST[-1] or len(B) <= I_POST[-1]: continue
    def seg(idx):
        d = [[A - Bv for A, Bv in zip(reqs[i][k][1], B[i][k][1])] for i in idx for k in reqs[i] if k in B[i] and reqs[i][k][0] == B[i][k][0] and k[0] >= 100]
        return [st.mean(x[j] for x in d) for j in range(6)] if d else [float("nan")] * 6
    pre, post = seg(I_PRE), seg(I_POST)
    c = [p - q for p, q in zip(post, pre)]
    I_MID = f0 + a.pre
    def mseg(sel):
        d = [[A - Bv for A, Bv in zip(reqs[I_MID][k][1], B[I_MID][k][1])] for k in reqs[I_MID] if k in B[I_MID] and reqs[I_MID][k][0] == B[I_MID][k][0] and sel(k[0])]
        return [st.mean(x[j] for x in d) for j in range(6)] if d else [float("nan")] * 6
    m = [p - q for p, q in zip(mseg(lambda p: p >= a.event + 10), mseg(lambda p: 100 <= p < a.event))]
    I_M1 = I_MID + 1  # MID1 over [100, 640) minus PRE: the primary window of r2_an.py
    d1 = [[A - Bv for A, Bv in zip(reqs[I_M1][k][1], B[I_M1][k][1])] for k in reqs[I_M1] if k in B[I_M1] and reqs[I_M1][k][0] == B[I_M1][k][0] and 100 <= k[0] < 640]
    m1 = [st.mean(x[j] for x in d1) - pre[j] for j in range(6)] if d1 else [float("nan")] * 6
    agg[arm].append((c, m, m1))
    print(f"{arm} r{rnd}: MID1-minus-PRE " + " ".join(f"{n} {v:+.3f}" for n, v in zip(names, m1)))
    print(f"{arm} r{rnd}: carry " + " ".join(f"{n} {v:+.3f}" for n, v in zip(names, c)))
    print(f"{arm} r{rnd}: in-request " + " ".join(f"{n} {v:+.3f}" for n, v in zip(names, m)))
for arm, v in agg.items():
    for i, lab in ((0, "carry"), (1, "in-request"), (2, "MID1-minus-PRE")):
        print(f"  {arm} mean {lab}: " + " ".join(f"{n} {st.mean(x[i][j] for x in v):+.3f}" for j, n in enumerate(names)))
