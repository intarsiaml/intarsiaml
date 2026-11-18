#!/usr/bin/env python3
"""postchain: analysis of scripts/postchain/serve.sh logs (serve-<arm>-r<round>.log[.gz] in DIR). Requests are told apart by the position dropping:
2 warm-ups, REF (500 tokens), PRE x6 (300), MID x2 (800; the probe fires in the first), POST x10 (300).
Per arm and round, matched to base of the same round and request (same prompt, same greedy tokens, so the same positions):
  pre   = mean over PRE requests of (arm - base) graph-step ms over positions [LO, HI)
  post  = the same over POST requests
  carry = post - pre (the event's carry-over into later requests, ms per step)
  mid   = in the first MID request: (arm - base) over positions after the event minus before it ([AFT_LO, AFT_HI) minus [BEF_LO, BEF_HI))
A round of an arm is void when one of its PRE or POST request means is more than 1 ms above the median PRE request mean of all base servers
(an outside load, as cpupin's L7 r3). Per round values are printed; the mean over valid rounds with its standard error over rounds.
usage: an.py DIR [--base base]"""
import re, glob, gzip, argparse, statistics as st, math, collections
ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--base", default="base")
ap.add_argument("--lo", type=int, default=100); ap.add_argument("--hi", type=int, default=370)
ap.add_argument("--bef", default="150,520"); ap.add_argument("--aft", default="560,870")
ap.add_argument("--ref", type=int, default=1); ap.add_argument("--pre", type=int, default=6); ap.add_argument("--mid", type=int, default=2); ap.add_argument("--post", type=int, default=10)
a = ap.parse_args()
G = re.compile(r"^spec: mtp graph step ([\d.]+) ms gap [\d.]+ ms n (\d+) len (\d+)")
C = re.compile(r"^spec: copy token \d+ pos (\d+) drafted (\d+) accepted (\d+)")
P = re.compile(r"^spec: probe (.*)")
def opn(f): return gzip.open(f, "rt", errors="replace") if f.endswith(".gz") else open(f, errors="replace")
def parse(fn):
    reqs, last, chains, probes = [[]], -1, [], []
    for l in opn(fn):
        g = G.match(l)
        if g:
            pos = int(g.group(3))
            if pos < last - 50: reqs.append([])
            last = pos; reqs[-1].append((pos, float(g.group(1)))); continue
        c = C.match(l)
        if c: chains.append((len(reqs) - 1, int(c.group(1)), int(c.group(3)))); continue
        p = P.match(l)
        if p: probes.append((len(reqs) - 1, p.group(1).strip()))
    return reqs, chains, probes
def mean_in(r, lo, hi):
    x = [ms for p, ms in r if lo <= p < hi]; return sum(x) / len(x) if x else float("nan")
runs = {}
for fn in sorted(glob.glob(f"{a.dir}/serve-*-r*.log*")):
    m = re.search(r"serve-(.+)-r(\d+)\.log(?:\.gz)?$", fn); runs[(m.group(1), int(m.group(2)))] = parse(fn)
f0 = 2 + a.ref
I_PRE = list(range(f0, f0 + a.pre)); I_MID = f0 + a.pre; I_POST = list(range(f0 + a.pre + a.mid, f0 + a.pre + a.mid + a.post))
base_pre = [mean_in(runs[k][0][i], a.lo, a.hi) for k in runs if k[0] == a.base for i in I_PRE if i < len(runs[k][0])]
normal = st.median(base_pre)
bl, bh = map(int, a.bef.split(",")); al, ah = map(int, a.aft.split(","))
def void(reqs):
    return any(mean_in(reqs[i], a.lo, a.hi) > normal + 1.0 for i in I_PRE + I_POST if i < len(reqs))
print(f"normal (median base PRE request) {normal:.3f} ms; requests per server: " + " ".join(f"{k[0]}-r{k[1]}:{len(v[0])}" for k, v in sorted(runs.items())))
for (arm, rnd), (reqs, chains, probes) in sorted(runs.items()):
    if chains or probes:
        print(f"  {arm} r{rnd}: chains (request, pos, accepted) {chains[:6]}{' ...' if len(chains) > 6 else ''}; probes {probes}")
res = collections.defaultdict(dict)
for (arm, rnd), (reqs, _, _) in sorted(runs.items()):
    if arm == a.base or (a.base, rnd) not in runs: continue
    breqs = runs[(a.base, rnd)][0]
    if len(reqs) != len(breqs) or len(reqs) < I_POST[-1] + 1:
        print(f"  {arm} r{rnd}: {len(reqs)} requests vs base {len(breqs)}: skipped"); continue
    d = lambda i, lo, hi: mean_in(reqs[i], lo, hi) - mean_in(breqs[i], lo, hi)
    pre = st.mean(d(i, a.lo, a.hi) for i in I_PRE); post = st.mean(d(i, a.lo, a.hi) for i in I_POST)
    mid = d(I_MID, al, ah) - d(I_MID, bl, bh)
    v = void(reqs) or void(breqs)
    res[arm][rnd] = (pre, post, post - pre, mid, v)
def ms(xs):
    xs = [x for x in xs if not math.isnan(x)]
    if not xs: return "n/a"
    se = st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else float("nan")
    return f"{st.mean(xs):+.3f} (SE {se:.3f}, {len(xs)} rounds)"
print(f"\narm minus {a.base}, ms per graph step; carry = post - pre (later requests); mid = after minus before the event inside the event request")
for arm, rr in res.items():
    print(f"{arm}:")
    for rnd, (pre, post, carry, mid, v) in sorted(rr.items()):
        print(f"   r{rnd}: pre {pre:+.3f} post {post:+.3f} carry {carry:+.3f} | mid {mid:+.3f}{'  VOID (outside load)' if v else ''}")
    ok = [x for x in rr.values() if not x[4]]
    print(f"   carry {ms([x[2] for x in ok])}   mid {ms([x[3] for x in ok])}")
