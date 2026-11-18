#!/usr/bin/env python3
"""mcdiag: analysis of served runs of the real A/B arms (scripts/mcdiag/ab.sh, CELLS=code-gen, INTARSIA_SPEC_TRACE=1; arms base, n256 = knob on and no
chain fires, n20 = real chains). For every (round, request, position bin) the arm's mean graph-step ms minus the same bin of the same request in the base
server of the round (same prompt, greedy: same tokens and positions; a bin is compared only when both have >= MINN steps). Bins of an arm are split by
`chain seen`: whether a chain pass has run earlier in that server process (a bin that holds the first chain counts as not seen). The persistent
slowdown of the issue would show as n20 `seen` bins well above n20 `not seen` bins and above n256.
usage: real_an.py DIR [--bin 50] [--lo 100] [--hi 800] [--warm 2] [--base base]"""
import re, sys, glob, argparse, collections, statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--bin", type=int, default=50)
ap.add_argument("--lo", type=int, default=100)
ap.add_argument("--hi", type=int, default=800)
ap.add_argument("--warm", type=int, default=2)
ap.add_argument("--base", default="base")
ap.add_argument("--minn", type=int, default=10)
a = ap.parse_args()
G = re.compile(r"^spec: mtp graph step ([\d.]+) ms gap ([\d.]+) ms n (\d+) len (\d+)")
C = re.compile(r"^spec: copy token \d+ pos (\d+)")


def parse(fn):
    reqs, last, chains_before = [], -1, 0
    cur = None
    seen_total = 0
    for l in open(fn, errors="replace"):
        g = G.match(l)
        if g:
            pos = int(g.group(4))
            if cur is None or pos < last - 50:
                cur = {"steps": [], "chain_pos": [], "seen_before": seen_total}
                reqs.append(cur)
            last = pos
            cur["steps"].append((pos, float(g.group(1)), seen_total + len(cur["chain_pos"])))
            continue
        c = C.match(l)
        if c and cur is not None:
            cur["chain_pos"].append(int(c.group(1)))
        if c:
            pass
    # chain count before each step: recompute with chain positions inside a request
    out = []
    total = 0
    for r in reqs:
        steps = []
        cp = sorted(r["chain_pos"])
        for pos, ms, _ in r["steps"]:
            seen = total + sum(1 for p in cp if p < pos) > 0
            steps.append((pos, ms, seen))
        out.append({"steps": steps, "chains": cp})
        total += len(cp)
    return out


runs = {}
for fn in sorted(glob.glob(f"{a.dir}/serve-*-r*.log")):
    m = re.search(r"serve-(.+)-r(\d+)\.log$", fn)
    runs[(m.group(1), int(m.group(2)))] = parse(fn)[a.warm:]
arms = sorted({k[0] for k in runs} - {a.base})
binmean = lambda steps: {b: (sum(x) / len(x), len(x), seen[-1]) for b, (x, seen) in
                         {b: ([s[1] for s in steps if s[0] // a.bin == b], [s[2] for s in steps if s[0] // a.bin == b]) for b in {s[0] // a.bin for s in steps}}.items()}
print(f"bins of {a.bin} positions in [{a.lo},{a.hi}); difference to {a.base} of the same round, request and bin (ms per step)")
for arm in arms:
    d = {True: [], False: []}
    nchain = 0
    per_round = collections.defaultdict(lambda: {True: [], False: []})
    for (aa, rnd), reqs in sorted(runs.items()):
        if aa != arm or (a.base, rnd) not in runs:
            continue
        for ra, rb in zip(reqs, runs[(a.base, rnd)]):
            nchain += len(ra["chains"])
            A, B = binmean(ra["steps"]), binmean(rb["steps"])
            for b in A:
                if b not in B or b * a.bin < a.lo or b * a.bin >= a.hi or A[b][1] < a.minn or B[b][1] < a.minn:
                    continue
                d[A[b][2]].append(A[b][0] - B[b][0])
                per_round[rnd][A[b][2]].append(A[b][0] - B[b][0])
    f = lambda x: f"{st.mean(x):+.3f} (sd {st.pstdev(x):.3f}, n {len(x)})" if x else "n/a"
    print(f"{arm:6s}: chains {nchain}; chain not yet seen: {f(d[False])}; chain seen: {f(d[True])}")
    print("        per round (not seen | seen):", "; ".join(f"r{r}: {st.mean(v[False]) if v[False] else float('nan'):+.3f} | {st.mean(v[True]) if v[True] else float('nan'):+.3f}" for r, v in sorted(per_round.items())))
