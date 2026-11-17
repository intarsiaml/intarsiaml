#!/usr/bin/env python3
"""mcdiag: graph-step time per served request from `serve --max-batch 1` logs with INTARSIA_SPEC_TRACE=1. Requests are told apart by the position
(`len`) dropping; the first --warm requests (the warm-up cell) are skipped. For each request: the position of its chain pass (if any), the mean
step ms over positions [LO, MID) (before the chain) and [MID2, HI) (after it), the same positions for requests without a chain. Positions are
the same across arms (same prompts, greedy), so the columns compare directly.
usage: serve_steps.py [--warm 2] [--lo 100] [--mid 320] [--mid2 340] [--hi 700] LOG..."""
import re, sys, argparse

ap = argparse.ArgumentParser()
ap.add_argument("logs", nargs="+")
ap.add_argument("--warm", type=int, default=2)
ap.add_argument("--lo", type=int, default=100)
ap.add_argument("--mid", type=int, default=320)
ap.add_argument("--mid2", type=int, default=340)
ap.add_argument("--hi", type=int, default=700)
a = ap.parse_args()
G = re.compile(r"^spec: mtp graph step ([\d.]+) ms gap ([\d.]+) ms n (\d+) len (\d+) captures (\d+)")
C = re.compile(r"^spec: copy token \d+ pos (\d+)")
P = re.compile(r"^spec: copy pass (\d+) rows ([\d.]+) ms")
mean = lambda x: sum(x) / len(x) if x else float("nan")
for fn in a.logs:
    reqs = [[]]
    chains = [[]]
    cp = [[]]
    last = -1
    for l in open(fn, errors="replace"):
        m = G.match(l)
        if m:
            pos = int(m.group(4))
            if pos < last - 50:
                reqs.append([]); chains.append([]); cp.append([])
            last = pos
            reqs[-1].append((pos, float(m.group(1))))
            continue
        m = C.match(l)
        if m:
            chains[-1].append(int(m.group(1)))
        m = P.match(l)
        if m:
            cp[-1].append(float(m.group(2)))
    print(fn, f"requests={len(reqs)}")
    for i, r in enumerate(reqs):
        if i < a.warm or not r:
            continue
        pre = [ms for p, ms in r if a.lo <= p < a.mid]
        post = [ms for p, ms in r if a.mid2 <= p < a.hi]
        print(f"  req {i - a.warm}: steps={len(r)} chains at {chains[i]} chain pass ms {cp[i]} | pre[{a.lo},{a.mid}) {mean(pre):.3f} ({len(pre)}) post[{a.mid2},{a.hi}) {mean(post):.3f} ({len(post)}) diff {mean(post) - mean(pre):+.3f}")
