#!/usr/bin/env python3
"""mcdiag: analysis of scripts/mcdiag/serve2.sh logs (serve-<arm>-r<round>.log in DIR). Requests are told apart by the position (`len`) dropping:
2 warm-ups, PRE pre-event requests, the event request (800 tokens), POST post-event requests. Prints, per arm, the mean graph-step ms over positions
[LO, HI) per request, and for each non-base arm the matched difference to base (same round, same request: same prompt, same greedy tokens, so
the same positions), averaged over PRE requests, over POST requests, and their difference (the carry-over of the event, in ms per step).
usage: serve2_an.py DIR [--base base] [--pre 6] [--post 6] [--lo 100] [--hi 370]"""
import re, sys, glob, argparse, collections, statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--base", default="base")
ap.add_argument("--pre", type=int, default=6)
ap.add_argument("--post", type=int, default=6)
ap.add_argument("--lo", type=int, default=100)
ap.add_argument("--hi", type=int, default=370)
a = ap.parse_args()
G = re.compile(r"^spec: mtp graph step ([\d.]+) ms gap ([\d.]+) ms n (\d+) len (\d+)")
data = {}  # (arm, round) -> list of per-request mean ms over [lo, hi)
for fn in sorted(glob.glob(f"{a.dir}/serve-*-r*.log")):
    m = re.search(r"serve-(.+)-r(\d+)\.log$", fn)
    arm, rnd = m.group(1), int(m.group(2))
    reqs, last = [[]], -1
    for l in open(fn, errors="replace"):
        g = G.match(l)
        if g:
            pos = int(g.group(4))
            if pos < last - 50:
                reqs.append([])
            last = pos
            reqs[-1].append((pos, float(g.group(1))))
    per = []
    for r in reqs:
        x = [ms for p, ms in r if a.lo <= p < a.hi]
        per.append(sum(x) / len(x) if x else float("nan"))
    data[(arm, rnd)] = per
phases = lambda per: {"pre": per[2:2 + a.pre], "post": per[3 + a.pre:3 + a.pre + a.post]}
arms = sorted({k[0] for k in data})
for arm in arms:
    for rnd in sorted(r for (aa, r) in data if aa == arm):
        per = data[(arm, rnd)]
        ph = phases(per)
        print(f"{arm:8s} r{rnd}: requests={len(per)} pre " + " ".join(f"{x:.2f}" for x in ph["pre"]) + " | post " + " ".join(f"{x:.2f}" for x in ph["post"]))
print()
for arm in arms:
    if arm == a.base:
        continue
    pre_d, post_d, per_round = [], [], []
    for rnd in sorted(r for (aa, r) in data if aa == arm):
        if (a.base, rnd) not in data:
            continue
        p, b = phases(data[(arm, rnd)]), phases(data[(a.base, rnd)])
        dp = [x - y for x, y in zip(p["pre"], b["pre"])]
        dq = [x - y for x, y in zip(p["post"], b["post"])]
        pre_d += dp; post_d += dq
        per_round.append((rnd, st.mean(dp), st.mean(dq), st.mean(dq) - st.mean(dp)))
    f = lambda x: f"{st.mean(x):+.3f} (sd {st.pstdev(x):.3f}, n {len(x)})"
    print(f"{arm} - {a.base}: pre {f(pre_d)}  post {f(post_d)}")
    print("   per round (pre, post, post-pre):", "; ".join(f"r{r}: {x:+.3f} {y:+.3f} {z:+.3f}" for r, x, y, z in per_round))
    print(f"   carry-over (post - pre), mean of rounds: {st.mean(z for *_, z in per_round):+.3f} ms per step")
