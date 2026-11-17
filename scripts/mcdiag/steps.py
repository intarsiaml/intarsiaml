#!/usr/bin/env python3
"""mcdiag: per-step table of an INTARSIA_SPEC_TRACE=1 `generate` log. For each file: the captured-graph steps (ms, host gap, queued head rows n, position),
split at the first chain pass (a `spec: copy token` line) or, with --at POS, at that position (for runs without a chain: the baseline of a probe run).
Prints mean step ms before / after the split (the first --skip steps dropped: the capture), mean ms per emitted token over the same ranges, and
the mean step by windows of --win steps. usage: steps.py [--at POS] [--skip N] [--win N] FILE..."""
import re, sys, argparse

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--at", type=int)
ap.add_argument("--skip", type=int, default=8)
ap.add_argument("--win", type=int, default=0)
a = ap.parse_args()
G = re.compile(r"^spec: mtp graph step ([\d.]+) ms gap ([\d.]+) ms n (\d+) len (\d+) captures (\d+)")
C = re.compile(r"^spec: copy token \d+ pos (\d+)")
for fn in a.files:
    steps, chains = [], []
    for l in open(fn, errors="replace"):
        m = G.match(l)
        if m:
            steps.append((float(m.group(1)), float(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))))
            continue
        m = C.match(l)
        if m:
            chains.append((int(m.group(1)), len(steps)))
    if not steps:
        print(fn, "no graph steps"); continue
    split = chains[0][1] if chains and a.at is None else None
    if a.at is not None:
        split = next((i for i, s in enumerate(steps) if s[3] >= a.at), len(steps))
    ss = steps[a.skip:]
    sp = max(split - a.skip, 0) if split is not None else len(ss)
    def mean(x): return sum(x) / len(x) if x else float("nan")
    def pertok(x):
        # ms per emitted token: step ms over the positions the steps advanced (len of the next step minus this one's, chain passes included in the gap)
        return float("nan")
    pre, post = ss[:sp], ss[sp:]
    print(f"{fn}: steps={len(steps)} chains={len(chains)} at_pos={[c[0] for c in chains][:6]} split_step={split} captures={steps[-1][4]} "
          f"pre={mean([s[0] for s in pre]):.3f} ms (n={len(pre)}) post={mean([s[0] for s in post]):.3f} ms (n={len(post)}) "
          f"diff={mean([s[0] for s in post]) - mean([s[0] for s in pre]):+.3f} | gap pre={mean([s[1] for s in pre]):.3f} post={mean([s[1] for s in post]):.3f} "
          f"| n(rows) mean pre={mean([s[2] for s in pre]):.2f} post={mean([s[2] for s in post]):.2f}")
    if a.win:
        print("  windows:", " ".join(f"{mean([s[0] for s in ss[i:i + a.win]]):.2f}" for i in range(0, len(ss), a.win)))
