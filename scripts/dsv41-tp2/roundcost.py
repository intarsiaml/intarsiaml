#!/usr/bin/env python3
"""Lane dsv41/decode2: the costs `choose_k` needs, from a serve log made with INTARSIA_DSV41_SPEC_TRACE=1.

Usage: roundcost.py SERVE.log [--old 31.3,38.1,42.9,47.8,52.7 --old-draft 9.0]

Reads `dsv41: spec round k=K conf=.. ok=N [pd=..] draft_ms=D pass_ms=P period_ms=T` lines. For each k (a pass of k+1 rows): rounds, median
and mean pass ms (the figure for INTARSIA_DSV41_SPEC_ROUND_MS), and the round's cost outside its pass (period - pass, the
figure for INTARSIA_DSV41_SPEC_DRAFT_MS; the draft block's own ms is shown beside it). Then what each table would have chosen
on the rounds' own confidences where that is known (a smaller k than the one chosen), and the modelled tokens a ms.
"""
import math
import re
import statistics as st
import sys
from collections import defaultdict

PAT = re.compile(r"spec round k=(\d+) conf=([-\d.,einf]*) ok=(\d+)(?: pd=[\d.,]*)? draft_ms=([\d.]+) pass_ms=([\d.]+) period_ms=([\d.]+)")


def main():
    path = sys.argv[1]
    old = [31.3, 38.1, 42.9, 47.8, 52.7]
    od = 9.0
    a = sys.argv[2:]
    for i, x in enumerate(a):
        if x == "--old":
            old = [float(v) for v in a[i + 1].split(",")]
        if x == "--old-draft":
            od = float(a[i + 1])
    rounds = []
    for l in open(path, errors="replace"):
        m = PAT.search(l)
        if m:
            conf = [float(v) for v in m[2].split(",") if v]
            rounds.append((int(m[1]), conf, int(m[3]), float(m[4]), float(m[5]), float(m[6])))
    # the first round of a request has a period that starts at its own begin; the last is cut by the end of the request
    print(f"{len(rounds)} rounds")
    by = defaultdict(list)
    for k, conf, ok, d, p, t in rounds:
        by[k].append((d, p, t))
    print(f"{'k':>2} {'rows':>4} {'rounds':>6} {'pass med':>9} {'pass mean':>9} {'draft med':>9} {'period med':>10} {'outside pass med':>16}")
    tab = {}
    for k in sorted(by):
        v = by[k]
        pm = st.median(x[1] for x in v)
        tab[k] = pm
        print(f"{k:>2} {k + 1:>4} {len(v):>6} {pm:>9.2f} {st.mean(x[1] for x in v):>9.2f} {st.median(x[0] for x in v):>9.2f} {st.median(x[2] for x in v):>10.2f} {st.median(x[2] - x[1] for x in v):>16.2f}")
    out = [x[2] - x[1] for v in by.values() for x in v if x[2] > 0]
    dm = st.median(out)
    print(f"round cost outside the pass: median {dm:.2f} ms, mean {st.mean(out):.2f} (old table: {od})")
    print("suggested INTARSIA_DSV41_SPEC_ROUND_MS=" + ",".join(f"{tab[k]:.1f}" if k in tab else "-" for k in range(max(tab) + 1)))
    print(f"suggested INTARSIA_DSV41_SPEC_DRAFT_MS={dm:.1f}")
    # acceptance by chosen k: tokens a round and tokens a ms (the quantity choose_k maximises, measured)
    print(f"{'k':>2} {'tok/round':>9} {'ms/round (period)':>18} {'tok/s':>7}")
    for k in sorted(by):
        v = [(ok, t) for kk, c, ok, d, p, t in rounds if kk == k]
        tok = st.mean(ok + 1 for ok, _ in v)
        ms = st.mean(t for _, t in v)
        print(f"{k:>2} {tok:>9.3f} {ms:>18.2f} {1000 * tok / ms:>7.1f}")
    # what the old table chose against what the measured one would, per round, where the conf list covers the choice
    new = [tab.get(k, old[k]) for k in range(len(old))]

    def choose(conf, kmax, tbl, dr):
        best, bv, surv, tok = 0, -1.0, 1.0, 1.0
        for k in range(0, min(kmax, len(conf), len(tbl) - 1) + 1):
            if k > 0:
                surv /= 1 + math.exp(-conf[k - 1])
                tok += surv
            v = tok / (tbl[k] + dr)
            if v > bv:
                best, bv = k, v
        return best

    diff = sum(1 for k, conf, *_ in rounds if conf and choose(conf, len(conf), new, dm) != choose(conf, len(conf), old, od))
    print(f"on the {len(rounds)} rounds' own (chosen-prefix) confidences, the measured table differs from the old one in {diff} rounds")


main()
