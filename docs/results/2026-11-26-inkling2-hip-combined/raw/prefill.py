#!/usr/bin/env python3
"""Prefill tok/s per prompt from `generate --prompt-ids` stderr (`prefill[<i>]: N tokens, X ms`), files <dir>/<arm>@<rep>.err.
Ratio arm / ref per rep (the same rep = the same card and position in the ABBA order), geometric mean over reps, 95 % CI: t-interval
on the log of the per-rep ratios. usage: prefill.py DIR REF ARM "4k=14,16k=15"   (prompt index in the file)"""
import math, re, sys
from pathlib import Path

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}
d, ref, arm, spec = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
pf = re.compile(r"^prefill\[(\d+)\]: (\d+) tokens, (\d+) ms")
data = {}
for p in sorted(d.glob("*@*.err")):
    a, r = p.stem.split("@")
    data[(a, r)] = {int(m.group(1)): (int(m.group(2)), int(m.group(3))) for m in map(pf.match, p.read_text(errors="replace").splitlines()) if m}
reps = sorted({r for a, r in data if a == arm and (ref, r) in data})
med = lambda xs: sorted(xs)[len(xs) // 2] if len(xs) % 2 else sum(sorted(xs)[len(xs) // 2 - 1:len(xs) // 2 + 1]) / 2
print(f"| prefill tok/s | {ref} median [range] | {arm} median [range] | {arm} / {ref}, 95 % CI |")
print("|---|---:|---:|---|")
for c in spec.split(","):
    name, i = c.split("="); i = int(i)
    tps = lambda a, r: data[(a, r)][i][0] / data[(a, r)][i][1] * 1000
    rs = [tps(arm, r) / tps(ref, r) for r in reps]
    lg = [math.log(x) for x in rs]; mu = sum(lg) / len(lg)
    sd = math.sqrt(sum((x - mu) ** 2 for x in lg) / (len(lg) - 1)) if len(lg) > 1 else 0
    h = T975.get(len(lg) - 1, 2.0) * sd / math.sqrt(len(lg))
    o = [tps(ref, r) for r in reps]; n = [tps(arm, r) for r in reps]
    ntok = data[(arm, reps[0])][i][0]
    print(f"| {name} ({ntok} tokens) | {med(o):.0f} [{min(o):.0f}-{max(o):.0f}] | {med(n):.0f} [{min(n):.0f}-{max(n):.0f}] | **{math.exp(mu):.3f}** [{math.exp(mu - h):.3f}-{math.exp(mu + h):.3f}] |")
print(f"reps: {reps}")
