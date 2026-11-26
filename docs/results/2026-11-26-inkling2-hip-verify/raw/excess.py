#!/usr/bin/env python3
"""Byte floor per op class and the row-dependent excess, from rowtrace.py's class table (stdin or file).
Floors at 627 GB/s: experts = visits x union(R) x bytes per expert (union of R consecutive real rows, 2026-11-21-inkling2-mtp.md s.1);
dense classes read once a pass (2026-11-25-inkling2-hip.md 4.3 floors; unembed 201024 x 2304 bf16); attention one read of the 1k
context's K / V per visit (0.32 ms), whatever the rows. Excess(R) = (t(R) - t(1)) - (floor(R) - floor(1))."""
import sys
U_NV = [8.0, 12.5, 16.0, 18.9, 21.4, 23.7, 25.7, 27.6]
U_MX = [8.0, 12.6, 16.3, 19.2, 21.7, 23.9, 25.9, 27.7]
BW = 627e9
nv = lambda r: 44 * U_NV[r - 1] * 4.479e6 / BW * 1e3
mx = lambda r: 4 * U_MX[r - 1] * 8.21e6 / BW * 1e3
FLOOR = {"experts nvfp4": nv, "experts mxfp8": mx, "route": lambda r: 0.045, "in-proj": lambda r: 3.05, "unembed": lambda r: 1.477,
         "blend": lambda r: 0.18, "wo / dense gemv": lambda r: 1.08, "mix (conv / attn-res)": lambda r: 0.0, "attention": lambda r: 0.32}
lines = (open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin).read().splitlines()
hdr_i = next(i for i, l in enumerate(lines) if l.split()[:1] == ["rows"])
names = ["experts nvfp4", "experts mxfp8", "route", "in-proj", "unembed", "blend", "wo / dense gemv", "mix (conv / attn-res)", "attention", "argmax", "norm / embed", "copy", "other"]
rows = {}
for l in lines[hdr_i + 1:]:
    p = l.split()
    if not p or not p[0].isdigit():
        break
    r = int(p[0]); rows[r] = dict(zip(["disp", "span", "kernel"] + names, map(float, p[1:])))
R = [r for r in (1, 2, 3, 4, 6, 8) if r in rows]
print("| class (owner) | " + " | ".join(f"{r} row{'s' if r > 1 else ''}" for r in R) + " | floor 1 / 4 / 8 | excess at 2 / 3 / 4 / 6 / 8 rows |")
print("|---|" + "---:|" * len(R) + "---:|---:|")
own = {"experts nvfp4": "this lane", "experts mxfp8": "this lane", "route": "this lane", "in-proj": "ink-hip-p12", "unembed": "ink-hip-p12",
       "wo / dense gemv": "ink-hip-p12", "blend": "ink_mix.hip: ink-hip-batch", "mix (conv / attn-res)": "ink-hip-batch", "attention": "ink-hip-batch"}
tot = {r: 0.0 for r in R}
for c in names[:9]:
    f = FLOOR[c]
    ex = {r: (rows[r][c] - rows[1][c]) - (f(r) - f(1)) for r in R}
    for r in R:
        tot[r] += ex[r]
    print(f"| {c} ({own[c]}) | " + " | ".join(f"{rows[r][c]:.2f}" for r in R) + f" | {f(1):.2f} / {f(4):.2f} / {f(8):.2f} | " + " / ".join(f"{ex[r]:+.2f}" for r in R if r > 1) + " |")
print("| kernel time (all classes) | " + " | ".join(f"{rows[r]['kernel']:.2f}" for r in R) + " | | " + " / ".join(f"{tot[r]:+.2f}" for r in R if r > 1) + " |")
print("| GPU span of the pass | " + " | ".join(f"{rows[r]['span']:.2f}" for r in R) + " | | |")
print("| dispatches | " + " | ".join(f"{rows[r]['disp']:.0f}" for r in R) + " | | |")
