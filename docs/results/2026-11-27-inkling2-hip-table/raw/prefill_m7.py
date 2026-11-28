#!/usr/bin/env python3
"""Lane ink-hip-specflip main7 check: prefill tok/s ratio arm / ref per prompt name, per rep pair (`<arm>@<rep>.err`, `prefill[name]: N
tokens, M ms`), geometric mean and a 95 % t-interval over the reps.  prefill_m7.py <dir> <arm> <ref> <name> [<name> ...]"""
import math, re, sys
from pathlib import Path
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}
d, arm, ref, names = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4:]
pf = re.compile(r"^prefill\[([^\]]*)\]: (\d+) tokens, (\d+) ms")
def read(p):
    out = {}
    for l in p.read_text(errors="replace").splitlines():
        m = pf.match(l)
        if m:
            out[m.group(1)] = int(m.group(2)) / int(m.group(3)) * 1e3
    return out
data = {}
for e in d.glob("*@*.err"):
    a, r = e.stem.split("@")
    data[(a, r)] = read(e)
reps = sorted({r for _, r in data})
for n in names:
    lr, abs_ = [], []
    for r in reps:
        x, y = data.get((ref, r), {}).get(n), data.get((arm, r), {}).get(n)
        if x and y:
            lr.append(math.log(y / x)); abs_.append(y)
    if not lr:
        print(f"prefill {n}: no data"); continue
    k = len(lr); m = sum(lr) / k
    ci = ""
    if k > 1:
        sd = math.sqrt(sum((v - m) ** 2 for v in lr) / (k - 1)); h = T975.get(k - 1, 2.0) * sd / math.sqrt(k)
        ci = f" [{math.exp(m - h):.3f}-{math.exp(m + h):.3f}]"
    print(f"prefill {n}: {arm} / {ref} **{math.exp(m):.3f}**{ci} n {k}; {arm} tok/s median {sorted(abs_)[len(abs_) // 2]:.0f}")
