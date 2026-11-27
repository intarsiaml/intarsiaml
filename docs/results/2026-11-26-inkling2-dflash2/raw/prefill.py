#!/usr/bin/env python3
"""Prefill tok/s ratio dfl / dspark per rep for the 4k (prompt 14) and 16k (prompt 15) prompts, from the `prefill[...]` lines of
logs/abkK/<arm>@<rep>.err; 95 % CI = t over the rep pairs. usage: prefill.py DIR"""
import math, re, sys
from pathlib import Path
d = Path(sys.argv[1]); T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
pf = re.compile(r"^prefill\[([^\]]*)\]: (\d+) tokens, (\d+) ms")
def rd(p): return [(int(m.group(2)), int(m.group(3))) for l in p.read_text(errors="replace").splitlines() if (m := pf.match(l))]
reps = sorted({p.stem.split("@")[1] for p in d.glob("dfl@*.err")})
for name, i in (("prefill 4k", 14), ("prefill 16k", 15)):
    rs, a, b = [], [], []
    for r in reps:
        x, y = rd(d / f"dspark@{r}.err")[i], rd(d / f"dfl@{r}.err")[i]
        a.append(x[0] / x[1] * 1e3); b.append(y[0] / y[1] * 1e3); rs.append(math.log((y[0] / y[1]) / (x[0] / x[1])))
    mu = sum(rs) / len(rs); sd = math.sqrt(sum((v - mu) ** 2 for v in rs) / (len(rs) - 1)) if len(rs) > 1 else 0
    h = T.get(len(rs) - 1, 2.0) * sd / math.sqrt(len(rs))
    sa, sb = sorted(a), sorted(b)
    print(f"{name}: dspark {sa[len(sa)//2]:.0f} dfl {sb[len(sb)//2]:.0f} tok/s, ratio {math.exp(mu):.3f} [{math.exp(mu-h):.3f}-{math.exp(mu+h):.3f}] n={len(rs)}")
