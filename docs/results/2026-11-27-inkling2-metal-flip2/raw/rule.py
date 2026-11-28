#!/usr/bin/env python3
"""Lane ink-metal-dflash2: the pre-registered rule on the ABBA runs. usage: rule.py DIR ARM [REF=dspark]; DIR holds <arm>@<rep>.err of every
rep (the two halves copied in, reps r1..r4 unique). Cells short / 1k / prose / chat / 4k / 16k (the #541 definitions: prompts 1-12, 13,
16-17, 5-12, 14, 15 of speed.json, 0 = warm-up). Ratio ARM / REF decode tok/s per (prompt, rep); a cell's rep value = geometric mean over its
prompts; the cell = geometric mean over reps, 95 % CI = t-interval on the log of the reps' values (the processes are the units).
GO: mean of the six cells >= 1.03, no cell with a CI upper bound < 0.99, ids identical (checked here: every process, every prompt)."""
import math, re, sys
from pathlib import Path
d, arm = Path(sys.argv[1]), sys.argv[2]
ref = sys.argv[3] if len(sys.argv) > 3 else "dspark"
CELLS = [("short", range(1, 13)), ("1k", [13]), ("prose", [16, 17]), ("chat", range(5, 13)), ("4k", [14]), ("16k", [15])]
T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}
dec = re.compile(r"^decode\[([^\]]*)\]: (\d+) tokens, (\d+) ms \(([\d.]+) tok/s\)")
idl = re.compile(r"^ids\[([^\]]*)\]: (.*)$")
def rd(p):
    r, ids = [], []
    for l in p.read_text(errors="replace").splitlines():
        if m := dec.match(l): r.append(float(m.group(4)))
        if m := idl.match(l): ids.append(m.group(2).strip())
    return r, ids
reps = sorted({p.stem.split("@")[1] for p in d.glob(f"{arm}@*.err")} & {p.stem.split("@")[1] for p in d.glob(f"{ref}@*.err")})
data = {(a, r): rd(d / f"{a}@{r}.err") for a in (arm, ref) for r in reps}
n = max(len(v[0]) for v in data.values())
ok = all(len(v[0]) == n for v in data.values())
ids0 = next(iter(data.values()))[1]
same = all(v[1] == ids0 for v in data.values())
print(f"{arm} vs {ref}: reps {reps}, {n} prompts each, all complete: {ok}, ids identical across every process (both arms): {same}")
gm = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs))
def ci(lg):
    mu = sum(lg) / len(lg)
    sd = math.sqrt(sum((x - mu) ** 2 for x in lg) / (len(lg) - 1)) if len(lg) > 1 else 0.0
    h = T.get(len(lg) - 1, 2.0) * sd / math.sqrt(len(lg))
    return math.exp(mu), math.exp(mu - h), math.exp(mu + h)
cells, per_rep_mean = [], []
for name, idx in CELLS:
    vals = [gm([data[(arm, r)][0][i] / data[(ref, r)][0][i] for i in idx]) for r in reps]
    g, lo, hi = ci([math.log(v) for v in vals])
    cells.append((name, g, lo, hi, vals))
print("| cell | ratio | 95 % CI | per rep |\n|---|---:|---|---|")
for name, g, lo, hi, vals in cells:
    print(f"| {name} | **{g:.3f}** | [{lo:.3f}-{hi:.3f}] | {' '.join(f'{v:.3f}' for v in vals)} |")
geo = gm([c[1] for c in cells]); ari = sum(c[1] for c in cells) / len(cells)
mrep = [gm([c[4][j] for c in cells]) for j in range(len(reps))]
mg, mlo, mhi = ci([math.log(v) for v in mrep])
print(f"mean of the six cells: geo {geo:.3f}, arith {ari:.3f}; per-rep geo means {' '.join(f'{v:.3f}' for v in mrep)} -> {mg:.3f} [{mlo:.3f}-{mhi:.3f}]")
slow = [c[0] for c in cells if c[3] < 0.99]
faster = [c[0] for c in cells if c[1] > 1.0]
verdict = "GO" if (geo >= 1.03 and not slow and same and ok) else ("OPT-IN" if faster else "NO-GO")
print(f"cells with CI upper bound < 0.99: {slow or 'none'}; cells faster by the point estimate: {faster}")
print(f"VERDICT by the pre-registered rule: {verdict}")
