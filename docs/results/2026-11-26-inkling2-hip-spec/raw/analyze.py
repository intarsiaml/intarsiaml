#!/usr/bin/env python3
"""Lane ink-hip-spec speed analysis: decode tok/s per prompt from `generate --prompt-ids` stderr (`decode[<name>]: N tokens, X ms
(Y tok/s)`), arms off / gate / cut, reps r1.. (one process per arm and rep). Ratio of an arm to off per (prompt, rep), geometric mean
per length class with a 95 % CI: t-interval on the log ratios of the rep means (reps are the independent units), and a bootstrap
over reps x prompts. Also: ids of every prompt identical across arms and reps (from the `ids[...]` lines)."""
import math
import random
import re
import sys
from pathlib import Path

d = Path(sys.argv[1])
# prompt 0 is a warm-up copy of prompt 1 (startup, first graph captures), left out of every class
CLASSES = [("short", range(1, 13)), ("1k", [13]), ("4k", [14]), ("16k", [15])]
NPROMPTS = 16
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}
dec = re.compile(r"^decode\[([^\]]*)\]: (\d+) tokens, (\d+) ms \(([\d.]+) tok/s\)")
idl = re.compile(r"^ids\[([^\]]*)\]: (.*)$")
spec = re.compile(r"^spec: verify passes=(\d+) proposed=(\d+) accepted=(\d+)")
pf = re.compile(r"^prefill\[([^\]]*)\]: (\d+) tokens, (\d+) ms")


def read(arm, rep):
    e, o = d / f"{arm}-r{rep}.err", d / f"{arm}-r{rep}.out"
    if not e.exists():
        return None
    rates, ids, sp, pre = [], [], None, []
    for line in e.read_text(errors="replace").splitlines() + (o.read_text(errors="replace").splitlines() if o.exists() else []):
        m = dec.match(line)
        if m:
            rates.append((int(m.group(2)), float(m.group(4))))
        m = idl.match(line)
        if m:
            ids.append(m.group(2).strip())
        m = spec.match(line)
        if m:
            sp = tuple(int(x) for x in m.groups())
        m = pf.match(line)
        if m:
            pre.append((int(m.group(2)), int(m.group(3))))
    return rates, ids, sp, pre


arms = ["off", "gate", "cut"]
reps = sorted({int(p.stem.split("-r")[1]) for p in d.glob("*-r*.err")})
data = {(a, r): read(a, r) for a in arms for r in reps}
data = {k: v for k, v in data.items() if v and len(v[0]) == NPROMPTS}
ref_ids = None
bad = []
for k, v in sorted(data.items()):
    if ref_ids is None:
        ref_ids = v[1]
    elif v[1] != ref_ids:
        bad.append(f"{k}: {sum(1 for a, b in zip(v[1], ref_ids) if a != b)} prompts differ")
print(f"processes parsed: {len(data)}; reps {reps}")
print("ids identical across every arm and rep:" if not bad else "IDS DIFFER:", "yes" if not bad else "; ".join(bad))
for a in arms:
    sps = [data[(a, r)][2] for r in reps if (a, r) in data and data[(a, r)][2]]
    if sps:
        p, q, acc = (sum(x[i] for x in sps) for i in range(3))
        print(f"spec {a}: verify passes {p}, proposed {q}, accepted {acc} ({100 * acc / max(q, 1):.1f} %)")


def gm(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


rng = random.Random(7)
print()
print("| arm / off | " + " | ".join(c for c, _ in CLASSES) + " |")
print("|---|" + "---:|" * len(CLASSES))
for b in ["gate", "cut"]:
    cells = []
    for cname, idx in CLASSES:
        per_rep = []
        pairs = []
        for r in reps:
            if (b, r) not in data or ("off", r) not in data:
                continue
            rs = [data[(b, r)][0][i][1] / data[("off", r)][0][i][1] for i in idx]
            pairs.append(rs)
            per_rep.append(gm(rs))
        if not per_rep:
            cells.append("n/a")
            continue
        g = gm([x for rs in pairs for x in rs])
        n = len(per_rep)
        logs = [math.log(x) for x in per_rep]
        mu = sum(logs) / n
        if n > 1:
            sd = math.sqrt(sum((x - mu) ** 2 for x in logs) / (n - 1))
            h = T975.get(n - 1, 2.0) * sd / math.sqrt(n)
            lo, hi = math.exp(mu - h), math.exp(mu + h)
        else:
            lo = hi = float("nan")
        boots = []
        for _ in range(4000):
            rr = [pairs[rng.randrange(n)] for _ in range(n)]
            xs = [rs[rng.randrange(len(rs))] for rs in rr for _ in rs]
            boots.append(gm(xs))
        boots.sort()
        cells.append(f"**{g:.3f}** [t {lo:.3f}-{hi:.3f}; boot {boots[100]:.3f}-{boots[3899]:.3f}] (n {n})")
    print(f"| {b} | " + " | ".join(cells) + " |")
print()
print("| median tok/s | " + " | ".join(c for c, _ in CLASSES) + " |")
print("|---|" + "---:|" * len(CLASSES))
for a in arms:
    cells = []
    for cname, idx in CLASSES:
        xs = sorted(gm([data[(a, r)][0][i][1] for i in idx]) for r in reps if (a, r) in data)
        cells.append(f"{xs[len(xs) // 2]:.2f} [{xs[0]:.2f}-{xs[-1]:.2f}]" if xs else "n/a")
    print(f"| {a} | " + " | ".join(cells) + " |")

print()
print("| median prefill tok/s | 1k | 4k | 16k |")
print("|---|---:|---:|---:|")
for a in arms:
    cells = []
    for i in (13, 14, 15):
        xs = sorted(data[(a, r)][3][i][0] / data[(a, r)][3][i][1] * 1e3 for r in reps if (a, r) in data and len(data[(a, r)][3]) == NPROMPTS)
        cells.append(f"{xs[len(xs) // 2]:.0f} [{xs[0]:.0f}-{xs[-1]:.0f}]" if xs else "n/a")
    print(f"| {a} | " + " | ".join(cells) + " |")
