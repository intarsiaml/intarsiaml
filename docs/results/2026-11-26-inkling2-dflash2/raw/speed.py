#!/usr/bin/env python3
"""Lane ink-dflash2 copy (from main, docs/results/2026-11-26-inkling2-hip-combined/raw/speed.py; adds per-class tokens per pass from
`spec[<name>]` lines and prefill tok/s per class). Lane ink-hip-verify speed analysis (after lane ink-hip-spec's analyze.py): decode tok/s per prompt from `generate --prompt-ids`
stderr (`decode[<name>]: N tokens, X ms (Y tok/s)`), one process per arm and rep: files <dir>/<arm>@<rep>.err (+ .out). Ratio of an
arm to the reference arm per (prompt, rep) (the same rep = the same card and position in the ABBA order), geometric mean per class,
95 % CI: t-interval on the log of the reps' geometric means (the processes are the units) and a bootstrap over reps x prompts.
Also: ids of every prompt identical across arms and reps; spec counters per arm; prefill ms per prompt.
usage: speed.py DIR REF "short=1-6,1k=7,4k=8,16k=9" [ARM ...]   (prompt 0 = warm-up, never in a class)"""
import math, random, re, sys
from pathlib import Path

d, ref, spec_cls = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
CLASSES = []
for part in spec_cls.split(","):
    name, rng_ = part.split("=")
    idx = []
    for piece in rng_.split("+"):
        a, _, b = piece.partition("-")
        idx += list(range(int(a), int(b or a) + 1))
    CLASSES.append((name, idx))
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
dec = re.compile(r"^decode\[([^\]]*)\]: (\d+) tokens, (\d+) ms \(([\d.]+) tok/s\)")
idl = re.compile(r"^ids\[([^\]]*)\]: (.*)$")
spec = re.compile(r"^spec: verify passes=(\d+) proposed=(\d+) accepted=(\d+)")
pf = re.compile(r"^prefill\[([^\]]*)\]: (\d+) tokens, (\d+) ms")
spp = re.compile(r"^spec\[([^\]]*)\]: verify passes=(\d+) proposed=(\d+) accepted=(\d+)")


def read(p):
    o = p.with_suffix(".out")
    rates, ids, sp, pre, per = [], [], None, [], {}
    for line in p.read_text(errors="replace").splitlines() + (o.read_text(errors="replace").splitlines() if o.exists() else []):
        if m := dec.match(line):
            rates.append((int(m.group(2)), float(m.group(4))))
        if m := idl.match(line):
            ids.append(m.group(2).strip())
        if m := spec.match(line):
            sp = tuple(int(x) for x in m.groups())
        if m := pf.match(line):
            pre.append((int(m.group(2)), int(m.group(3))))
        if m := spp.match(line):
            per[len(rates) - 1] = (int(m.group(2)), int(m.group(4)))  # (passes, accepted) of the prompt just decoded
    return rates, ids, sp, pre, per


data = {}
for p in sorted(d.glob("*@*.err")):
    arm, rep = p.stem.split("@")
    data[(arm, rep)] = read(p)
arms = sys.argv[4:] or sorted({a for a, _ in data})
reps = sorted({r for _, r in data})
n = max(len(v[0]) for v in data.values())
full = {k: v for k, v in data.items() if len(v[0]) == n}
print(f"processes: {len(data)} ({len(full)} complete, {n} prompts each); reps {reps}")
ref_ids, bad = None, []
for k, v in sorted(full.items()):
    if ref_ids is None:
        ref_ids = v[1]
    elif v[1] != ref_ids:
        bad.append(f"{k}: {sum(1 for a, b in zip(v[1], ref_ids) if a != b)} prompts differ")
print("ids identical across every arm and rep:", "yes" if not bad and ref_ids else ("no ids" if not ref_ids else "NO: " + "; ".join(bad)))
for a in arms:
    sps = [full[(a, r)][2] for r in reps if (a, r) in full and full[(a, r)][2]]
    if sps:
        p, q, acc = (sum(x[i] for x in sps) for i in range(3))
        print(f"spec {a}: verify passes {p}, proposed {q}, accepted {acc} ({100 * acc / max(q, 1):.1f} %, {(acc + p) / max(p, 1):.2f} tokens a pass), {len(sps)} processes")
gm = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs))
rnd = random.Random(7)
print()
print(f"| decode, arm / {ref} | " + " | ".join(c for c, _ in CLASSES) + " |")
print("|---|" + "---:|" * len(CLASSES))
for b in arms:
    if b == ref:
        continue
    cells = []
    for cname, idx in CLASSES:
        per_rep, pairs = [], []
        for r in reps:
            if (b, r) not in full or (ref, r) not in full:
                continue
            rs = [full[(b, r)][0][i][1] / full[(ref, r)][0][i][1] for i in idx]
            per_rep.append(gm(rs))
            pairs.append(rs)
        if not per_rep:
            cells.append("-")
            continue
        g = gm(per_rep)
        if len(per_rep) > 1:
            lg = [math.log(x) for x in per_rep]
            mu = sum(lg) / len(lg)
            sd = math.sqrt(sum((x - mu) ** 2 for x in lg) / (len(lg) - 1))
            h = T975.get(len(lg) - 1, 2.0) * sd / math.sqrt(len(lg))
            ci = f"[{math.exp(mu - h):.3f}-{math.exp(mu + h):.3f}"
        else:
            ci = "[n=1"
        if len(idx) > 1:
            boots = []
            for _ in range(2000):
                rr = [pairs[rnd.randrange(len(pairs))] for _ in pairs]
                boots.append(gm([gm([p[rnd.randrange(len(p))] for _ in p]) for p in rr]))
            boots.sort()
            ci += f"; boot {boots[50]:.3f}-{boots[1949]:.3f}"
        cells.append(f"**{g:.3f}** {ci}]")
    print(f"| {b} | " + " | ".join(cells) + " |")
print()
print("| decode tok/s, median [range] | " + " | ".join(c for c, _ in CLASSES) + " |")
print("|---|" + "---:|" * len(CLASSES))
for a in arms:
    cells = []
    for cname, idx in CLASSES:
        vals = sorted(gm([full[(a, r)][0][i][1] for i in idx]) for r in reps if (a, r) in full)
        if vals:
            md = vals[len(vals) // 2] if len(vals) % 2 else (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2
            cells.append(f"{md:.2f} [{vals[0]:.2f}-{vals[-1]:.2f}]")
        else:
            cells.append("-")
    print(f"| {a} | " + " | ".join(cells) + " |")
print()
print("per process (class geomean tok/s):")
for (a, r), v in sorted(full.items()):
    print(f"  {a}@{r}: " + " ".join(f"{c} {gm([v[0][i][1] for i in idx]):.2f}" for c, idx in CLASSES) + (f" | spec {v[2]}" if v[2] else ""))
print()
print("tokens per verify pass (passes and accepted summed over the processes): | " + " | ".join(c for c, _ in CLASSES) + " |")
for a in arms:
    cells = []
    for cname, idx in CLASSES:
        ps = sum(full[(a, r)][4].get(i, (0, 0))[0] for r in reps if (a, r) in full for i in idx)
        ac = sum(full[(a, r)][4].get(i, (0, 0))[1] for r in reps if (a, r) in full for i in idx)
        cells.append(f"{(ps + ac) / ps:.2f}" if ps else "-")
    print(f"| {a} | " + " | ".join(cells) + " |")
print()
print("prefill tok/s per class (median over processes of the class's summed tokens / summed ms): | " + " | ".join(c for c, _ in CLASSES) + " |")
for a in arms:
    cells = []
    for cname, idx in CLASSES:
        vals = sorted(sum(full[(a, r)][3][i][0] for i in idx) / max(sum(full[(a, r)][3][i][1] for i in idx), 1) * 1e3 for r in reps if (a, r) in full and len(full[(a, r)][3]) > max(idx))
        cells.append(f"{vals[len(vals) // 2]:.0f}" if vals else "-")
    print(f"| {a} | " + " | ".join(cells) + " |")
