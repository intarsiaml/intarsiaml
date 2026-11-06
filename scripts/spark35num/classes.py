#!/usr/bin/env python3
"""Lane spark35: rules Q1 to Q4 of docs/results/2026-11-05-spark35-numerics.md by LENGTH CLASS (analyze.py checks them per prompt; the rule says per class:
the 16k class holds two prompts, cmtp-16k-0 and code-16k, pooled here weighted by scored positions).
  python3 scripts/spark35num/classes.py docs/results/2026-11-05-spark35-numerics/raw-merged
Reads acc-<set>-<arm>.log and pos-<set>-<arm>.tsv.gz exactly like analyze.py (it imports its parser)."""
import gzip, pathlib, re, sys
raw = pathlib.Path(sys.argv[1])
ROW = re.compile(r"^(\S+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+([\d.]+)\s+(\d+)/(\d+)\s+(\d+)\s*$")
CLS = {"cmtp-4k-0": "4k", "cmtp-16k-0": "16k", "code-16k": "16k", "docs-32k": "32k", "docs-64k": "64k"}
ARMS = ["noise", "fp8", "cemu", "cdemu", "creal"]
def parse(path):
    out, mode = {}, None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("== mode: prefill"): mode = "prefill"
        elif line.startswith("== mode: decode"): mode = "decode"
        elif mode:
            m = ROW.match(line.strip())
            if m and m.group(1) != "OVERALL":
                out.setdefault(mode, {})[m.group(1)] = dict(pos=int(m.group(3)), flips=100 - float(m.group(4)), kl=float(m.group(7)))
    return out
def pooled(rows, prompts):
    n = sum(rows[p]["pos"] for p in prompts)
    return sum(rows[p]["kl"] * rows[p]["pos"] for p in prompts) / n, sum(rows[p]["flips"] * rows[p]["pos"] for p in prompts) / n
def quarters(arm, set_, prompts):
    """first / last quarter mean KL21 of the decode window, pooled over the prompts (rows by position within each prompt)"""
    f = raw / f"pos-{set_}-{arm}.tsv.gz"
    if not f.exists(): return None
    per = {}
    with gzip.open(f, "rt") as fh:
        for l in fh:
            p = l.split("\t")
            if p[0].startswith("decode") and p[1] in prompts: per.setdefault(p[1], []).append((int(p[2]), float(p[5])))
    fq, lq = [], []
    for rows in per.values():
        rows.sort(); q = len(rows) // 4
        fq += [k for _, k in rows[:q]]; lq += [k for _, k in rows[-q:]]
    return (sum(fq) / len(fq), sum(lq) / len(lq)) if fq else None
logs = {(s, a): parse(raw / f"acc-{s}-{a}.log") for s in ("std", "dec8k") for a in ARMS if (raw / f"acc-{s}-{a}.log").exists()}
N = logs.get(("std", "noise"))
if not N: sys.exit("no noise arm")
verdict = {}
def say(arm, rule, ok): verdict.setdefault(arm, {}).setdefault(rule, []).append(ok)
print("### Prefill rows by length class: K (mean 21-way KL, nats) / flips %, bound = max(5 x noise, 2e-3) and max(3 x noise flips, 1.5)\n")
print("| class | prompts | positions | noise K / flips | " + " | ".join(f"{a} K / flips (K/noise)" for a in ARMS[1:]) + " |\n|---|---|---:|---|" + "---|" * (len(ARMS) - 1))
for c in ("4k", "16k", "32k", "64k"):
    ps = [p for p in N["prefill"] if CLS[p] == c]
    nk, nf = pooled(N["prefill"], ps); cells = []
    for a in ARMS[1:]:
        L = logs.get(("std", a))
        if not L or "prefill" not in L: cells.append("n/a"); continue
        k, f = pooled(L["prefill"], ps)
        q1, q3 = k <= max(5 * nk, 2e-3), f <= max(3 * nf, 1.5)
        if a != "creal": say(a, "Q1", q1); say(a, "Q3", q3)
        cells.append(f"{k:.2e} / {f:.2f} ({k / nk:.2f}x){'' if q1 and q3 else ' **FAIL**'}")
    print(f"| {c} | {', '.join(ps)} | {sum(N['prefill'][p]['pos'] for p in ps)} | {nk:.2e} / {nf:.2f} | " + " | ".join(cells) + " |")
k4n = pooled(N["prefill"], [p for p in N["prefill"] if CLS[p] == "4k"])[0]
print("\n### Q2: K_64k <= 3 x max(K_4k, N_4k)\n")
for a in ARMS[1:]:
    L = logs.get(("std", a))
    if not L or "prefill" not in L or a == "creal": continue
    k4 = pooled(L["prefill"], ["cmtp-4k-0"])[0]; k64 = pooled(L["prefill"], ["docs-64k"])[0]
    bound = 3 * max(k4, k4n); say(a, "Q2", k64 <= bound)
    print(f"- {a}: K_64k {k64:.2e} <= 3 x max({k4:.2e}, {k4n:.2e}) = {bound:.2e}: {'PASS' if k64 <= bound else 'FAIL'}")
print("\n### Q4: decode windows (teacher-forced), K bound = max(5 x noise, 2e-3), no drift: last quarter <= 2 x first quarter + 5e-4\n")
print("| window | arm | K (noise K) | flips % (noise) | first / last quarter | verdict |\n|---|---|---|---|---|---|")
for s, cls_list in (("std", ["4k", "16k", "32k", "64k"]), ("dec8k", [None])):
    for c in cls_list:
        for a in ARMS[1:]:
            L, Ns = logs.get((s, a)), logs.get((s, "noise"))
            if not L or not Ns or "decode" not in L or "decode" not in Ns: continue
            ps = [p for p in Ns["decode"] if c is None or CLS[p] == c]
            k, f = pooled(L["decode"], ps); nk, nf = pooled(Ns["decode"], ps)
            q = quarters(a, s, ps)
            ok = k <= max(5 * nk, 2e-3) and (q is None or q[1] <= 2 * q[0] + 5e-4)
            if a != "creal": say(a, "Q4", ok)
            print(f"| {s} {c or 'code-16k 8192'} | {a} | {k:.2e} ({nk:.2e}) | {f:.2f} ({nf:.2f}) | " + (f"{q[0]:.2e} / {q[1]:.2e}" if q else "n/a") + f" | {'PASS' if ok else 'FAIL'} |")
print("\n### Verdict by arm (all classes and windows)\n")
for a in ("fp8", "cemu", "cdemu"):
    v = verdict.get(a, {})
    print(f"- {a}: " + "; ".join(f"{r} {'PASS' if v.get(r) and all(v[r]) else ('FAIL' if v.get(r) else 'n/a')}" for r in ("Q1", "Q2", "Q3", "Q4")))
