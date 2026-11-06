#!/usr/bin/env python3
"""Lane spark35: turn the raw KL logs of kl.sh into the tables and the Q1..Q4 verdicts of docs/results/2026-11-05-spark35-numerics.md.
  python3 scripts/spark35num/analyze.py docs/results/2026-11-05-spark35-numerics/raw
Reads acc-<set>-<arm>.log (the `intarsia accuracy` output: per-prompt rows of each mode, against the base arm's dump) and
pos-<set>-<arm>.tsv.gz (INTARSIA_ACC_POS_DUMP: mode prompt pos token top1 kl21 tail_ref tail_ours)."""
import gzip, math, pathlib, re, sys
raw = pathlib.Path(sys.argv[1])
ARMS = ["noise", "fp8", "cemu", "cdemu", "creal"]
ROW = re.compile(r"^(\S+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+([\d.]+)\s+(\d+)/(\d+)\s+(\d+)\s*$")
def parse(path):
    """-> {mode: {prompt: dict(n, pos, top1, kl21, greedy_match)}}"""
    out, mode = {}, None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("== mode: prefill"): mode = "prefill"
        elif line.startswith("== mode: decode"): mode = "decode"
        elif mode:
            m = ROW.match(line.strip())
            if m and m.group(1) != "OVERALL":
                out.setdefault(mode, {})[m.group(1)] = dict(n=int(m.group(2)), pos=int(m.group(3)), top1=float(m.group(4)), kl21=float(m.group(7)), gm=int(m.group(12)))
    return out
def pos_quarters(path, mode, prompt):
    """mean KL21 of the first and the last quarter (by position) of a prompt's rows in a mode"""
    if not path.exists(): return None
    rows = []
    with gzip.open(path, "rt") as f:
        for l in f:
            p = l.split("\t")
            if p[0].startswith(mode) and p[1] == prompt: rows.append((int(p[2]), float(p[5])))
    if len(rows) < 8: return None
    rows.sort(); q = len(rows) // 4
    mean = lambda r: sum(k for _, k in r) / len(r)
    return mean(rows[:q]), mean(rows[-q:])
def get(set_, arm):
    p = raw / f"acc-{set_}-{arm}.log"
    return parse(p) if p.exists() else None
res = {(s, a): get(s, a) for s in ("std", "dec8k") for a in ARMS}
def table(set_, mode, title):
    ps = None
    for a in ARMS:
        r = res[(set_, a)]
        if r and mode in r: ps = list(r[mode]); break
    if not ps: return
    print(f"\n### {title}\n\n| prompt | tokens scored | " + " | ".join(f"{a} KL21 / flips %" for a in ARMS) + " |\n|---|---:|" + "---:|" * len(ARMS))
    for p in ps:
        cells = []
        for a in ARMS:
            r = res[(set_, a)]
            d = r[mode].get(p) if r and mode in r else None
            cells.append(f"{d['kl21']:.2e} / {100 - d['top1']:.2f}" if d else "n/a")
        n = next((res[(set_, a)][mode][p]["pos"] for a in ARMS if res[(set_, a)] and mode in res[(set_, a)] and p in res[(set_, a)][mode]), "")
        print(f"| {p} | {n} | " + " | ".join(cells) + " |")
table("std", "prefill", "Prefill rows, every position (mean 21-way KL in nats against base / top-1 flips %)")
table("std", "decode", "Decode window 2048 (teacher-forced, state rounded every token)")
table("dec8k", "decode", "Decode window 8192")
print("\n### Early vs late quarter of the decode windows (mean KL21)\n\n| set | prompt | " + " | ".join(f"{a} first / last" for a in ARMS) + " |\n|---|---|" + "---|" * len(ARMS))
for s in ("std", "dec8k"):
    r0 = next((res[(s, a)] for a in ARMS if res[(s, a)] and "decode" in res[(s, a)]), None)
    for p in (r0["decode"] if r0 else []):
        cells = []
        for a in ARMS:
            q = pos_quarters(raw / f"pos-{s}-{a}.tsv.gz", "decode", p)
            cells.append(f"{q[0]:.2e} / {q[1]:.2e}" if q else "n/a")
        print(f"| {s} | {p} | " + " | ".join(cells) + " |")
# verdicts
def L(p):  # length class of a prompt name
    return {"cmtp-4k-0": "4k", "cmtp-16k-0": "16k", "code-16k": "16k", "docs-32k": "32k", "docs-64k": "64k"}.get(p, p)
print("\n### Verdicts (rules Q1 to Q4 of the results doc)\n")
def cls(arm, mode, set_="std"):
    r = res[(set_, arm)]
    return (r or {}).get(mode, {})
for arm in ("fp8", "cemu", "cdemu", "creal"):
    pf, nz = cls(arm, "prefill"), cls("noise", "prefill")
    if not pf or not nz: print(f"- {arm}: no prefill data"); continue
    q1 = {p: (pf[p]["kl21"] <= max(5 * nz[p]["kl21"], 2e-3)) for p in pf if p in nz}
    q3 = {p: ((100 - pf[p]["top1"]) <= max(3 * (100 - nz[p]["top1"]), 1.5)) for p in pf if p in nz}
    k4 = pf.get("cmtp-4k-0", {}).get("kl21"); n4 = nz.get("cmtp-4k-0", {}).get("kl21"); k64 = pf.get("docs-64k", {}).get("kl21")
    q2 = (k64 <= 3 * max(k4, n4)) if None not in (k4, n4, k64) else None
    print(f"- {arm}: Q1 {'PASS' if all(q1.values()) else 'FAIL ' + str([p for p, v in q1.items() if not v])}; Q2 {q2}; Q3 {'PASS' if all(q3.values()) else 'FAIL ' + str([p for p, v in q3.items() if not v])}")
    for s in ("std", "dec8k"):
        dc, dn = cls(arm, "decode", s), cls("noise", "decode", s)
        for p in dc:
            ok = p in dn and dc[p]["kl21"] <= max(5 * dn[p]["kl21"], 2e-3)
            q = pos_quarters(raw / f"pos-{s}-{arm}.tsv.gz", "decode", p)
            drift = (q[1] <= 2 * q[0] + 5e-4) if q else None
            print(f"    Q4 {s}/{p}: KL<=bound {'PASS' if ok else 'FAIL'}; no-drift {drift}")
