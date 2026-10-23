#!/usr/bin/env python3
"""Lane pf: one row per qualify run (raw/q*/<arm>/report/logs/*.out): short/oracle/long KL, baseline-gate and drift numbers."""
import glob, os, re, sys
print("| arm | acc-short prefill KL21 (scored) | short top-1 | short decode KL21 | short baseline gate | oracle KL21 (scored) vs f64 oracle | oracle drift vs recorded main dump | acc-long prefill KL21 (scored) | long decode KL21 | acc-4k |")
print("|---|---|---|---|---|---|---|---|---|---|")
for d in sys.argv[1:]:
    for arm in sorted(os.listdir(d)):
        L = f"{d}/{arm}/report/logs"
        if not os.path.isdir(L): continue
        r = lambda n: open(f"{L}/{n}", errors="replace").read() if os.path.exists(f"{L}/{n}") else ""
        s, o, l = r("acc-short.out"), r("acc-oracle.out"), r("acc-long.out")
        k = lambda t, mode: (re.search(mode + r" KL threshold.*?KL ([\d.e+-]+) (PASS|FAIL)", t) or [None, "-", "-"])
        sk, ok, lk = k(s, "prefill"), k(o, "prefill"), k(l, "prefill")
        sd, ld = k(s, "decode"), k(l, "decode")
        top1 = re.search(r"OVERALL\s+\d+\s+([\d.]+)", s)
        base = re.search(r"floor verdict.*?baseline verdict[^:]*: (\w+)", s)
        drift = re.search(r"mean KL\(21-way\)\s+([\d.e+-]+)\s+\(limit", r("acc-oracle-drift.out") + r("acc-oracle.out") + open(f"{d}/{arm}/report/quick-report.md").read())
        a4 = re.search(r"`acc-4k` \*\*(\w+)\*\*", open(f"{d}/{arm}/report/full-report.md", errors="replace").read()) if os.path.exists(f"{d}/{arm}/report/full-report.md") else None
        print(f"| {os.path.basename(d)}/{arm} | {sk[1]} {sk[2]} | {top1.group(1) if top1 else '-'} | {sd[1]} | {base.group(1) if base else '-'} | {ok[1]} {ok[2]} | {drift.group(1) if drift else '-'} | {lk[1]} {lk[2]} | {ld[1]} | {a4.group(1) if a4 else '-'} |")
