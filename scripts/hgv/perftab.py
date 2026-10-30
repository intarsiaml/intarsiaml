#!/usr/bin/env python3
"""Lane hgv: markdown A/B table from the perf windows' localeval logs.
  perftab.py raw/perf-r1 raw/perf-r2 ...   (logs localeval-perf-r<R>-tp<N>-<base|hgv>-<k>.log)"""
import glob, re, statistics as st, sys, collections

res = collections.defaultdict(lambda: collections.defaultdict(list))  # (tp, tree) -> col -> [per round]
for d in sys.argv[1:]:
    for f in sorted(glob.glob(f"{d}/localeval-perf-*.log")):
        m = re.search(r"perf-r(\d+)-(tp\d)-(base|hgv)-\d+\.log", f)
        if not m:
            continue
        t = open(f, errors="replace").read()
        s1 = re.search(r"decode=([\d.]+) tok/s", t)
        row = {"1": float(s1[1])} if s1 else {}
        for n, v in re.findall(r"(\d+) streams: aggregate ([\d.]+) tok/s", t):
            row[n] = float(v)
        for k, v in row.items():
            res[(m[2], m[3])][k].append((int(m[1]), v))
cols = ["1", "4", "8", "16"]
print("| TP | streams | main median | hgv median | change | main rounds | hgv rounds |\n|---|---:|---:|---:|---:|---|---|")
for tp in ("tp1", "tp2"):
    for c in cols:
        a, b = res[(tp, "base")][c], res[(tp, "hgv")][c]
        if not a or not b:
            continue
        ma, mb = st.median(v for _, v in a), st.median(v for _, v in b)
        fa = " / ".join(f"{v:.1f}" for _, v in sorted(a))
        fb = " / ".join(f"{v:.1f}" for _, v in sorted(b))
        print(f"| {tp} | {c} | {ma:.1f} | {mb:.1f} | {100 * (mb / ma - 1):+.1f} % | {fa} | {fb} |")
