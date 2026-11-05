#!/usr/bin/env python3
"""Lane dsv41/prefill3: per launch shape (class, kernel, grid) of one 4106-token request (the second of each pair), ours or TF.

Usage: groups.py ours|tf TRACE.csv[.gz] [--top 4]
Prints, per class, the launch shapes with the most time: launches, mean ms, total ms. A 2048-row launch is the one with the largest
grid of its kernel; the 118/128-row decoder launches and the 10-row tail launches are the smaller ones.
"""
import csv
import gzip
import io
import re
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from classes import OURS, TF, classify  # noqa: E402


def rows(path):
    f = io.TextIOWrapper(gzip.open(path)) if path.endswith(".gz") else open(path, newline="")
    out = []
    for r in csv.DictReader(l for l in f if l.strip()):
        if r.get("Name") and r.get("Duration (ns)"):
            out.append((int(r["Start (ns)"]), int(r["Duration (ns)"]), r["Name"], (r["GrdX"], r["GrdY"], r["GrdZ"])))
    out.sort()
    return out


def main():
    eng, path = sys.argv[1], sys.argv[2]
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 4
    rules = OURS if eng == "ours" else TF
    reqs, cur, end = [], [], 0
    for r in rows(path):
        if cur and r[0] - end > 1e9:
            reqs.append(cur)
            cur, end = [], 0
        cur.append(r)
        end = max(end, r[0] + r[1])
    reqs.append(cur)
    reqs = [q for q in reqs if (max(r[0] + r[1] for r in q) - q[0][0]) > 50e6]
    q = reqs[1]
    g = defaultdict(list)
    for s, d, n, grid in q:
        nm = re.sub(r"\(.*$", "", n.replace("(anonymous namespace)::", "").replace("<unnamed>::", ""))[:60]
        g[(classify(rules, n), nm, grid)].append(d / 1e6)
    byc = defaultdict(list)
    for (c, nm, grid), v in g.items():
        byc[c].append((sum(v), len(v), st.mean(v), nm, grid))
    for c in sorted(byc, key=lambda c: -sum(x[0] for x in byc[c])):
        print(f"{c}  total {sum(x[0] for x in byc[c]):.1f} ms")
        for tot, n, mean, nm, grid in sorted(byc[c], reverse=True)[:top]:
            print(f"    {n:4d} x {mean:8.3f} ms = {tot:8.1f}  {nm}  grid {','.join(grid)}")


if __name__ == "__main__":
    main()
