#!/usr/bin/env python3
"""Lane lb: per-kernel before/after table from rocprofv3 kernel traces.

usage: kagg.py <mode> <base csv>[,<base csv>...] <dev csv>[,<dev csv>...] [min_total_us]
Each csv is one run; per kernel the launch count and mean duration are taken over all runs of a side (mean of per-run totals),
kernels are matched by name, and the table is sorted by the base total. Prints a markdown table.
"""
import csv
import sys
from collections import defaultdict


def load(paths):
    runs = []
    for p in paths.split(","):
        agg = defaultdict(lambda: [0, 0])
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                name = r["Kernel_Name"]
                if len(name) > 90:
                    name = name[:90]
                d = int(r["End_Timestamp"]) - int(r["Start_Timestamp"])
                agg[name][0] += 1
                agg[name][1] += d
        runs.append(agg)
    return runs


def mean_side(runs):
    names = set()
    for r in runs:
        names |= set(r)
    out = {}
    for n in names:
        cnt = sum(r[n][0] for r in runs if n in r) / len(runs)
        tot = sum(r[n][1] for r in runs if n in r) / len(runs)
        out[n] = (cnt, tot)
    return out


def main():
    mode, b, d = sys.argv[1:4]
    floor = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    base, dev = mean_side(load(b)), mean_side(load(d))
    rows = []
    for n in set(base) | set(dev):
        bc, bt = base.get(n, (0, 0))
        dc, dt = dev.get(n, (0, 0))
        rows.append((n, bc, bt, dc, dt))
    rows.sort(key=lambda r: -r[2])
    tb = sum(r[2] for r in rows)
    td = sum(r[4] for r in rows)
    print(f"### {mode}: kernel time base {tb/1e6:.2f} ms, dev {td/1e6:.2f} ms ({(td-tb)/tb*100:+.2f} %)\n")
    print("| kernel | launches | base total ms | dev total ms | base mean us | dev mean us | delta |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for n, bc, bt, dc, dt in rows:
        if bt / 1e3 < floor and dt / 1e3 < floor:
            continue
        bm = bt / bc / 1e3 if bc else 0
        dm = dt / dc / 1e3 if dc else 0
        dl = f"{(dt-bt)/bt*100:+.1f} %" if bt else "new"
        print(f"| `{n}` | {bc:.0f} / {dc:.0f} | {bt/1e6:.3f} | {dt/1e6:.3f} | {bm:.2f} | {dm:.2f} | {dl} |")


main()
