#!/usr/bin/env python3
"""Lane pf: tabulate scripts/pf/acc.sh output (acc-<arm>-<golden>.log, reuse-<arm>-c<chunk>.log) into markdown."""
import glob, os, re, sys
d = sys.argv[1]
print("| arm | golden | prefill top-1 % | prefill KL21 scored rows | threshold verdict | decode KL21 scored rows |"); print("|---|---|---|---|---|---|")
for f in sorted(glob.glob(f"{d}/acc-*.log")):
    m = re.match(r"acc-(.+)-(short|long|4k)\.log", os.path.basename(f)); arm, g = m.groups()
    t = open(f, errors="replace").read()
    top1 = re.search(r"OVERALL\s+\d+\s+([\d.]+)", t)
    pk = re.search(r"prefill KL threshold.*?KL ([\d.e+-]+) (PASS|FAIL)", t)
    dk = re.search(r"decode KL threshold.*?KL ([\d.e+-]+) (PASS|FAIL)", t)
    print(f"| {arm} | {g} | {top1.group(1) if top1 else '-'} | {pk.group(1) if pk else '-'} | {pk.group(2) if pk else '-'} | {dk.group(1) + ' ' + dk.group(2) if dk else '-'} |")
print(); print("| arm | reuse chunk | rows judged | failing rows (prompt/case: why) | gate |"); print("|---|---|---|---|---|")
for f in sorted(glob.glob(f"{d}/reuse-*.log")):
    m = re.match(r"reuse-(.+)-c(\d+)\.log", os.path.basename(f)); arm, c = m.groups()
    t = open(f, errors="replace").read()
    rows = re.findall(r"^(\w+_(?:4k|12k))\s+(continue|restore|aligned)\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+/\d+)\s+(ok|FAIL)(?: \((.{0,70}))?", t, re.M)
    bad = [f"{p}/{k} {a}: {(w or '').strip()[:60]}" for p, k, a, r, w in rows if r == "FAIL"]
    g = re.search(r"prefix-reuse gate: (\w+)", t)
    print(f"| {arm} | {c} | {len(rows)} | {'; '.join(bad) or 'none'} | {g.group(1) if g else '-'} |")
