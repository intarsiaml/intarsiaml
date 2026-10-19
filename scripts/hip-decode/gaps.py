#!/usr/bin/env python3
"""Idle between kernels in a rocprofv3 kernel_trace.csv, attributed to the kernel pair around each gap.
usage: gaps.py kernel_trace.csv [min_gap_us=0] [top=25]
Uses the longest burst of back-to-back launches (gaps < 5 ms), as analyze_prefill.py does. Prints the gap histogram and the
summed gap time per (previous kernel -> next kernel) pair, so host-side stalls (syncs, copies) show up by where they happen."""
import collections
import csv
import re
import sys

path = sys.argv[1]
min_gap = float(sys.argv[2]) * 1e3 if len(sys.argv) > 2 else 0.0
top = int(sys.argv[3]) if len(sys.argv) > 3 else 25
rows = []
with open(path) as f:
    for r in csv.DictReader(f):
        rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]))
rows.sort()


def short(n):
    n = re.sub(r"^void ", "", n.replace("(anonymous namespace)::", ""))
    d = 0
    for i, ch in enumerate(n):
        if ch == "<":
            d += 1
        elif ch == ">":
            d -= 1
        elif ch == "(" and d == 0:
            n = n[:i]
            break
    return n[:48]


bursts, cur = [], [rows[0]]
for a, b in zip(rows, rows[1:]):
    if b[0] - a[1] > 5_000_000:
        bursts.append(cur)
        cur = []
    cur.append(b)
bursts.append(cur)
main = max(bursts, key=lambda s: s[-1][1] - s[0][0])
if len(sys.argv) > 4 and sys.argv[4] == "prefill":  # only the launches up to the first argmax (the prompt's prefill)
    cut = next((i for i, r in enumerate(main) if "argmax" in r[2]), len(main))
    main = main[:cut]
span = main[-1][1] - main[0][0]
busy = sum(e - s for s, e, _ in main)
print(f"burst: {len(main)} kernels, span {span / 1e6:.2f} ms, kernel {busy / 1e6:.2f} ms, idle {(span - busy) / 1e6:.2f} ms")
gaps = [(b[0] - a[1], short(a[2]), short(b[2]), i) for i, (a, b) in enumerate(zip(main, main[1:]))]
hist = collections.Counter()
for g, *_ in gaps:
    k = "<0" if g < 0 else "<5us" if g < 5e3 else "<20us" if g < 2e4 else "<100us" if g < 1e5 else "<1ms" if g < 1e6 else ">=1ms"
    hist[k] += 1
tot = collections.Counter()
for g, *_ in gaps:
    k = "<0" if g < 0 else "<5us" if g < 5e3 else "<20us" if g < 2e4 else "<100us" if g < 1e5 else "<1ms" if g < 1e6 else ">=1ms"
    tot[k] += max(g, 0)
for k in ["<0", "<5us", "<20us", "<100us", "<1ms", ">=1ms"]:
    print(f"  gaps {k:>7}: {hist[k]:6d}  sum {tot[k] / 1e6:8.2f} ms")
pair = collections.defaultdict(lambda: [0, 0])
for g, a, b, _ in gaps:
    if g >= min_gap:
        pair[(a, b)][0] += 1
        pair[(a, b)][1] += max(g, 0)
print(f"\ngap time by (previous -> next) kernel, gaps >= {min_gap / 1e3:.0f} us:")
for (a, b), (n, t) in sorted(pair.items(), key=lambda kv: -kv[1][1])[:top]:
    print(f"  {t / 1e6:8.3f} ms  n {n:5d}  avg {t / n / 1e3:8.1f} us  {a} -> {b}")
print("\nlargest gaps:")
for g, a, b, i in sorted(gaps, reverse=True)[:12]:
    print(f"  {g / 1e3:9.1f} us  #{i}  {a} -> {b}")
