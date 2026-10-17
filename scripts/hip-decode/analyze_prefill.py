#!/usr/bin/env python3
"""Per-kernel table of one rocprofv3 kernel_trace.csv of a prefill run (generate --max-tokens 1).

usage: analyze_prefill.py kernel_trace.csv

Prints, per kernel name: launches, total ms, average us, share of kernel time, first launch's grid x workgroup, and the
span / idle of the run. The trace holds the whole process (weight upload helpers, a warm-up-free single prefill), so the
span is first-to-last kernel start/end of the biggest contiguous burst: gaps over 50 ms split bursts."""
import collections
import csv
import re
import sys

rows = []
with open(sys.argv[1]) as f:
    for r in csv.DictReader(f):
        rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"],
                     "%sx%sx%s" % (r.get("Grid_Size_X", ""), r.get("Grid_Size_Y", ""), r.get("Grid_Size_Z", "")),
                     r.get("Workgroup_Size_X", "")))
rows.sort()


def short(n):
    n = n.replace("(anonymous namespace)::", "")
    n = re.sub(r"^void ", "", n)
    d, out = 0, n
    for i, ch in enumerate(n):
        if ch == "<":
            d += 1
        elif ch == ">":
            d -= 1
        elif ch == "(" and d == 0:
            out = n[:i]
            break
    return out.replace("ik_bf16", "bf16")[:78]


bursts, cur = [], [rows[0]]
for a, b in zip(rows, rows[1:]):
    if b[0] - a[1] > 50_000_000:
        bursts.append(cur)
        cur = []
    cur.append(b)
bursts.append(cur)
main = max(bursts, key=lambda s: s[-1][1] - s[0][0])
span = main[-1][1] - main[0][0]
tot = sum(e - s for s, e, *_ in main)
print(f"kernels in main burst: {len(main)} (of {len(rows)}), span {span/1e6:.1f} ms, kernel time {tot/1e6:.1f} ms, idle {100*(span-tot)/span:.1f} %")
agg = collections.OrderedDict()
for s, e, n, g, w in main:
    k = short(n)
    a = agg.setdefault(k, [0, 0, g, w])
    a[0] += 1
    a[1] += e - s
print(f"\n{'kernel':<80}{'n':>6}{'ms':>9}{'avg us':>9}{'%kern':>7}  grid x wg")
for k, (c, t, g, w) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    print(f"{k:<80}{c:>6}{t/1e6:>9.2f}{t/c/1e3:>9.1f}{100*t/tot:>6.1f}%  {g} x {w}")
