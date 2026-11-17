#!/usr/bin/env python3
"""Lane mchunkx: markdown tables for the results doc from the CHUNKX logs (min over runs). usage: ttables.py run1.log run2.log"""
import re, sys
d = {}
for f in sys.argv[1:]:
    for l in open(f, errors="replace"):
        m = re.match(r"CHUNKX group=(\S+) m=(\d+) plan=(\S+) us=([\d.]+)", l)
        if m:
            k = (m.group(1), int(m.group(2)), m.group(3))
            d[k] = min(d.get(k, 1e18), float(m.group(4)))
groups = ["fp8_35b", "nvfp4_35b", "nvfp4_head_35b", "pair_35b", "fp8_head_27b", "bf16_35bshapes", "bf16_27bshapes", "logits_bf16_35b", "logits_bf16_27b"]
print("### A. One launch of n rows: us per decode step of the kernel (us per row in brackets)\n")
print("| group | " + " | ".join(f"n={n}" for n in range(2, 9)) + " |")
print("|---|" + "---:|" * 7)
for g in groups:
    cells = []
    for n in range(2, 9):
        u = d.get((g, n, str(n)))
        cells.append(f"{u:.0f} ({u/n:.0f})" if u else "-")
    print(f"| {g} | " + " | ".join(cells) + " |")
print("\n### B. Today's rule against the best split, M = 5..16 (us per step of the kernel; today = single launch up to 8 rows, halves above)\n")
print("| group | " + " | ".join(f"M={m}" for m in range(5, 17)) + " |")
print("|---|" + "---:|" * 12)
for g in groups:
    cells = []
    for m in range(5, 17):
        cp = str(m) if m <= 8 else f"{(m+1)//2}+{m//2}"
        today = min(d.get((g, m, "cur"), 1e18), d.get((g, m, cp), 1e18))
        alts = sorted((u, p) for (gg, mm, p), u in d.items() if gg == g and mm == m and p not in ("cur", cp))
        if not alts:
            cells.append("-"); continue
        u, p = alts[0]
        cells.append(f"{today:.0f} / {u:.0f} ({p}, {u/today:.2f})")
    print(f"| {g} | " + " | ".join(cells) + " |")
