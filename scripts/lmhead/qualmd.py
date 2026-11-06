#!/usr/bin/env python3
"""Markdown KL / top-1 table from `intarsia accuracy --self-golden` outputs: usage qualmd.py LABEL=file.out [LABEL=file.out ...]
Rows: arm x mode x prompt group (4k = 4k-*, 16k = 16k-*); mean KL(21-way) and top-1 flips are position weighted over the group's prompts."""
import re, sys
print("| arm | mode | prompts | positions | mean KL(21-way) | top-1 agreement | top-1 flips | max position KL |\n|---|---|---|---|---|---|---|---|")
for spec in sys.argv[1:]:
    label, path = spec.split("=", 1); mode = None; rows = {}
    for l in open(path, errors="replace"):
        m = re.match(r"== mode: (\w+)", l)
        if m: mode = m.group(1); continue
        m = re.match(r"(4k|16k)-(\d)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", l)
        if m and mode:
            g, _, n, pos, top1, kls, kl21, _, maxkl = m.groups()
            rows.setdefault((mode, g), []).append((int(pos), float(top1), float(kl21), float(maxkl)))
    for (mode, g), v in rows.items():
        P = sum(p for p, *_ in v); kl = sum(p * k for p, _, k, _ in v) / P; t1 = sum(p * t for p, t, *_ in v) / P
        print(f"| {label} | {mode.split()[0]} | {g} ({len(v)}) | {P} | {kl:.3e} | {t1:.2f} % | {100-t1:.2f} % | {max(x[3] for x in v):.3e} |")
