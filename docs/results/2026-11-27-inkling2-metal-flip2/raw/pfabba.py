#!/usr/bin/env python3
"""Lane ink-prologue2: prefill tok/s per prompt and arm from generate's `prefill[<name>]: N tokens, T ms` lines (files r<NN>-<arm>.err),
the median per arm, the ratio on / off of the medians, and the ratio within each ABBA block (median over blocks). pfabba.py <dir>"""
import re, statistics, sys
from pathlib import Path

d = Path(sys.argv[1])
pat = re.compile(r"^prefill\[([^\]]+)\]: (\d+) tokens, ([\d.]+) ms")
runs = []  # (index, arm, {prompt: tok/s})
for f in sorted(d.glob("r*-*.err")):
    idx, arm = f.stem.split("-", 1)
    v = {}
    for l in open(f, errors="replace"):
        m = pat.match(l)
        if m:
            v[m.group(1)] = int(m.group(2)) / (float(m.group(3)) / 1000.0)
    runs.append((int(idx[1:]), arm, v))
prompts = list(dict.fromkeys(p for _, _, v in runs for p in v))
print("| prompt | off tok/s (runs) | on tok/s (runs) | median off | median on | on / off | per-block on / off |")
print("|---|---|---|---|---|---|---|")
for p in prompts:
    off = [v[p] for _, a, v in runs if a == "off" and p in v]
    on = [v[p] for _, a, v in runs if a == "on" and p in v]
    blocks = []
    for b in range(0, len(runs), 4):
        blk = runs[b:b + 4]
        bo = [v[p] for _, a, v in blk if a == "off" and p in v]
        bn = [v[p] for _, a, v in blk if a == "on" and p in v]
        if bo and bn:
            blocks.append(statistics.mean(bn) / statistics.mean(bo))
    if not off or not on:
        print(f"| {p} | {off} | {on} | | | missing | |")
        continue
    mo, mn = statistics.median(off), statistics.median(on)
    print(f"| {p} | {' / '.join(f'{x:.0f}' for x in off)} | {' / '.join(f'{x:.0f}' for x in on)} | {mo:.1f} | {mn:.1f} | "
          f"**{mn / mo:.3f}** | {' / '.join(f'{x:.3f}' for x in blocks)} |")
