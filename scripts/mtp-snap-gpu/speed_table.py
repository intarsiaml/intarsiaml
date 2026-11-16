#!/usr/bin/env python3
"""Lane mtp-snap-gpu: main vs PR speed table from the `localeval speed` logs of speed-window.sh.
  speed_table.py DIR     (DIR holds localeval-<main|pr>-r<N>.log; prints markdown)
Cell: median (min-max) over rounds (each round's row of the localeval table is already the median over its reps). Several DIRs are pooled."""
import re, statistics, sys, pathlib

dirs = [pathlib.Path(a) for a in sys.argv[1:]]
cells = {}  # (size, metric, arm) -> [values]
SIZES = [int(x) for x in __import__("os").environ.get("SIZES", "1024,4096,16384").split(",")]  # SIZES env: the sweep of the DIR; nominal sizes of speed-window.sh; the prompt a row reports varies by a few tokens per run
for f in [f for d in dirs for f in sorted(d.glob("localeval-*-r*.log"))]:
    arm = f.name.split("-")[1]
    row = 0
    for line in f.read_text().splitlines():
        m = re.match(r"\| (\d+) \| ([\d.]+) \| (\d+) \(.*?\)\s*\*? \| (\S+) \| ([\d.]+) \(", line)
        if m:
            _, ttft, pre, marg, dec = m.groups()
            size = SIZES[row]; row += 1
            cells.setdefault((int(size), "prefill tok/s", arm), []).append(float(pre))
            cells.setdefault((int(size), "decode tok/s", arm), []).append(float(dec))
            if marg != "-":
                cells.setdefault((int(size), "marginal prefill tok/s", arm), []).append(float(marg))


def fmt(v):
    return f"{statistics.median(v):.0f} ({min(v):.0f}-{max(v):.0f})"


print("| prompt tok | metric | main median (min-max) | PR median (min-max) | PR vs main | rounds |")
print("|---:|---|---|---|---:|---:|")
for size, metric in sorted({(k[0], k[1]) for k in cells}, key=lambda x: (x[0], x[1] != "prefill tok/s", x[1])):
    a, b = cells.get((size, metric, "main")), cells.get((size, metric, "pr"))
    if a and b:
        print(f"| {size} | {metric} | {fmt(a)} | {fmt(b)} | {100 * (statistics.median(b) / statistics.median(a) - 1):+.1f}% | {len(a)}/{len(b)} |")
