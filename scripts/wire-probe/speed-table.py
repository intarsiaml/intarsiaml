#!/usr/bin/env python3
"""Lane wire-probe: the on/off table from a speed window's localeval logs (scripts/wire-probe/speed-window.sh).

    scripts/wire-probe/speed-table.py docs/results/2026-11-16-hip-wire-probe/raw/speed-HHMM

Per phase and prompt size: the median over all measured reps of all rounds (range in brackets) of the prefill tok/s
(prompt tokens / TTFT) and the decode tok/s, each arm, and on / off.
"""
import glob
import os
import re
import statistics
import sys
from collections import defaultdict

LINE = re.compile(r"\[localeval\]\s+(\d+) tok rep \d+/\d+: prompt=(\d+) ttft=([\d.]+)s prefill=([\d.]+) tok/s decode=([\d.]+) tok/s")
NAME = re.compile(r"localeval-(.+?)-(tp\d-(?:off|wht6|wht8))-r(\d+)\.log$")

data = defaultdict(lambda: defaultdict(lambda: {"prefill": [], "decode": [], "prompt": 0}))  # (phase, size) -> arm -> lists
for path in sorted(glob.glob(os.path.join(sys.argv[1], "localeval-*.log"))):
    m = NAME.search(path)
    if not m:
        continue
    phase, arm = m.group(1), m.group(2)
    for line in open(path, errors="replace"):
        g = LINE.search(line)
        if g:
            d = data[(phase, int(g.group(1)))][arm]
            d["prefill"].append(float(g.group(4)))
            d["decode"].append(float(g.group(5)))
            d["prompt"] = int(g.group(2))


def cell(v, digits=0):
    if not v:
        return "-"
    return f"{statistics.median(v):.{digits}f} ({min(v):.{digits}f}-{max(v):.{digits}f})"


ARMS = ["off", "wht6", "wht8"]
print("| prompt tok | n | prefill off | prefill wht6 | prefill wht8 | wht6 / off | wht8 / off | wht8 / wht6 | gain kept (wht8 - 1) / (wht6 - 1) |")
print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
dec = []
for (phase, size), arms in sorted(data.items()):
    a = {k: next((v for n, v in arms.items() if n.endswith("-" + k)), None) for k in ARMS}
    if not all(a.values()) or not all(v["prefill"] for v in a.values()):
        continue
    m = {k: statistics.median(v["prefill"]) for k, v in a.items()}
    md = {k: statistics.median(v["decode"]) for k, v in a.items()}
    n = " / ".join(str(len(a[k]["prefill"])) for k in ARMS)
    kept = (m["wht8"] / m["off"] - 1) / (m["wht6"] / m["off"] - 1) if m["wht6"] != m["off"] else float("nan")
    print(f'| {a["off"]["prompt"]} | {n} | {cell(a["off"]["prefill"])} | {cell(a["wht6"]["prefill"])} | {cell(a["wht8"]["prefill"])} | {m["wht6"] / m["off"]:.3f} | {m["wht8"] / m["off"]:.3f} | {m["wht8"] / m["wht6"]:.3f} | {kept:.2f} |')
    dec.append((a["off"]["prompt"], a, md))
print()
print("| prompt tok | decode off | decode wht6 | decode wht8 | wht6 / off | wht8 / off |")
print("|---:|---:|---:|---:|---:|---:|")
for p, a, md in dec:
    print(f'| {p} | {cell(a["off"]["decode"], 1)} | {cell(a["wht6"]["decode"], 1)} | {cell(a["wht8"]["decode"], 1)} | {md["wht6"] / md["off"]:.3f} | {md["wht8"] / md["off"]:.3f} |')
