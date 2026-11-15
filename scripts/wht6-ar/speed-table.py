#!/usr/bin/env python3
"""Lane wht6-ar: the on/off table from a speed window's localeval logs (scripts/wht6-ar/speed-window.sh).

    scripts/wht6-ar/speed-table.py docs/results/2026-11-14-hip-wht6-ar/raw/speed-HHMM

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
NAME = re.compile(r"localeval-(.+?)-(tp\d-(?:off|on))-r(\d+)\.log$")

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


print("| phase | prompt tok | n | prefill off | prefill on | on / off | decode off | decode on | on / off |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for (phase, size), arms in sorted(data.items()):
    off = next((v for k, v in arms.items() if k.endswith("-off")), None)
    on = next((v for k, v in arms.items() if k.endswith("-on")), None)
    if not off or not on or not off["prefill"] or not on["prefill"]:
        continue
    rp = statistics.median(on["prefill"]) / statistics.median(off["prefill"])
    rd = statistics.median(on["decode"]) / statistics.median(off["decode"])
    n = f'{len(off["prefill"])} / {len(on["prefill"])}'
    print(f'| {phase} | {off["prompt"]} | {n} | {cell(off["prefill"])} | {cell(on["prefill"])} | {rp:.3f} | {cell(off["decode"], 1)} | {cell(on["decode"], 1)} | {rd:.3f} |')
