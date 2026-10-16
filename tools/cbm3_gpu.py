#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""GPU clock per arm run of a cbm3_slot.sh directory: macmon.jsonl samples between consecutive arm-times.txt marks, busy samples
(GPU utilisation >= 50 %) only.   cbm3_gpu.py DIR   -> markdown table (arm, round, position, busy samples, mean / p5 MHz, mean W, max C)"""
import json
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

d = Path(sys.argv[1])
marks = []
for ln in (d / "arm-times.txt").read_text().splitlines():
    p = ln.split()
    marks.append((int(p[0]), p[2], p[4], p[6]))  # epoch, round, arm, pos
rows = []
for ln in (d / "macmon.jsonl").read_text().splitlines():
    try:
        j = json.loads(ln)
    except ValueError:
        continue
    rows.append((datetime.fromisoformat(j["timestamp"]).timestamp(), j["gpu_usage"][0], j["gpu_usage"][1] * 100,
                 j["gpu_power"], j["temp"]["gpu_temp_avg"]))
end = rows[-1][0] + 1
print("| round | arm | position | busy samples (util >= 50 %) | GPU MHz mean | p5 | GPU W mean | max temp C |")
print("|---|---|---|---|---|---|---|---|")
allb = []
for i, (t, rd, arm, pos) in enumerate(marks):
    t1 = marks[i + 1][0] if i + 1 < len(marks) else end
    b = [r for r in rows if t <= r[0] < t1 and r[2] >= 50]
    allb += b
    if b:
        f = sorted(r[1] for r in b)
        print(f"| {rd} | {arm} | {pos} | {len(b)} | {st.mean(f):.0f} | {f[int(0.05 * (len(f) - 1))]:.0f} | "
              f"{st.mean(r[3] for r in b):.1f} | {max(r[4] for r in b):.0f} |")
if allb:
    f = sorted(r[1] for r in allb)
    print(f"| all | | | {len(allb)} | {st.mean(f):.0f} | {f[int(0.05 * (len(f) - 1))]:.0f} | {st.mean(r[3] for r in allb):.1f} | {max(r[4] for r in allb):.0f} |")
