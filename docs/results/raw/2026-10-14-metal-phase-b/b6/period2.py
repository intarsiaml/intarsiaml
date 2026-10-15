#!/usr/bin/env python3
"""period2.py <cb-trace.jsonl> <dispatches-per-step> <steps> [window=16]: robust step period (ms/step) from a cb trace.

period.py (b2b) divides the whole span by the step count, so one stall (display, another process) moves the mean. Here the
last <steps> steps are cut into windows of <window> steps by cumulative dispatch count; each window gives span/steps, and
the median, min and mean over the windows are printed (a window boundary lands on a command-buffer boundary, so one window
carries up to +-1 command buffer of error that cancels in the median)."""
import json, sys, statistics

f, d, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
w = int(sys.argv[4]) if len(sys.argv) > 4 else 16
c = [json.loads(l) for l in open(f)]
acc, j = 0, len(c)
while acc < d * n and j > 0:
    j -= 1
    acc += c[j]["nd"]
s = c[j:]
# windows by cumulative dispatches
per = []
cum = 0
start = s[0]["s"]
nd0 = 0
target = d * w
for r in s:
    cum += r["nd"]
    if cum - nd0 >= target:
        per.append((r["e"] - start) / ((cum - nd0) / d) * 1e3)
        start = r["e"]
        nd0 = cum
if not per:
    per = [(s[-1]["e"] - s[0]["s"]) / n * 1e3]
print(f"{statistics.median(per):.3f} {min(per):.3f} {statistics.fmean(per):.3f}")
