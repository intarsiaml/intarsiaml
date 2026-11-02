#!/usr/bin/env python3
"""Lane hload: step times from the batched == solo gate (scripts/hload/tests.sh tp1 / tp2, mtest-<tp>.log), main vs branch.
  python3 steps.py <main window dir> <branch window dir>
Per tp: the 1-row step (the solo steps of the batch-16 line / 16, averaged over every batch line), and the batched step at
2, 4, 8, 16 rows, main / branch / branch - main."""
import os, re, statistics, sys

pat = re.compile(r"batch (\d+): \d+ steps bitwise in [\d.]+ s \(per step: ([\d.]+) ms batched, ([\d.]+) ms for the \d+ solo steps\)")


def read(d, tp):
    p = os.path.join(d, f"mtest-{tp}.log")
    if not os.path.exists(p):
        return None
    batched, solo = {}, []
    for ln in open(p, errors="replace"):
        m = pat.search(ln)
        if m:
            n = int(m[1])
            batched[n] = float(m[2])
            solo.append(float(m[3]) / n)
    return {"solo": statistics.mean(solo) if solo else float("nan"), **batched}


main, new = sys.argv[1], sys.argv[2]
print("| TP | rows | main ms/step | branch ms/step | branch - main |")
print("|---|---:|---:|---:|---:|")
for tp in ("tp1", "tp2"):
    a, b = read(main, tp), read(new, tp)
    if not a or not b:
        continue
    for k in ("solo", 2, 4, 8, 16):
        label = "1 (solo)" if k == "solo" else str(k)
        print(f"| {tp.upper().replace('TP', 'TP=')} | {label} | {a[k]:.2f} | {b[k]:.2f} | {b[k] - a[k]:+.2f} ({100 * (b[k] / a[k] - 1):+.1f} %) |")
