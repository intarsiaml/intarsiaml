#!/usr/bin/env python3
"""storeprefault: tables from the raw dir (stdlib only).  usage: tables.py RAW_DIR [window-prefix ...]  > tables.md
Reads <window>-v<k>-<arm>-speed.out (localeval speed lines) and <window>-v<k>-<arm>-gap*.out (ttft_gap.py lines).
For every (window, arm) prints the TTFT median of the reps per visit and the arm mean over visits, in ABBA order of visit number,
and the ratio of each arm's mean to the arm named by BASE (default: off)."""
import pathlib, re, statistics, sys, os

RAW = pathlib.Path(sys.argv[1])
WINDOWS = sys.argv[2:] or None
BASE = os.environ.get("BASE", "off")
SIZES = (4096, 16384, 32768)
LINE = re.compile(r"\]\s+(\d+) tok rep (\d+)/\d+: prompt=(\d+) ttft=([\d.]+)s prefill=(\d+) tok/s decode=([\d.]+)")
GAP = re.compile(r"\[gap\] \S+ gap=([\d.]+) (\d+) rep(\d+): prompt=(\d+) cached=(\d+) ttft=([\d.]+)s")

visits = {}  # (window, k, arm) -> {size: [ttft]}
for p in sorted(RAW.glob("*-speed.out")):
    m = re.match(r"(w\d+)-v(\d+)-(\w+)-speed\.out", p.name)
    if not m or (WINDOWS and m.group(1) not in WINDOWS):
        continue
    d = {}
    for l in p.read_text(errors="replace").splitlines():
        x = LINE.search(l)
        if x:
            d.setdefault(int(x.group(1)), []).append((float(x.group(4)), int(x.group(5)), float(x.group(6))))
    visits[(m.group(1), int(m.group(2)), m.group(3))] = d

arms = {}
for (w, k, arm), d in sorted(visits.items(), key=lambda kv: (kv[0][0], kv[0][1])):
    arms.setdefault(arm, []).append((w, k, d))

def med(v):
    return statistics.median(v) if v else None

print("### TTFT s, median of 3 reps per visit (visit number in the window in brackets)\n")
print("| arm | visits | " + " | ".join(f"{s}" for s in SIZES) + " |")
print("|---|---|" + "---|" * len(SIZES))
mean = {}
for arm, vs in arms.items():
    cells = []
    for s in SIZES:
        per = [med([t[0] for t in d.get(s, [])]) for (_, _, d) in vs]
        per = [x for x in per if x is not None]
        mean.setdefault(arm, {})[s] = statistics.mean(per) if per else None
        cells.append(" / ".join(f"{x:.2f}" for x in per) + (f" (mean {statistics.mean(per):.3f})" if per else ""))
    print(f"| {arm} | " + ", ".join(f"{w}v{k}" for (w, k, _) in vs) + " | " + " | ".join(cells) + " |")

print(f"\n### Arm mean TTFT / `{BASE}` mean (1.00 = same; >1 slower)\n")
print("| arm | " + " | ".join(f"{s}" for s in SIZES) + " |")
print("|---|" + "---|" * len(SIZES))
for arm in arms:
    b = mean.get(BASE, {})
    print(f"| {arm} | " + " | ".join((f"{mean[arm][s] / b[s]:.3f}" if mean[arm].get(s) and b.get(s) else "-") for s in SIZES) + " |")

print("\n### Prefill tok/s (uncached prompt tokens / TTFT), arm mean of the visit medians\n")
print("| arm | " + " | ".join(f"{s}" for s in SIZES) + " |")
print("|---|" + "---|" * len(SIZES))
for arm, vs in arms.items():
    cells = []
    for s in SIZES:
        per = [med([t[1] for t in d.get(s, [])]) for (_, _, d) in vs]
        per = [x for x in per if x is not None]
        cells.append(f"{statistics.mean(per):.0f}" if per else "-")
    print(f"| {arm} | " + " | ".join(cells) + " |")

print("\n### Decode tok/s, arm mean of the visit medians\n")
print("| arm | " + " | ".join(f"{s}" for s in SIZES) + " |")
print("|---|" + "---|" * len(SIZES))
for arm, vs in arms.items():
    cells = []
    for s in SIZES:
        per = [med([t[2] for t in d.get(s, [])]) for (_, _, d) in vs]
        per = [x for x in per if x is not None]
        cells.append(f"{statistics.mean(per):.1f}" if per else "-")
    print(f"| {arm} | " + " | ".join(cells) + " |")

gaps = {}
for p in sorted(RAW.glob("*-gap*.out")):
    m = re.match(r"(w\d+)-v(\d+)-(\w+?)-gap(\d*)\.out", p.name)
    if not m or (WINDOWS and m.group(1) not in WINDOWS):
        continue
    for l in p.read_text(errors="replace").splitlines():
        x = GAP.search(l)
        if x and int(x.group(3)) > 0:
            gaps.setdefault((m.group(3), float(x.group(1))), {}).setdefault((m.group(1), int(m.group(2)), int(x.group(2))), []).append(float(x.group(6)))
if gaps:
    print("\n### TTFT s with an idle gap between requests (ttft_gap.py): median of the reps, mean over visits\n")
    print("| arm | gap s | " + " | ".join(f"{s}" for s in SIZES) + " |")
    print("|---|---|" + "---|" * len(SIZES))
    for (arm, g), d in sorted(gaps.items()):
        cells = []
        for s in SIZES:
            per = [med(v) for (w, k, sz), v in d.items() if sz == s or abs(sz - s) < 100]
            cells.append(f"{statistics.mean(per):.3f} (n={len(per)})" if per else "-")
        print(f"| {arm} | {g:g} | " + " | ".join(cells) + " |")
