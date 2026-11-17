#!/usr/bin/env python3
"""Lane mfp8split: per group and M, from the timing logs: the old halves (explicit plan), the plan under test, the best other split,
per run. usage: tsum.py run*.log   (each log: CHUNKX group=G m=M plan=P us=U lines)"""
import re, sys, collections
runs = []
for f in sys.argv[1:]:
    d = collections.defaultdict(dict)
    for l in open(f, errors="replace"):
        m = re.match(r"CHUNKX group=(\S+) m=(\d+) plan=(\S+) us=([\d.]+)", l)
        if m:
            d[(m.group(1), int(m.group(2)))][m.group(3)] = float(m.group(4))
    runs.append(d)
PLAN = {9: "7+2", 10: "8+2", 11: "8+3", 12: "8+4"}
HALF = {9: "5+4", 10: "5+5", 11: "6+5", 12: "6+6", 13: "7+6"}
keys = sorted(set(k for r in runs for k in r))
print("| group | M | halves us (per run) | plan us (per run) | plan / halves (per run) | best split (min over runs, / halves) |")
print("|---|---|---|---|---|---|")
for g, m in keys:
    if m < 9:
        continue
    h = [r[(g, m)][HALF[m]] for r in runs if (g, m) in r and HALF[m] in r[(g, m)]]
    p = [r[(g, m)][PLAN[m]] for r in runs if (g, m) in r and PLAN.get(m) in r[(g, m)]] if m in PLAN else []
    allp = collections.defaultdict(list)
    for r in runs:
        for pl, u in r.get((g, m), {}).items():
            if pl != "cur":
                allp[pl].append(u)
    best = min(allp.items(), key=lambda kv: min(kv[1]))
    hm = min(h)
    print(f"| {g} | {m} | {' / '.join(f'{x:.0f}' for x in h)} | {' / '.join(f'{x:.0f}' for x in p)} | "
          f"{' / '.join(f'{a/b:.3f}' for a, b in zip(p, h))} | {best[0]} {min(best[1]):.0f} ({min(best[1])/hm:.3f}) |")
