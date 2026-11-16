#!/usr/bin/env python3
"""Lane mtp2s: F / B ratios from the itl2s analyze.py output (raw/abba-analysis.md). usage: abba_summary.py <analysis.md>
Client table: combined tok/s per cell and arm (median pair). Server table: per-visit emission gap p50, 2-seq verify pass p50, draft (prepare) p50 (ms), averaged
over the arm's visits, per cell. Prints one line per cell and the geometric mean of the tok/s ratios for 2-stream and 1-stream cells."""
import math, re, sys, collections
rows = [l.strip().strip("|").split("|") for l in open(sys.argv[1]) if l.startswith("|")]
tps = {}
for r in rows:
    r = [c.strip() for c in r]
    if len(r) >= 4 and r[1] in ("B", "F") and re.match(r"^[\d.]+ \(", r[3]):
        tps[(r[0], r[1])] = float(r[3].split()[0])
srv = collections.defaultdict(list)
for r in rows:
    r = [c.strip() for c in r]
    if len(r) >= 9 and re.match(r"^[BF]\.[ab]$", r[0]) and re.match(r"^[\d.]+ / ", r[3]):
        gap = float(r[3].split("/")[0]); ver = r[5].split("/")[0].strip(); dr = float(r[6].split("/")[0])
        srv[(r[1], r[0][0])].append((gap, float(ver) if ver != "-" else None, dr))
g = collections.defaultdict(list)
print(f"{'cell':18s} {'B tok/s':>8s} {'F tok/s':>8s} {'F/B':>6s} | iter ms B -> F | verify ms B -> F")
for (cell, arm), v in sorted(tps.items()):
    if arm != "B": continue
    f = tps.get((cell, "F")); ratio = f / v
    g[cell.split()[-1]].append(ratio)
    m = lambda a, i: (sum(x[i] for x in a) / len(a)) if a and all(x[i] is not None for x in a) else float("nan")
    key = cell
    b, ff = srv.get((key, "B"), []), srv.get((key, "F"), [])
    print(f"{cell:18s} {v:8.1f} {f:8.1f} {ratio:6.3f} | {m(b,0):5.1f} -> {m(ff,0):5.1f} | {m(b,1):5.1f} -> {m(ff,1):5.1f}")
for k, v in g.items():
    print(f"geomean F/B {k}: {math.exp(sum(map(math.log, v)) / len(v)):.4f} over {len(v)} cells (min {min(v):.3f}, max {max(v):.3f})")
