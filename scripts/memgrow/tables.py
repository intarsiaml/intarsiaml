#!/usr/bin/env python3
"""memgrow: markdown 10-minute tables for the results doc. usage: tables.py <raw-dir> <tag>...  (stdlib only; reuses analyze.py's loader)"""
import sys, pathlib, math
src = open(pathlib.Path(__file__).with_name("analyze.py")).read().replace("\nmain()\n", "\n")
exec(src)
raw = pathlib.Path(sys.argv[1])
for tag in sys.argv[2:]:
    d, m = load(raw, tag)
    tl = m["load"]; te = m.get("idle", d[-1]["epoch"])
    print(f"\n**Run {tag}** (load phase, mean of the 30 s samples in each 10 min; MB)\n")
    print("| min of load | MemAvailable | engine GPU (nvidia-smi) | engine RssAnon | host tier | RssAnon minus host tier | RssAnon max in window | requests (cum.) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    t = tl
    while t < te:
        e = min(t + 600, te)
        rows = [r for r in d if t <= r["epoch"] < e]
        if rows:
            f = lambda k: sum(r[k] for r in rows) / len(rows)
            print(f"| {(t - tl) / 60:.0f} to {(e - tl) / 60:.0f} | {f('mi_MemAvailable'):.0f} | {f('gpu_mb_intarsia'):.0f} | {f('st_RssAnon'):.0f} | {f('host_tier_mb'):.0f} | {f('st_RssAnon') - f('host_tier_mb'):.0f} | {max(r['st_RssAnon'] for r in rows):.0f} | {rows[-1]['requests_total']:.0f} |")
        t = e
