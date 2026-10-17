#!/usr/bin/env python3
"""GPU clock (macmon) per arm/round window of a tools/cbm3.py slot: m3v_clock.py DIR  (DIR has m3-<arm>-<r>.json, macmon.jsonl[.gz])."""
import datetime, glob, gzip, json, os, re, statistics, sys
d = sys.argv[1]
mp = os.path.join(d, "macmon.jsonl")
f = open(mp) if os.path.exists(mp) else gzip.open(mp + ".gz", "rt")
ts, mhz, use = [], [], []
for line in f:
    try:
        j = json.loads(line)
    except ValueError:
        continue
    ts.append(datetime.datetime.fromisoformat(j["timestamp"]).timestamp())
    mhz.append(j["gpu_usage"][0])
    use.append(j["gpu_usage"][1])
allbusy = [m for m, u in zip(mhz, use) if u > 0.5]
print(f"{d}: whole slot, GPU busy samples (usage > 50%): median {statistics.median(allbusy):.0f} MHz, p10 {sorted(allbusy)[len(allbusy)//10]:.0f}, p90 {sorted(allbusy)[9*len(allbusy)//10]:.0f}, n {len(allbusy)}")
rows = []
for p in sorted(glob.glob(os.path.join(d, "m3-*-*.json"))):
    m = re.search(r"m3-(\w+)-(\d+)\.json", p)
    j = json.load(open(p))
    # per scenario windows
    for sc, v in j["scenarios"].items():
        reqs = v["reqs"]
        lo = min(r["t_send"] for r in reqs); hi = max(r["t_end"] for r in reqs if r.get("t_end"))
        w = [mh for t, mh, u in zip(ts, mhz, use) if lo <= t <= hi and u > 0.5]
        if w:
            rows.append((m.group(1), m.group(2), sc, statistics.median(w), len(w)))
print("arm round scenario: median busy GPU MHz (n samples)")
for arm in ("off", "on"):
    for sc in ("grounded-n2", "grounded-n3", "open-n2", "open-n3"):
        v = [(r[1], r[3]) for r in rows if r[0] == arm and r[2] == sc]
        if v:
            print(f"  {arm:3s} {sc:12s} " + "  ".join(f"r{a}:{b:.0f}" for a, b in v))
