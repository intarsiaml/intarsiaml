#!/usr/bin/env python3
"""usage: wclock.py <macmon.jsonl> <start_epoch> <end_epoch>: clock over the busy samples (util > 0.5) of a window, whole window.
Prints 'median_MHz p10_MHz pct_below_1450 median_W max_C n'; exit 0 if median >= 1450 and < 40 percent of samples below 1450."""
import sys, json, datetime, statistics as st
mmf, a, b = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
lo, hi = a, b
w = []
for l in open(mmf).read().splitlines()[-6000:]:
    try:
        j = json.loads(l); t = datetime.datetime.fromisoformat(j["timestamp"]).timestamp()
        if lo <= t <= hi and j["gpu_usage"][1] > 0.3: w.append((j["gpu_usage"][0], j["gpu_power"], j["temp"]["gpu_temp_avg"]))
    except Exception: pass
if len(w) < 3: print("few-samples"); sys.exit(1)
f = sorted(x[0] for x in w); med = st.median(f); low = 100 * sum(1 for x in f if x < 1450) / len(f)
print(f"{med:.0f} {f[len(f)//10]:.0f} {low:.0f} {st.median(x[1] for x in w):.0f} {max(x[2] for x in w):.0f} {len(w)}")
sys.exit(0 if med >= 1450 and low < 40 else 1)
