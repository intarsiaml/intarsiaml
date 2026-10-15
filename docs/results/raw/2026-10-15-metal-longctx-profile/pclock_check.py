#!/usr/bin/env python3
"""usage: pclock_check.py <macmon.jsonl> <start_epoch> <end_epoch> <run.err>
Median GPU clock over the prefill window of a generate run (process end - decode time - 0.3 s, back by the prefill time, inner 90%).
Prints 'median_MHz p10_MHz pct_below_1450 median_W max_C'; exit 0 if median >= 1450 and fewer than 40% of busy samples (util > 0.5) are < 1450, else 1."""
import sys, json, re, datetime, statistics as st
mmf, a, b, err = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
s = open(err).read()
m = re.search(r"prefill (\d+) ms \(([\d.]+) tok/s\) \| decode (\d+) ms", s)
if not m: print("no-result"); sys.exit(1)
pf, dc = int(m.group(1)) / 1e3, int(m.group(3)) / 1e3
pe = b - dc - 0.3
lo, hi = pe - pf + 0.05 * pf, pe - 0.05 * pf
w = []
for l in open(mmf).read().splitlines()[-3000:]:
    try:
        j = json.loads(l); t = datetime.datetime.fromisoformat(j["timestamp"]).timestamp()
        if lo <= t <= hi and j["gpu_usage"][1] > 0.5: w.append((j["gpu_usage"][0], j["gpu_power"], j["temp"]["gpu_temp_avg"]))
    except Exception: pass
if len(w) < 3: print("few-samples"); sys.exit(1)
f = sorted(x[0] for x in w); med = st.median(f); low = 100 * sum(1 for x in f if x < 1450) / len(f)
print(f"{med:.0f} {f[len(f)//10]:.0f} {low:.0f} {st.median(x[1] for x in w):.0f} {max(x[2] for x in w):.0f}")
sys.exit(0 if med >= 1450 and low < 40 else 1)
