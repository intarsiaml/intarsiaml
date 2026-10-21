#!/usr/bin/env python3
"""gate.py <ab-dir>: per plain-generate run, decode tok/s with the macmon medians (GPU MHz, GPU busy, CPU load) over its
decode window (end - decode_ms .. end - 0.15 s; G3's clock.py). A run counts under G3's state-admission rule only when
the GPU clock is >= 1300 MHz and the CPU load <= 0.30. Prints the table and the per-arm means (all runs, admitted runs)."""
import datetime as dt, gzip, json, os, re, statistics as st, sys

D = sys.argv[1]
MHZ, CPU = 1300, 0.30
rows = []
for line in open(os.path.join(D, "times.txt")):
    name, end = line.split()[:2]
    m = re.search(r"decode (\d+) ms \(([\d.]+) tok/s\)", open(os.path.join(D, f"{name}.err")).read())
    if not m:
        continue
    dms, tps = int(m.group(1)), float(m.group(2))
    t0, t1 = float(end) - dms / 1e3 - 0.1, float(end) - 0.15  # times.txt is stamped when the run returns
    s = []
    for l in gzip.open(os.path.join(D, f"{name}-macmon.jsonl.gz"), "rt"):
        try:
            d = json.loads(l)
        except Exception:
            continue
        t = dt.datetime.fromisoformat(d["timestamp"]).timestamp()
        if t0 <= t <= t1:
            s.append((d["gpu_usage"][0], d["gpu_usage"][1], d["cpu_usage_pct"]))
    if not s:
        continue
    mhz, busy, cpu = (st.median(x[i] for x in s) for i in range(3))
    rows.append((name, tps, mhz, busy, cpu, len(s), mhz >= MHZ and cpu <= CPU))
print(f"{'run':10} {'tok/s':>7} {'MHz':>6} {'busy':>5} {'cpu':>5} {'n':>4} admitted")
for r in rows:
    print(f"{r[0]:10} {r[1]:7.2f} {r[2]:6.0f} {r[3]:5.2f} {r[4]:5.2f} {r[5]:4d} {'yes' if r[6] else 'no'}")
for arm in ("base", "fb"):
    a = [r for r in rows if r[0].startswith(arm + "-")]
    ok = [r for r in a if r[6]]
    mean = lambda x: f"{st.mean(r[1] for r in x):.2f}" if x else "n/a"
    print(f"{arm}: all {len(a)} runs mean {mean(a)} tok/s; admitted {len(ok)} mean {mean(ok)}")
