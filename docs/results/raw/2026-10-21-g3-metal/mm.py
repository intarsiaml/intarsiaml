#!/usr/bin/env python3
"""mm.py <macmon.jsonl[.gz]>...: GPU clock / power / temp summary over samples with GPU utilisation >= 50 %."""
import gzip, json, sys, statistics as st

for f in sys.argv[1:]:
    op = gzip.open if f.endswith(".gz") else open
    rows = []
    for l in op(f, "rt"):
        try:
            d = json.loads(l)
        except Exception:
            continue
        mhz, util = d["gpu_usage"]
        rows.append((mhz, util, d.get("gpu_power", 0), d.get("gpu_ram_power", 0), d["temp"]["gpu_temp_avg"], d.get("cpu_usage_pct", 0)))
    busy = [r for r in rows if r[1] >= 0.5]
    if not busy:
        print(f"{f}: {len(rows)} samples, none busy"); continue
    q = lambda i: st.quantiles([r[i] for r in busy], n=10)
    md = lambda i: st.median([r[i] for r in busy])
    print(f"{f.split('/')[-1]}: n={len(busy)}/{len(rows)} MHz med {md(0):.0f} p10 {q(0)[0]:.0f} p90 {q(0)[-1]:.0f} | gpuW {md(2):.1f} ramW {md(3):.2f} | T {md(4):.0f}C max {max(r[4] for r in busy):.0f} | cpu {md(5):.2f}")
