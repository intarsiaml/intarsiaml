#!/usr/bin/env python3
"""Lane moedn: per-kernel mean durations of the MoE decode kernels in an nsys cuda_gpu_trace csv of bench_moe_rows_cold,
by kernel (with template args) and token count T (from the preceding gate_up launch's grid x = T * 8).
  mk.py trace.csv"""
import csv, re, sys
rows = sorted(csv.DictReader(open(sys.argv[1])), key=lambda r: float(r["Start (ns)"]))
g, t = {}, None
for r in rows:
    n = r.get("Name") or ""
    if "moe_dec_gate_up" in n:
        t = int(r["GrdX"]) // 8
    if "moe_dec_" not in n: continue
    k = (re.sub(r"\(.*", "", n.replace("void ", "").replace("(int)", "")).replace("(anonymous namespace)::", "").replace("<unnamed>::", "")[:40], t, f'{r["GrdX"]}x{r["GrdY"]}')
    a = g.setdefault(k, [0, 0.0]); a[0] += 1; a[1] += float(r["Duration (ns)"])
for k, (c, s) in sorted(g.items(), key=lambda x: (x[0][1], x[0][0])):
    print(f"T={k[1]:<3} {k[0]:<40} grid {k[2]:<10} {c:6d} calls {s/c/1e3:8.1f} us")
