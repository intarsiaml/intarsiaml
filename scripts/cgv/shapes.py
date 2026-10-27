#!/usr/bin/env python3
"""Lane cgv: per launch-shape table of the MXFP4 GEMV kernels of one decode nsys trace (cuda_gpu_trace csv).
    shapes.py <trace.csv> <decode_wall_seconds> [steps]      (the decode window is the last <wall> seconds of kernels)
Grid X identifies N (one-row kernel: N/8 blocks; multi-row kernel: N/(8R)); prints calls, median / mean us and total ms per (kernel, grid)."""
import csv, collections, statistics as st, sys
rows = list(csv.DictReader(open(sys.argv[1])))
ev = sorted(((float(r['Start (ns)']), float(r['Duration (ns)']), r) for r in rows if not (r.get('Name') or '').startswith('[CUDA')), key=lambda e: e[0])
cut = max(e[0] + e[1] for e in ev) - float(sys.argv[2]) * 1e9
dec = [e for e in ev if e[0] >= cut]
g = collections.defaultdict(list)
for s, d, r in dec:
    n = r['Name']
    if 'mxfp4_gemv' in n: g[(('mr' if '_mr_' in n else 'gemv'), int(r['GrdX']))].append(d / 1000)
tot = sum(sum(v) for v in g.values()) / 1000
print(f"decode kernels {len(dec)}; mxfp4 GEMV total {tot:.1f} ms")
print("kernel  grid.x  calls  median_us  mean_us  total_ms")
for k, v in sorted(g.items(), key=lambda kv: -sum(kv[1])):
    print(f"{k[0]:6s} {k[1]:6d} {len(v):6d} {st.median(v):9.1f} {sum(v)/len(v):8.1f} {sum(v)/1000:9.1f}")
