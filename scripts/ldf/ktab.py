#!/usr/bin/env python3
"""Lane LDF: rocprofv3 kernel_trace.csv.gz (timing_a8_tp2 at one M) -> median duration (us) of the a8 GEMM, the activation quantisation and the weight
span kernels per grid size (one grid size per shape). usage: ktab.py <csv.gz> [...]"""
import csv, gzip, re, sys, statistics, collections
for f in sys.argv[1:]:
    rows = list(csv.DictReader(gzip.open(f, "rt")))
    print(f"== {f}: {len(rows)} dispatches; columns {list(rows[0].keys())[:14]}" if rows else f"== {f}: empty")
    g = collections.defaultdict(list)
    for r in rows:
        k = r["Kernel_Name"]
        cls = "gemm" if ("a8_gemm_kernel" in k or "a8_line_kernel" in k) else "quant" if "mx_a8_quant" in k else "span" if "a8_wspan" in k else None
        if cls is None: continue
        if cls == "gemm" and ("<false" in k or "Lb0" in k): continue  # the per-group launch exits at once
        gx = r.get("Grid_Size_X") or r.get("Grid_Size") or "?"
        gy = r.get("Grid_Size_Y", "")
        dur = (int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1000.0
        g[(cls, gx, gy, re.sub(r".*(a8_\w+_kernel<[^>]*>).*", r"\1", k)[:60] if cls == "gemm" else "")].append(dur)
    for key, v in sorted(g.items()):
        print(f"  {key[0]:6s} grid {key[1]}x{key[2]:>6s} {key[3]:48s} median {statistics.median(v):9.1f} us  (n={len(v)})")
