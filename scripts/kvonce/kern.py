#!/usr/bin/env python3
"""kvonce: verify-attention kernel time per pass from nsys.sh's cuda_gpu_kern_sum CSVs (python3 scripts/kvonce/kern.py RAW_DIR).
ms per pass = mean `splitkv_rows_persist_kernel` launch x 10 (the 35B's 10 full-attention layers, one row-table launch per layer and pass);
also the draft head's own attention (`splitkv_kernel<8,...>`, not changed here) per pass = its total / (persist launches / 10)."""
import csv, pathlib, sys
raw = pathlib.Path(sys.argv[1])
print("| arm | ctx | persist launches | mean launch ms | **verify attention ms / pass** | head attention ms / pass |\n|---|---|---|---|---|---|")
for arm in ("off", "on"):
    for sz in (4096, 32768, 65536):
        f = raw / f"nsys-{arm}" / f"nsys-{arm}-T0.7-{sz}_cuda_gpu_kern_sum.csv"
        if not f.exists():
            continue
        rows = list(csv.DictReader(open(f)))
        key = lambda r: r.get("Name", "")
        p = [r for r in rows if "splitkv_rows_persist_kernel" in key(r)]
        h = [r for r in rows if "splitkv_kernel<(int)8" in key(r)]
        n = sum(int(r["Instances"]) for r in p); tot = sum(float(r["Total Time (ns)"]) for r in p) / 1e6
        ht = sum(float(r["Total Time (ns)"]) for r in h) / 1e6
        print(f"| {arm} | {sz // 1024}k | {n} | {tot / n:.3f} | **{10 * tot / n:.2f}** | {ht / (n / 10):.2f} |")
