#!/usr/bin/env python3
"""Lane HDC: per-shape efficiency of the W4A8 decode GEMV (mxfp4_a8_rows kernels, fused or not) of a tp=2 27B decode trace, rank 'Agent 1'.
usage: gemvshapes.py k_kernel_trace.csv [agent]
The launch grid is (output rows * 16) threads, so grid_x / 16 = the output rows of the (pair) launch. Ideal time = rows * (K / 2 + K / 32)
bytes (e2m1 weights + e8m0 scales) at 640 GB/s. The row count identifies the op; N = 5120 is the down_proj (K 8704) and the two K 3072 ops,
told apart by duration."""
import csv, collections, statistics as st, sys
path = sys.argv[1]; agent = sys.argv[2] if len(sys.argv) > 2 else "Agent 1"
rows = collections.defaultdict(list)
for r in csv.DictReader(open(path)):
    if r["Agent_Id"] != agent: continue
    n = r["Kernel_Name"]
    if "mxfp4_a8_rows" in n:
        rows[int(r["Grid_Size_X"]) // 16].append(int(r["End_Timestamp"]) - int(r["Start_Timestamp"]))
def ideal(N, K): return N * (K / 2 + K / 32) / 640e3  # us at 640 GB/s
steps = max(len(v) for v in rows.values()) // 1
names = {17408: "gate+up pair (K 5120)", 8192: "GDN in_proj qkv+z (K 5120)", 6144: "attn q+gate (K 5120)", 1024: "attn k,v pair (K 5120)", 48: "GDN in_proj a,b (K 5120)"}
tot = 0
print(f"{'shape (rows)':26s} {'calls/step':>10s} {'median us':>10s} {'ideal us':>9s} {'% of 640 GB/s':>14s} {'us/step':>8s}")
nsteps = None
for N, v in sorted(rows.items(), key=lambda x: -sum(x[1])):
    v = sorted(v)
    if nsteps is None: nsteps = len(rows[17408]) / 64 if 17408 in rows else 130
    groups = [(names.get(N, f"rows {N}"), v, 5120)]
    if N == 5120:
        groups = [("down_proj (K 8704)", [x for x in v if x >= 22000], 8704), ("GDN out / attn o (K 3072)", [x for x in v if x < 22000], 3072)]
    for name, g, K in groups:
        if not g: continue
        md = st.median(g) / 1e3
        print(f"{name:26s} {len(g)/nsteps:10.0f} {md:10.1f} {ideal(N, K):9.1f} {100*ideal(N, K)/md:13.0f}% {sum(g)/nsteps/1e3:8.0f}")
        tot += sum(g) / nsteps / 1e3
print(f"{'total rows kernel':26s} {'':10s} {'':10s} {'':9s} {'':14s} {tot:8.0f}")
