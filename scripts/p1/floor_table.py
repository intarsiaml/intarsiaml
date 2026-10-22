#!/usr/bin/env python3
"""Per-op time vs HBM byte floor for one Qwen3.8-27B MXFP4 decode step, from a rocprofv3 kernel_trace.csv.
usage: floor_table.py kernel_trace.csv [bandwidth GB/s, default 627]
MXFP4 GEMV shapes are told apart by grid size (8 rows per 256-thread block) and, for N = 5120, by the launch before."""
import collections, csv, sys

path = sys.argv[1]
bw = float(sys.argv[2]) if len(sys.argv) > 2 else 627.0
rows = sorted((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"], int(r["Grid_Size_X"]))
              for r in csv.DictReader(open(path)))
idx = [i for i, r in enumerate(rows) if "argmax" in r[2]]
steps = list(zip(idx[20:-2], idx[21:-1]))
ns = len(steps)
shape = {557056: ("ffn gate / up", 17408, 5120), 327680: ("gdn in_qkv", 10240, 5120), 196608: ("gdn in_z", 6144, 5120),
         1536: ("gdn in_b / in_a", 48, 5120), 393216: ("attn q", 12288, 5120), 32768: ("attn k / v", 1024, 5120)}
mx = collections.defaultdict(lambda: [0, 0.0, 0])
other = collections.defaultdict(lambda: [0, 0.0])
wall = 0.0
for a, b in steps:
    prev = None
    st = rows[a + 1:b + 1]
    wall += (st[-1][1] - st[0][0]) / 1000
    for s, e, n, g in st:
        d = (e - s) / 1000
        if "mxfp4_gemv" in n:
            if g == 163840:
                nm, N, K = ("ffn down", 5120, 17408) if prev == 557056 else ("gdn out / attn o", 5120, 6144)
            else:
                nm, N, K = shape[g]
            prev = g
            x = mx[nm]
            x[0] += 1; x[1] += d; x[2] += N * (K // 2 + K // 32)
        else:
            k = n.replace("(anonymous namespace)::", "").replace("void ", "").split("(")[0].split("<")[0]
            if k == "gemv_kernel" and g == 7946240:
                k = "lm_head (bf16 gemv)"
            other[k][0] += 1; other[k][1] += d
print(f"{ns} decode steps; wall/step {wall / ns:.0f} us")
print(f"{'op':24s} {'n/step':>6s} {'us/step':>8s} {'MB/step':>8s} {'GB/s':>6s} {'%floor':>6s} {'floor us':>8s}")
tot = tf = 0.0
for nm, (c, d, by) in sorted(mx.items(), key=lambda x: -x[1][1]):
    us, mb = d / ns, by / ns / 1e6
    fl = by / ns / (bw * 1e3)
    tot += us; tf += fl
    print(f"{nm:24s} {c / ns:6.0f} {us:8.0f} {mb:8.1f} {mb / us * 1e3:6.0f} {100 * fl / us:6.0f} {fl:8.0f}")
lm = 248320 * 5120 * 2
for k, (c, d) in sorted(other.items(), key=lambda x: -x[1][1])[:12]:
    us = d / ns
    if k.startswith("lm_head"):
        fl = lm / (bw * 1e3)
        tot += us; tf += fl
        print(f"{k:24s} {c / ns:6.0f} {us:8.0f} {lm / 1e6:8.1f} {lm / 1e6 / us * 1e3:6.0f} {100 * fl / us:6.0f} {fl:8.0f}")
    else:
        print(f"{k:24s} {c / ns:6.0f} {us:8.0f}")
print(f"weight GEMVs + lm_head: {tot:.0f} us/step vs floor {tf:.0f} us ({100 * tf / tot:.0f} %)")
