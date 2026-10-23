#!/usr/bin/env python3
"""Roofline table of one rocprofv3 kernel trace of a 35B prefill (generate --max-tokens 1), lane hp.

usage: roofline.py kernel_trace.csv <prompt tokens N> [ceiling.txt]

Kernels are grouped by class; useful work per class is analytic (Qwen3.6-35B-A3B: hidden 2048, 40 layers = 30 GatedDeltaNet
+ 10 full attention, 256 experts top-8 of 512, shared expert 512, GDN 16 key / 32 value heads of 128, attention 16 q / 2 kv
heads of 256 with an output gate). FLOPs count 2 per MAC; attention is causal (sum over positions of p + 1). The ceiling of
a class is the best measured rate of the same kind of work on this card (ceiling.txt: hipBLASLt at the class's shape, or the
WMMA issue peak where no library GEMM is the same work); memory-bound classes are given in GB/s of their minimum traffic.
"""
import collections
import csv
import re
import sys

trace, N = sys.argv[1], int(sys.argv[2])
ceil = {}
if len(sys.argv) > 3:
    for line in open(sys.argv[3]):
        m = re.match(r"WMMA issue peak (\w+)\s+([\d.]+)", line)
        if m:
            ceil["peak_" + m.group(1)] = float(m.group(2))
        m = re.match(r"(.+?)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+) \(\s*\d+\)\s+([\d.]+) \(\s*\d+\)\s+([\d.]+)", line)
        if m:
            ceil[m.group(1).strip()] = (float(m.group(5)), float(m.group(6)), float(m.group(7)))

H, L, LG, LA, E, TOPK, I, IS = 2048, 40, 30, 10, 256, 8, 512, 512
dense_fp8 = LG * (8192 * H + 4096 * H + H * 4096) + LA * (8192 * H + 2 * 512 * H + H * 4096)
small_bf16 = L * E * H + LG * 2 * 32 * H + L * H  # router, GDN a / b, shared-expert gate
attn_pairs = N * (N + 1) // 2
work = {  # class -> (FLOPs, bytes, ceiling TFLOPS, ceiling source)
    "routed experts gate_up (NVFP4 -> bf16 WMMA)": 2 * N * TOPK * 2 * I * H * L,
    "routed experts down (NVFP4 -> bf16 WMMA)": 2 * N * TOPK * I * H * L,
    "shared expert gate/up/down (NVFP4 -> bf16 WMMA)": 2 * N * 3 * IS * H * L,
    "dense fp8 projections (fp8 x fp8 WMMA)": 2 * N * dense_fp8,
    "small bf16 GEMMs (router, GDN a/b)": 2 * N * small_bf16,
    "flash attention (bf16 WMMA)": LA * 4 * 16 * 256 * attn_pairs,
    "GatedDeltaNet scan (f32 VALU, sequential)": 7 * N * LG * 32 * 128 * 128,
}
classes = [
    ("routed experts gate_up (NVFP4 -> bf16 WMMA)", r"moe_wmma_kernel_r7<true|moe_wmma_kernel_v<true|moe_wmma_kernel<true"),
    ("routed experts down (NVFP4 -> bf16 WMMA)", r"moe_wmma_kernel_r7<false|moe_wmma_kernel_v<false|moe_wmma_kernel<false"),
    ("shared expert gate/up/down (NVFP4 -> bf16 WMMA)", r"dq_gemm_kernel<.*Nvfp4"),
    ("dense fp8 projections (fp8 x fp8 WMMA)", r"f8_gemm_kernel"),
    ("small bf16 GEMMs (router, GDN a/b)", r"gemm_wmma_kernel"),
    ("flash attention (bf16 WMMA)", r"flash_hpb|flash_"),
    ("GatedDeltaNet scan (f32 VALU, sequential)", r"gdn_delta"),
    ("GDN conv + gated norm", r"gdn_conv|gated_rmsnorm"),
    ("MoE glue (top-k, group, reduce, combine, silu)", r"moe_reduce|moe_combine|moe_topk|moe_group|silu_glu"),
    ("fp8 activation quantise", r"f8_actq"),
    ("norms, rope, sigmoid, copies, other", r"."),
]
rows = []
import gzip
with (gzip.open(trace, "rt") if trace.endswith(".gz") else open(trace)) as f:
    for r in csv.DictReader(f):
        rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]))
rows.sort()
bursts, cur = [], [rows[0]]
for a, b in zip(rows, rows[1:]):
    if b[0] - a[1] > 50_000_000:
        bursts.append(cur)
        cur = []
    cur.append(b)
bursts.append(cur)
main = max(bursts, key=lambda s: sum(e - b for b, e, _ in s))
span = (main[-1][1] - main[0][0]) / 1e6
agg = collections.OrderedDict((c, [0.0, 0]) for c, _ in classes)
for s, e, n in main:
    for c, pat in classes:
        if re.search(pat, n):
            agg[c][0] += (e - s) / 1e6
            agg[c][1] += 1
            break
tot = sum(v[0] for v in agg.values())


def cl(c):  # ceiling TFLOPS and its source
    if "gate_up" in c and "routed" in c:
        return ceil.get("experts gate_up (dense ideal)", (None,))[0], "hipBLASLt bf16, all pairs as one GEMM"
    if "routed" in c:
        return ceil.get("experts down (dense ideal)", (None,))[0], "hipBLASLt bf16, all pairs as one GEMM"
    if "shared" in c:
        g, d = ceil.get("shared expert gate_up", (None,)), ceil.get("shared expert down", (None,))
        if g[0] and d[0]:  # time-weighted: 2/3 of the FLOPs at the gate/up shape, 1/3 at the down shape
            return 1 / ((2 / 3) / g[0] + (1 / 3) / d[0]), "hipBLASLt bf16 at the shapes"
        return None, ""
    if "fp8 proj" in c:
        sh = [("qkv / q+gate", 30 * 8192 + 10 * 8192), ("z", 30 * 4096), ("o / out_proj", 30 * 2048 * 2 + 10 * 2048 * 2), ("k / v", 10 * 1024)]
        if all(s in ceil for s, _ in sh):
            w = sum(x for _, x in sh)
            return w / sum(x / ceil[s][1] for s, x in sh), "hipBLASLt fp8 at the shapes"
        return None, ""
    if "small bf16" in c:
        return ceil.get("k / v", (None,))[0], "hipBLASLt bf16 (k/v shape, N=512)"
    if "flash" in c:
        return ceil.get("peak_bf16"), "bf16 WMMA issue peak"
    if "scan" in c:  # measured, gdnbench.hip (raw/w4): v2's lane map with the LDS reads of q / k removed (wrong numbers, timing
        # probe "v3i-nolds") runs at 0.324 us per token per layer at the 35B's 256 waves; the same work per token as the table's
        return 7 * 32 * 128 * 128 / 0.324e-6 / 1e12, "measured floor of this lane map: LDS-free probe, 0.324 us/token/layer (raw/w4)"
    return None, ""


print(f"N = {N} tokens; main burst span {span:.1f} ms, kernel time {tot:.1f} ms, idle {span - tot:.1f} ms ({100 * (span - tot) / span:.1f} %)\n")
print("| kernel class | launches | ms | share | achieved TFLOPS | ceiling TFLOPS | % of ceiling | ceiling source |")
print("|---|---:|---:|---:|---:|---:|---:|---|")
for c, (ms, n) in agg.items():
    fl = work.get(c)
    ach = fl / (ms * 1e-3) / 1e12 if fl and ms else None
    cv, src = cl(c)
    pct = f"{100 * ach / cv:.0f} %" if ach and cv else ""
    print(f"| {c} | {n} | {ms:.1f} | {100 * ms / tot:.1f} % | {f'{ach:.1f}' if ach else ''} | {f'{cv:.0f}' if cv else ''} | {pct} | {src} |")
print(f"| idle (gaps between kernels) | | {span - tot:.1f} | | | | | |")
print(f"\nprefill-equivalent rate over the burst: {N / (span * 1e-3):.0f} tok/s; useful matrix work "
      f"{sum(v for k, v in work.items() if 'scan' not in k) / 1e12:.2f} TFLOP = "
      f"{sum(v for k, v in work.items() if 'scan' not in k) / (span * 1e-3) / 1e12:.1f} TFLOPS over the span")
