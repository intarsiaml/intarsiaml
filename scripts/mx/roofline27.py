#!/usr/bin/env python3
"""Roofline table of one rocprofv3 kernel trace of a Qwen3.8-27B MXFP4 prefill (generate --max-tokens 1), lane mx.

usage: roofline27.py kernel_trace.csv <prompt tokens N> [peak_fp8_tflops peak_bf16_tflops]

27B: hidden 5120, 64 layers = 48 GatedDeltaNet (16 key / 48 value heads of 128) + 16 full attention (24 q / 4 kv heads of
256, output gate), MLP 17408, all projections MXFP4 (W4A4). FLOPs count 2 per MAC; attention is causal (sum over positions
of p + 1). The MXFP4 GEMM launches are split by output width N (grid y * 128 for the 128-wide weight tile of both GEMM
kernels); N = 5120 is the MLP down projection (K = 17408) right after silu_glu, else out_proj / o_proj (K = 6144).
Byte floor: every MXFP4 weight read once per prefill chunk (0.5 B + 1/32 B per weight), at 640 GB/s.
"""
import collections
import csv
import re
import sys

trace, N = sys.argv[1], int(sys.argv[2])
P8 = float(sys.argv[3]) if len(sys.argv) > 3 else 377.0
P16 = float(sys.argv[4]) if len(sys.argv) > 4 else 190.0
H, L, LG, LA, I = 5120, 64, 48, 16, 17408
SHAPES = {17408: ("mlp gate / up", 5120), 10240: ("gdn in_proj_qkv", 5120), 6144: ("gdn in_proj_z", 5120),
          12288: ("attn q + gate", 5120), 1024: ("attn k / v", 5120), 128: ("gdn in_proj_a / b (N=48)", 5120)}

rows = []
with open(trace) as f:
    for r in csv.DictReader(f):
        rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"],
                     int(r.get("Grid_Size_X") or 0), int(r.get("Grid_Size_Y") or 0), int(r.get("Workgroup_Size_X") or 1)))
rows.sort()
bursts, cur = [], [rows[0]]
for a, b in zip(rows, rows[1:]):
    if b[0] - a[1] > 50_000_000:
        bursts.append(cur)
        cur = []
    cur.append(b)
bursts.append(cur)
main = max(bursts, key=lambda s: sum(e - b for b, e, *_ in s))
span = main[-1][1] - main[0][0]
tot = sum(e - s for s, e, *_ in main)

cls = collections.OrderedDict()  # class -> [launches, ns, flops]
def add(c, ns, fl=0.0):
    a = cls.setdefault(c, [0, 0, 0.0])
    a[0] += 1
    a[1] += ns
    a[2] += fl

chunk_tokens = None
prev = ""
for s, e, name, gx, gy, wg in main:
    ns = e - s
    if "gemm" in name and ("Mxfp4" in name or "mxf8" in name):
        m = (gx // max(wg, 1))  # blocks along M; tokens per launch derived below
        n = gy * 128
        if n == 5120:
            what, k = ("mlp down", 17408) if "silu" in prev else ("gdn out_proj / attn o_proj", 6144)
        else:
            what, k = SHAPES.get(n, (f"N~{n}", 5120))
        add("GEMM " + what, ns)
    elif "flash" in name:
        add("flash attention (bf16 WMMA)", ns)
    elif "gdn_delta" in name:
        add("GDN delta rule (sequential scan)", ns)
    elif "xprep" in name or "actq" in name:
        add("MXFP4 activation QDQ pre-pass", ns)
    else:
        add("other: " + re.sub(r"<.*", "", re.sub(r"^void ", "", name.split("(")[0]))[:40], ns)
    if not ("xprep" in name or "actq" in name):
        prev = name

# useful FLOPs per class (whole prompt)
gemm_macs = {"mlp gate / up": 2 * I * H * L, "mlp down": I * H * L, "gdn in_proj_qkv": 10240 * H * LG,
             "gdn in_proj_z": 6144 * H * LG, "gdn out_proj / attn o_proj": 6144 * H * L, "attn q + gate": 12288 * H * LA,
             "attn k / v": 2 * 1024 * H * LA, "gdn in_proj_a / b (N=48)": 2 * 48 * H * LG}
pairs = N * (N + 1) // 2
flops = {"GEMM " + k: 2.0 * N * v for k, v in gemm_macs.items()}
flops["flash attention (bf16 WMMA)"] = LA * 24 * 4 * 256 * pairs
wbytes = sum(gemm_macs.values()) * (0.5 + 1 / 32)  # per chunk read of every weight
print(f"N={N} tokens; kernels {len(main)}, span {span/1e6:.1f} ms, kernel time {tot/1e6:.1f} ms, idle {100*(span-tot)/span:.1f} %")
print(f"prefill {N/(span/1e9):.0f} tok/s (span); GEMM FLOPs {2*N*sum(gemm_macs.values())/1e12:.1f} TFLOP; weights {wbytes/1e9:.2f} GB per chunk read")
print(f"\n{'class':<44}{'n':>7}{'ms':>10}{'%kern':>7}{'TFLOP/s':>9}{'%peak':>7}")
gt = 0
for c, (n, ns, _) in sorted(cls.items(), key=lambda kv: -kv[1][1]):
    if ns / tot < 0.002:
        continue
    fl = flops.get(c)
    rate = f"{fl/ns/1e3:9.1f}" if fl else " " * 9
    pk = P8 if c.startswith("GEMM") else P16
    pct = f"{100*fl/ns/1e3/pk:6.1f}%" if fl else " " * 7
    print(f"{c:<44}{n:>7}{ns/1e6:>10.2f}{100*ns/tot:>6.1f}%{rate}{pct}")
    if c.startswith("GEMM"):
        gt += ns
gf = 2 * N * sum(gemm_macs.values())
print(f"\nall MXFP4 GEMMs: {gt/1e6:.1f} ms = {gf/gt/1e3:.1f} TFLOP/s ({100*gf/gt/1e3/P8:.1f} % of the {P8:.0f} TFLOP/s fp8 WMMA peak, "
      f"{100*gf/gt/1e3/P16:.1f} % of the {P16:.0f} bf16 peak); floor at the fp8 peak {gf/P8/1e9:.1f} ms")
