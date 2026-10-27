#!/usr/bin/env python3
"""Per-rank class table of a rocprofv3 kernel_trace.csv of a tensor-parallel prefill (generate --max-tokens 1), lane H27P.

usage: trace_tp2.py kernel_trace.csv[.gz] NTOKENS [--tp 2] [--shapes] [--map shapes.json] [--peak8 366] [--peak16 190]

* One burst per agent (rank): gaps over 50 ms split bursts, the burst with the most kernel time is the prefill (the weight
  upload is idle). Ranks are NOT summed: the wall time of a prefill is the slowest rank's span.
* Classes by kernel name (first match): gemm_a8 / gemm_other (the MXFP4 and vendor GEMMs), quant (the a8 activation pre-pass
  and weight-span check), gdn (conv, delta rule), attn (flash, splitkv), comm (all-reduce, all-gather, fused ar+norm),
  norm, elementwise (everything else of the model loop), runtime (rocclr copies and fills).
* comm kernel time INCLUDES the wait for the peer rank (the one-shot all-reduce spins on the peer's flag): it is an upper
  bound of the wire time, and equal on both ranks' skew by construction.
* idle = span minus kernel time of the burst (launch gaps, host stalls, sync).
* --shapes lists every distinct GEMM launch geometry (grid x workgroup) with launches and time; --map takes
  {"<gx>x<gy>x<gz>": ["name", N, K]} (N, K per rank) and adds achieved TFLOP/s (2*M*N*K per launch, M = rows of the chunk, from
  --rows or NTOKENS / chunks) per projection. Without --map the GEMMs are only listed by shape.
"""
import argparse
import collections
import csv
import gzip
import json
import re
import sys

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("ntokens", type=int)
ap.add_argument("--tp", type=int, default=2)
ap.add_argument("--shapes", action="store_true")
ap.add_argument("--map")
ap.add_argument("--rows", type=int, default=0, help="rows per GEMM launch (chunk size); default: NTOKENS")
ap.add_argument("--peak8", type=float, default=366.0)
ap.add_argument("--peak16", type=float, default=190.0)
ap.add_argument("--proj", action="store_true", help="tp2 27B per-projection TFLOP/s by GEMM grid")
ap.add_argument("--names", action="store_true", help="print every distinct short kernel name with its class")
a = ap.parse_args()

CLASSES = [
    ("quant", r"a8_quant|mx_a8|a8_wspan|quant_kernel|act_quant"),
    ("gemm_a8", r"a8_gemm_kernel|mxfp4_a8_(gemv|rows|mma)"),
    ("gemm_other", r"dq_gemm|gemm_kernel|Cijk|f8_gemm|gemv_kernel|mxfp4_gemv|nvfp4|matmul"),
    ("gdn", r"gdn_|delta_rule|conv_delta"),
    ("attn", r"flash|splitkv|combine_kernel|attention"),
    ("comm", r"ar2_kernel|ag2_kernel|ar_norm|allreduce|all_reduce|ar_rms"),
    ("norm", r"rms|norm"),
    ("runtime", r"rocclr"),
]


def short(n):
    n = n.replace("(anonymous namespace)::", "")
    n = re.sub(r"^void ", "", n)
    d, out = 0, n
    for i, ch in enumerate(n):
        if ch == "<":
            d += 1
        elif ch == ">":
            d -= 1
        elif ch == "(" and d == 0:
            out = n[:i]
            break
    return out.replace("ik_bf16", "bf16")


def klass(n):
    for c, pat in CLASSES:
        if re.search(pat, n):
            return c
    return "elementwise"


opener = gzip.open if a.csv.endswith(".gz") else open
rows = collections.defaultdict(list)
with opener(a.csv, "rt") as f:
    for r in csv.DictReader(f):
        rows[r["Agent_Id"]].append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), short(r["Kernel_Name"]),
                                    "%sx%sx%s" % (r.get("Grid_Size_X", ""), r.get("Grid_Size_Y", ""), r.get("Grid_Size_Z", "")),
                                    r.get("Workgroup_Size_X", "")))
shapes_map = json.load(open(a.map)) if a.map else {}
M = a.rows or a.ntokens
results = {}
for ag, rs in sorted(rows.items()):
    rs.sort()
    bursts, cur = [], [rs[0]]
    for x, y in zip(rs, rs[1:]):
        if y[0] - x[1] > 50_000_000:
            bursts.append(cur)
            cur = []
        cur.append(y)
    bursts.append(cur)
    main = max(bursts, key=lambda s: sum(e - b for b, e, *_ in s))
    if len(main) < 100:
        continue
    span = main[-1][1] - main[0][0]
    tot = sum(e - s for s, e, *_ in main)
    cls = collections.defaultdict(lambda: [0, 0])
    for s, e, n, g, w in main:
        c = klass(n)
        cls[c][0] += 1
        cls[c][1] += e - s
    results[ag] = (main, span, tot, cls)

if not results:
    sys.exit("no burst with kernels found")
wall = max(v[1] for v in results.values())
print(f"{a.csv}: {a.ntokens} tokens, tp {a.tp}, wall (slowest rank span) {wall/1e6:.1f} ms -> {a.ntokens/(wall/1e9):.0f} tok/s (profiler on; slower than a plain run)")
for ag, (main, span, tot, cls) in results.items():
    print(f"\n== {ag}: {len(main)} kernels, span {span/1e6:.1f} ms, kernel time {tot/1e6:.1f} ms, idle {100*(span-tot)/span:.1f} %")
    print(f"{'class':<12}{'n':>7}{'ms':>10}{'% span':>8}{'avg us':>9}")
    for c, (n, t) in sorted(cls.items(), key=lambda kv: -kv[1][1]):
        print(f"{c:<12}{n:>7}{t/1e6:>10.1f}{100*t/span:>8.1f}{t/n/1e3:>9.1f}")
    print(f"{'idle/gaps':<12}{'':>7}{(span-tot)/1e6:>10.1f}{100*(span-tot)/span:>8.1f}")
    if a.names:
        agg = collections.defaultdict(lambda: [0, 0])
        for s, e, n, g, w in main:
            agg[n][0] += 1
            agg[n][1] += e - s
        print(f"  {'kernel':<70}{'class':<11}{'n':>6}{'ms':>9}{'avg us':>9}")
        for n, (c, t) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:40]:
            print(f"  {n[:68]:<70}{klass(n):<11}{c:>6}{t/1e6:>9.2f}{t/c/1e3:>9.1f}")
    if a.shapes or shapes_map:
        sh = collections.defaultdict(lambda: [0, 0, ""])
        for s, e, n, g, w in main:
            if klass(n) in ("gemm_a8", "gemm_other"):
                k = f"{g}"
                sh[k][0] += 1
                sh[k][1] += e - s
                sh[k][2] = n[:40]
        print(f"  {'GEMM grid':<20}{'kernel':<42}{'n':>6}{'ms':>9}{'avg us':>9}  projection / TFLOP/s")
        for g, (c, t, n) in sorted(sh.items(), key=lambda kv: -kv[1][1]):
            extra = ""
            if g in shapes_map:
                name, N, K = shapes_map[g]
                fl = 2.0 * M * N * K * c
                extra = f"{name} N{N} K{K}: {fl/(t*1e-9)/1e12:.1f} TF ({100*fl/(t*1e-9)/1e12/a.peak8:.0f} % fp8 peak)"
            print(f"  {g:<20}{n:<42}{c:>6}{t/1e6:>9.2f}{t/c/1e3:>9.1f}  {extra}")

# ---- Qwen3.8-27B tp2 per-projection achieved TFLOP/s (--proj): launches under 20 us (the not-chosen fold / per-group twin of the a8 GEMM,
# which returns at once) are dropped; the grid y is N/128 (N/256 for the 256-wide pair tile). Shapes per rank at tp2: gate/up N 8704 K 5120,
# down N 5120 K 8704, gdn qkv N 5120, z N 3072, out/o_proj K 3072 N 5120, attn q+gate N 6144, k/v N 512.
if a.tp == 2 and "--proj" in sys.argv:
    H, I = 5120, 17408
    proj = {  # grid_y: (label, flops per prompt token per rank, launches per chunk to expect)
        34: ("gate+up pair (N 2x8704)", 2 * 2 * 8704 * H * 64),   # y 34 = the 256-wide tile, y 68 = the 128-wide tile of the same op (tile by M)
        68: ("gate+up pair (N 2x8704)", 2 * 2 * 8704 * H * 64),
        40: ("down + qkv + out + o_proj (N 5120)", 2 * (64 * H * 8704 + 48 * H * 3072 + 16 * H * 3072 + 48 * H * 5120)),
        24: ("gdn z (N 3072)", 2 * 48 * H * 3072),
        48: ("attn q+gate (N 6144)", 2 * 16 * H * 6144),
    }
    for ag, (main, span, tot, cls) in results.items():
        by = collections.defaultdict(lambda: [0, 0])
        for s, e, n, g, w in main:
            if klass(n) == "gemm_a8" and e - s >= 20_000:
                y = int(g.split("x")[1])
                by[y][0] += 1
                by[y][1] += e - s
        print(f"\n== {ag}: real GEMM launches by grid y (tokens {a.ntokens})")
        if 34 in by or 68 in by:  # one op, two tiles: report the pair once
            c = by.pop(34, [0, 0]); d = by.pop(68, [0, 0]); by[34] = [c[0] + d[0], c[1] + d[1]]
        for y, (c, t) in sorted(by.items(), key=lambda kv: -kv[1][1]):
            lab, fl = proj.get(y, ("?", 0))
            tf = fl * a.ntokens / (t * 1e-9) / 1e12 if fl else 0
            print(f"  y={y:<3} {lab:<38} n={c:<5} {t/1e6:8.1f} ms  {tf:6.1f} TFLOP/s ({100*tf/a.peak8:.0f} % fp8 peak)" if tf else f"  y={y:<3} {lab:<38} n={c:<5} {t/1e6:8.1f} ms")
