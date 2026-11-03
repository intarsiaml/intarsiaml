#!/usr/bin/env python3
"""Lane dsv41/prefill: the per-kernel-class table, ours against TensorFold, from two nsys cuda_gpu_trace CSVs of rank 0.

Usage: class_table.py ENGINE=TRACE.csv[:ROWS_PER_CHUNK] ... --prompts 1034,4106

Each trace holds requests (idle gap >= 1 s) alternating prompt A, prompt B, ... for `--prompts` (token counts), repeated.
Per engine and prompt: the median over its requests of the wall time attributed to each class ("excl": the kernel that
started last among those running gets the time, so PDL waits and overlap are not charged twice; classes add up to GPU busy),
reported as ms per chunk (full chunks of ROWS plus the tail chunk counted as a chunk) and us per prompt token.
ENGINE is `ours` or `tf` (they pick the class rules); ours:ROWS chunk 512 by default, tf 2048.
"""
import re
import statistics as st
import sys
from collections import defaultdict
sys.path.insert(0, __import__("os").path.dirname(__file__))
from prefill_table import load, split, excl_times  # noqa: E402

OURS = [
    ("moe gate/up", r"grouped_rows_kernel<\(bool\)1"),
    ("moe down", r"grouped_rows_kernel<\(bool\)0"),
    ("moe other (group, rot, epilogue, combine, cp)", r"ik_exl3x::"),
    ("router gate (f16 gate matmul)", r"dense_f16w"),
    ("dense linears (gemm, linear, rot_in, transpose)", r"^void gemm_kernel|gemm_kernel<|linear_kernel|^void rot_in_kernel|rot_in_kernel$|transpose01|split_"),
    ("collectives", r"nccl|ik_f32_to_bf16"),
    ("attention (sparse, rope, index, kv)", r"attn_|rope_|index_|window|compress|qdq|topk|kv_"),
    ("hyper-connection", r"hc_"),
]
TF = [
    ("moe gate/up", r"grouped_mma_kernel"),
    ("moe down", r"grouped_rows_kernel"),
    ("moe other (group, rot, epilogue, combine, cp)", r"gateup_epilogue|down_combine|grouped_cp|group_count|group_place|work_list|decode_prep|rot_in_kernel<__nv_bfloat16>"),
    ("router gate (f16 gate matmul)", r"cutlass"),
    ("dense linears (gemm, linear, rot_in, transpose)", r"^_gemm|glinear|unpack_kernel|rot_in_kernel|rot_many|rowmm|_rmsnorm_rot"),
    ("collectives", r"nccl|rdma|tf_ds_add2|bf16_to_f32|f16_to_f32|f32_to_bf16"),
    ("attention (sparse, rope, index, kv)", r"sparse_attn|index_score|topk|rope|kv_|_q_kv|compress|swa|_finish$|_partial|engram"),
    ("hyper-connection", r"_hc_"),
]


def classify(rules, name):
    n = name.replace("(anonymous namespace)::", "").replace("<unnamed>::", "")
    for c, p in rules:
        if re.search(p, n):
            return c
    return "other"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    prompts = [int(x) for x in (sys.argv[sys.argv.index("--prompts") + 1] if "--prompts" in sys.argv else "1034,4106").split(",")]
    args = [a for a in args if not re.fullmatch(r"[\d,]+", a)]
    res = {}
    for a in args:
        eng, rest = a.split("=", 1)
        path, _, rows = rest.partition(":")
        rows = int(rows) if rows else (512 if eng.startswith("ours") else 2048)
        rules = OURS if eng.startswith("ours") else TF
        reqs = [q for q in split(load(path), 1_000_000_000) if (max(r[0] + r[1] for r in q) - q[0][0]) > 50e6]
        per = defaultdict(list)
        for qi, q in enumerate(reqs):
            pi = qi % len(prompts)
            ex = excl_times(q)
            cls = defaultdict(float)
            for i, r in enumerate(q):
                cls[classify(rules, r[2])] += ex.get(i, 0) / 1e6
            span = (max(r[0] + r[1] for r in q) - q[0][0]) / 1e6
            per[pi].append((cls, span))
        for pi, lst in per.items():
            n = prompts[pi]
            full, tail = divmod(n, rows)
            nch = full + (1 if tail else 0)
            names = sorted({c for cls, _ in lst for c in cls})
            med = {c: st.median(cls.get(c, 0.0) for cls, _ in lst) for c in names}
            res[(eng, n)] = (med, st.median(s for _, s in lst), nch, rows, len(lst))
    for n in prompts:
        keys = [k for k in res if k[1] == n]
        print(f"\n### {n}-token prompt, rank 0, median of requests; ms total / ms per chunk / us per token")
        names = sorted({c for k in keys for c in res[k][0]}, key=lambda c: -max(res[k][0].get(c, 0) for k in keys))
        hdr = f"{'class':<50}" + "".join(f"{k[0] + ' (' + str(res[k][2]) + ' chunks of ' + str(res[k][3]) + ')':>34}" for k in keys)
        print(hdr)
        for c in names:
            print(f"{c:<50}" + "".join(f"{res[k][0].get(c, 0):>10.1f} {res[k][0].get(c, 0) / res[k][2]:>10.1f} {1000 * res[k][0].get(c, 0) / n:>11.1f}" for k in keys))
        print(f"{'GPU busy (sum)':<50}" + "".join(f"{sum(res[k][0].values()):>10.1f} {sum(res[k][0].values()) / res[k][2]:>10.1f} {1000 * sum(res[k][0].values()) / n:>11.1f}" for k in keys))
        print(f"{'span (first to last kernel)':<50}" + "".join(f"{res[k][1]:>10.1f} {res[k][1] / res[k][2]:>10.1f} {1000 * res[k][1] / n:>11.1f}" for k in keys))


if __name__ == "__main__":
    main()
