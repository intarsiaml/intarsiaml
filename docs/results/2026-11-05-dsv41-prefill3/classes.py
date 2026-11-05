#!/usr/bin/env python3
"""Lane dsv41/prefill3: per-kernel-class GPU time per request, ours against TensorFold, from two nsys cuda_gpu_trace CSVs of rank 0.

Usage: classes.py ours=TRACE.csv tf=TRACE.csv [--prompts 1034,4106]
A trace that holds one prompt only: ENGINE=TRACE.csv|4106 (all its requests are that prompt; it appears in that prompt's table only).

Requests are split at idle gaps of >= 1 s and alternate over --prompts (token counts), as in 2026-11-02-dsv41-prefill/class_table.py.
Time per class is "excl" (prefill_table.excl_times: the kernel that started last among those running gets the wall time, so PDL waits
and overlap are not charged twice; the classes add up to GPU busy). Median over the requests of each prompt. Finer classes than
class_table.py: rope, index, kv, engram and top-k are split out of "attention", and collectives are split into NCCL and the rest.
Both engines run the same layer schedule now (bounded replay), which is why the totals can be compared. A launch-count check per
class is printed beside the times: a class whose counts differ between the engines is a different amount of work, not a slower kernel.
"""
import os
import re
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "2026-11-02-dsv41-prefill"))
from prefill_table import load as _load, split, excl_times  # noqa: E402
import gzip
import tempfile


def load(path):
    """prefill_table.load, also for a .csv.gz (decompressed to a temporary file)."""
    if not path.endswith(".gz"):
        return _load(path)
    with tempfile.NamedTemporaryFile(suffix=".csv") as t:
        t.write(gzip.open(path).read())
        t.flush()
        return _load(t.name)

OURS = [
    ("moe gate/up", r"grouped_rows_smem_kernel|grouped_rows_kernel<\(bool\)1"),
    ("moe down", r"grouped_rows_kernel<\(bool\)0"),
    ("moe other (group, rot, epilogue, combine, cp)", r"ik_exl3x::"),
    ("router gate", r"dense_f16w"),
    ("dense linears (gemm, linear, rot_in, transpose)", r"^void gemm_kernel|gemm_kernel<|linear_kernel|^void rot_in_kernel|rot_in_kernel$|transpose01|split_|rmsnorm"),
    ("collectives (nccl)", r"nccl"),
    ("collectives (f32->bf16 convert, rdma)", r"ik_f32_to_bf16|rdma"),
    ("hyper-connection", r"hc_"),
    ("attention core", r"attn_"),
    ("rope", r"rope_"),
    ("index (score, select, top-k)", r"index_|sel_|topk"),
    ("kv (window, compress, qdq, ring)", r"window|compress|qdq|kv_|ring_write"),
    ("engram", r"engram"),
]
TF = [
    ("moe gate/up", r"grouped_mma_kernel"),
    ("moe down", r"grouped_rows_kernel"),
    ("moe other (group, rot, epilogue, combine, cp)", r"gateup_epilogue|down_combine|grouped_cp|group_count|group_place|work_list|decode_prep|rot_in_kernel<__nv_bfloat16>|_route"),
    ("router gate", r"cutlass"),
    ("dense linears (gemm, linear, rot_in, transpose)", r"^_gemm|glinear|unpack_kernel|rot_in_kernel|rot_many|rowmm|_rmsnorm"),
    ("collectives (nccl)", r"nccl"),
    ("collectives (f32->bf16 convert, rdma)", r"rdma|tf_ds_add2|bf16_to_f32|f16_to_f32|f32_to_bf16"),
    ("hyper-connection", r"_hc_"),
    ("attention core", r"sparse_attn"),
    ("rope", r"_rope_heads"),
    ("index (score, select, top-k)", r"index_score|topk|_topk_sel"),
    ("kv (window, compress, qdq, ring)", r"kv_norm_rope|compress|swa|_fp4|_q_kv"),
    ("engram", r"engram"),
]
ORDER = [c for c, _ in OURS] + ["other"]


def classify(rules, name):
    n = name.replace("(anonymous namespace)::", "").replace("<unnamed>::", "")
    for c, p in rules:
        if re.search(p, n):
            return c
    return "other"


def tabulate(path, rules, prompts):
    reqs = [q for q in split(load(path), 1_000_000_000) if (max(r[0] + r[1] for r in q) - q[0][0]) > 50e6]
    per = defaultdict(lambda: defaultdict(list))  # prompt idx -> class -> [ms per request]
    cnt = defaultdict(lambda: defaultdict(list))
    busy = defaultdict(list)
    for k, q in enumerate(reqs):
        pi = k % len(prompts)
        ex = excl_times(q)
        t = defaultdict(float)
        n = defaultdict(int)
        for i, r in enumerate(q):
            c = classify(rules, r[2])
            t[c] += ex.get(i, 0) / 1e6
            if not r[2].startswith("[CUDA"):
                n[c] += 1
        for c in ORDER:
            per[pi][c].append(t[c])
            cnt[pi][c].append(n[c])
        busy[pi].append(sum(t.values()))
    return per, cnt, busy, len(reqs)


def main():
    args = [a for a in sys.argv[1:] if "=" in a and not a.startswith("--")]
    prompts = [int(x) for x in (sys.argv[sys.argv.index("--prompts") + 1] if "--prompts" in sys.argv else "1034,4106").split(",")]
    res, only = {}, {}
    for a in args:
        eng, path = a.split("=", 1)
        if "|" in path:
            path, one = path.split("|", 1)
            only[eng] = int(one)
            res[eng] = tabulate(path, OURS if eng.startswith("ours") else TF, [int(one)])
        else:
            res[eng] = tabulate(path, OURS if eng.startswith("ours") else TF, prompts)
    engs = list(res)
    for pi, p in enumerate(prompts):
        shown = [e for e in engs if only.get(e, p) == p]
        print(f"\n### {p}-token prompt, rank 0, median of requests; ms per request (launches per request)")
        print(f"{'class':50s} " + " ".join(f"{e:>18s}" for e in shown) + ("   ratio" if len(shown) == 2 else "") + ("   gap ms" if len(shown) == 2 else ""))
        for c in ORDER + ["GPU busy (sum)"]:
            cells, vals = [], []
            for e in shown:
                per, cnt, busy, _ = res[e]
                pi_e = 0 if e in only else pi
                if c == "GPU busy (sum)":
                    v = st.median(busy[pi_e]) if busy[pi_e] else 0.0
                    cells.append(f"{v:18.1f}")
                else:
                    v = st.median(per[pi_e][c]) if per[pi_e][c] else 0.0
                    n = int(st.median(cnt[pi_e][c])) if cnt[pi_e][c] else 0
                    cells.append(f"{v:10.1f} ({n:5d})".rjust(18))
                vals.append(v)
            tail = ""
            if len(shown) == 2:
                tail = f"  {vals[0] / vals[1]:5.2f}x  {vals[0] - vals[1]:8.1f}" if vals[1] > 0.05 else f"  {'-':>6s}  {vals[0] - vals[1]:8.1f}"
            if max(vals) > 0.05:
                print(f"{c:50s} " + " ".join(cells) + tail)
    for e in engs:
        print(f"\n{e}: {res[e][3]} requests")


if __name__ == "__main__":
    main()
