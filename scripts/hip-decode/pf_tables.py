#!/usr/bin/env python3
"""Lane pf: per-kernel / per-layer-type tables from rocprofv3 kernel traces of one prefill (`generate --max-tokens 1`).

  pf_tables.py kinds <trace.csv> [<trace.csv> ...]     share by layer type per trace (+ ms per 1000 prompt tokens)
  pf_tables.py kernels <trace.csv>                     top kernels of one trace
  pf_tables.py chunks <trace.csv> <rows,rows,..>        flash_hpb ms and TFLOP/s per prefill chunk (one dispatch per attention layer)

Kernel -> layer type: attention = flash_hpb (prefill attention), attn_prep = gate_split_qk_norm_rope_kv (RoPE + KV write),
gdn = gdn_*, moe = moe_*, matmul = f8_gemm / dq_gemm / gemm_wmma / f8_actq / *gemv*, norm = *norm* / sigmoid_mul / silu_glu / add,
copy = rocclr copy / fill, other = everything else.  Durations are the profiler's (End - Start of each dispatch); the sum over
dispatches is the GPU busy time, not the wall time (gaps between dispatches are not counted).
"""
import collections
import csv
import gzip
import re
import sys


def core(name: str) -> str:
    n = name.replace("void ", "").replace("(anonymous namespace)::", "")
    m = re.match(r"([A-Za-z0-9_]+)(<[^>(]*>)?", n)
    return (m.group(1) + (m.group(2) or "")) if m else n[:60]


def kind(name: str) -> str:
    n = name
    if "flash_hpb" in n or "attn_prefill" in n or "flash_kernel" in n:
        return "attention"
    if "split_qk_norm_rope" in n or "rope" in n or "kv_write" in n:
        return "attn_prep (RoPE + KV write)"
    if "gdn_" in n:
        return "gdn"
    if "moe_" in n:
        return "moe"
    if any(k in n for k in ("f8_gemm", "dq_gemm", "gemm_wmma", "f8_actq", "gemv", "matmul")):
        return "matmul"
    if any(k in n for k in ("norm", "sigmoid_mul", "silu_glu", "add_kernel", "embedding", "argmax")):
        return "norm / elementwise"
    if "rocclr" in n:
        return "copy / fill"
    return "other"


def load(path):
    rows = []
    opener = gzip.open if path.endswith('.gz') else open
    for r in csv.DictReader(opener(path, 'rt')):
        rows.append((int(r["Start_Timestamp"]), (int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e6, r["Kernel_Name"]))
    rows.sort()
    return rows


def cmd_kinds(paths):
    tabs = []
    for p in paths:
        agg = collections.defaultdict(float)
        for _, ms, n in load(p):
            agg[kind(n)] += ms
        tabs.append((p, agg))
    kinds = sorted({k for _, a in tabs for k in a}, key=lambda k: -sum(a.get(k, 0) for _, a in tabs))
    print("| layer type | " + " | ".join(f"{p.split('/')[-2]} ms (share)" for p, _ in tabs) + " |")
    print("|---|" + "---|" * len(tabs))
    for k in kinds:
        cells = []
        for _, a in tabs:
            tot = sum(a.values())
            cells.append(f"{a.get(k, 0):.1f} ({100 * a.get(k, 0) / tot:.1f} %)")
        print(f"| {k} | " + " | ".join(cells) + " |")
    print("| total kernel ms | " + " | ".join(f"{sum(a.values()):.1f}" for _, a in tabs) + " |")


def cmd_kernels(path, top=14):
    agg = collections.defaultdict(lambda: [0, 0.0])
    for _, ms, n in load(path):
        c = core(n)
        agg[c][0] += 1
        agg[c][1] += ms
    tot = sum(v[1] for v in agg.values())
    print("| kernel | calls | ms | share | avg us |")
    print("|---|---|---|---|---|")
    for c, (n, ms) in sorted(agg.items(), key=lambda x: -x[1][1])[:top]:
        print(f"| `{c}` | {n} | {ms:.1f} | {100 * ms / tot:.1f} % | {1000 * ms / n:.0f} |")


def cmd_chunks(path, rows, n_head=16, d=256, n_layers=10):
    """rows: tokens of each prefill chunk in order, e.g. [4096, 4096, 4096, 4096, 270]."""
    fl = [(ms, n) for _, ms, n in load(path) if "flash_hpb" in n]
    print("| chunk | rows | pos0 | flash ms (10 layers) | us per layer-call | causal GFLOP per layer-call | TFLOP/s |")
    print("|---|---|---|---|---|---|---|")
    pos0 = 0
    for i, t in enumerate(rows):
        part = fl[i * n_layers : (i + 1) * n_layers]
        ms = sum(x[0] for x in part)
        flop = 4 * n_head * d * (t * pos0 + t * (t + 1) / 2)  # 4 * H * D * sum over rows of the keys they see
        print(f"| {i} | {t} | {pos0} | {ms:.1f} | {1000 * ms / len(part):.0f} | {flop / 1e9:.0f} | {flop * len(part) / (ms * 1e-3) / 1e12:.1f} |")
        pos0 += t


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    if sys.argv[1] == "kinds":
        cmd_kinds(sys.argv[2:])
    elif sys.argv[1] == "kernels":
        cmd_kernels(sys.argv[2])
    elif sys.argv[1] == "chunks":
        cmd_chunks(sys.argv[2], [int(x) for x in sys.argv[3].split(",")])
    else:
        sys.exit(__doc__)
