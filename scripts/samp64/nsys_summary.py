#!/usr/bin/env python3
"""samp64: per-pass attribution table from the decode-window JSONs of scripts/samp64/nsys_tables.py (passes = launches of the verify-attention rows kernel / 10 full-attention layers)."""
import json, pathlib, sys
raw = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "docs/results/2026-11-20-samp64/raw/nsys-n1")
cells = [("sampled 4k", "T0.7-4096"), ("greedy 4k", "T0-4096"), ("sampled 64k", "T0.7-65536"), ("greedy 64k", "T0-65536")]
rows = {}
for lab, f in cells:
    d = json.load(open(raw / f"{f}-decode-window.json"))
    passes = sum(n for (name, ms, n) in d["top_kernels"] if "splitkv_rows_persist" in name) / 10
    c = d["by_category_ms"]
    rows[lab] = {"passes": passes, "wall": d["decode_wall_ms"] / passes, "busy": d["gpu_busy_ms"] / passes, "idle": (d["decode_wall_ms"] - d["gpu_busy_ms"]) / passes,
                 "attention": c.get("attention (KV read)", 0) / passes, "matmul": c.get("matmul / GEMV (dense, lm head)", 0) / passes, "moe": c.get("MoE experts + routing", 0) / passes,
                 "gdn": c.get("GatedDeltaNet / conv", 0) / passes, "norm": c.get("norm / rope / elementwise / copy", 0) / passes,
                 "sampling": c.get("sampling (top-k, argmax, softmax)", 0) / passes, "other": c.get("other", 0) / passes,
                 "d2h_copies": d["d2h_copies"] / passes, "d2h_mb": d["d2h_bytes"] / 1e6 / passes, "tokens_per_pass": 256 / passes}
cols = [l for l, _ in cells]
print("| ms per verify pass | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
for k, lab in [("passes", "passes in the 256-token window"), ("tokens_per_pass", "tokens per pass in that window"), ("wall", "wall (first to last decode kernel)"), ("busy", "GPU busy (union of kernels)"),
               ("idle", "GPU idle (host gaps)"), ("attention", "attention kernels (verify rows + draft head)"), ("matmul", "dense matmul / GEMV (fp8, nvfp4, bf16, lm head)"), ("moe", "MoE experts + routing"),
               ("gdn", "GatedDeltaNet"), ("norm", "norm / rope / elementwise / copies"), ("sampling", "sampling kernels (argmax, top-k)"), ("other", "other"), ("d2h_copies", "D2H copies per pass"), ("d2h_mb", "D2H MB per pass")]:
    print(f"| {lab} | " + " | ".join(f"{rows[c][k]:.2f}" if k in ("tokens_per_pass", "d2h_copies") else (f"{rows[c][k]:.1f}") for c in cols) + " |")
