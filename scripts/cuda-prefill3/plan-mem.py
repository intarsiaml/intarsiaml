#!/usr/bin/env python3
"""Memory fields of `intarsia plan show --json` outputs (scripts/cuda-prefill3/admission.sh) side by side, and the scratch
of the planned fp8 GEMM rows.  plan-mem.py <plan-a.json> <plan-b.json> ..."""
import json
import sys


def flat(d, pre=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flat(v, f"{pre}{k}.")
        elif not isinstance(v, list):
            yield f"{pre}{k}", v


plans = {p: json.load(open(p)) for p in sys.argv[1:]}
words = ("scratch", "mem", "kv", "weight", "arena", "bytes", "budget", "resident")
keys = sorted({k for p in plans.values() for k, _ in flat(p) if any(w in k.lower() for w in words)})
for k in keys:
    print(f"{k:<40} " + "  ".join(str(dict(flat(p)).get(k)) for p in plans.values()))
for name, p in plans.items():
    for o in p.get("ops", []):
        if str(o.get("kernel", "")).startswith("matmul_fp8_gemm"):
            print(name, o.get("op"), o.get("kernel"), "scratch", o.get("scratch"))
