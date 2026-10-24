#!/usr/bin/env python3
"""Lane c8k: per-prompt KL / top-1 of the KL-golden logs at chunk 2048 vs 8192 (docs/results/.../raw/gates/ga2), and the gate verdicts.
usage: gate-table.py GA2_DIR"""
import re, sys, os
d = sys.argv[1]
for g in ("vllm-top20", "vllm-long-top20", "vllm-4k-top20"):
    rows = {}
    for c in (2048, 8192):
        t = open(os.path.join(d, f"kl-qwen36-35b-a3b-nvfp4-{g}-c{c}.log"), errors="replace").read()
        gate = re.findall(r"golden gate: (\w+)", t)
        span = re.search(r"prompts spanning more than one chunk: (\d+/\d+)", t)
        for m in re.finditer(r"^(\S+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+\S+\s+\S+\s+([\d.]+)\s+(\d+)/(\d+)\s+(\d+)\s*$", t, re.M):
            rows.setdefault(m.group(1), {})[c] = (float(m.group(4)), float(m.group(6)), float(m.group(7)), int(m.group(10)), int(m.group(11)), m.group(0))
        rows.setdefault("_gate", {})[c] = (gate[-1] if gate else "none", span.group(1) if span else "?")
    print(f"\n{g}: gate 2048 {rows['_gate'][2048][0]} (multi-chunk prompts {rows['_gate'][2048][1]}), 8192 {rows['_gate'][8192][0]} (multi-chunk {rows['_gate'][8192][1]})\n")
    print("| prompt | top-1 % 2048 / 8192 | KL21 2048 / 8192 | KL21 scored 2048 / 8192 | greedy-track 2048 / 8192 (tokens before first divergence) |\n|---|---|---|---|---|")
    for p, v in rows.items():
        if p == "_gate" or 2048 not in v or 8192 not in v:
            continue
        a, b = v[2048], v[8192]
        print(f"| {p} | {a[0]:.2f} / {b[0]:.2f} | {a[1]:.3e} / {b[1]:.3e} | {a[2]:.3e} / {b[2]:.3e} | {a[3]}/{a[4]} / {b[3]}/{b[4]} |")
