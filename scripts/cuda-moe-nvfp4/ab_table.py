#!/usr/bin/env python3
"""Tables of the window-B speed A/B (docs/results/2026-10-17-cuda-moe-nvfp4-mma/raw/wB/ab): base vs FP4 MoE, 3 rounds."""
import json
import sys

base = (sys.argv[1] if len(sys.argv) > 1 else "docs/results/2026-10-17-cuda-moe-nvfp4-mma/raw/wB/ab") + "/"


def rows(arm, r):
    d = json.load(open(f"{base}35b-{arm}-r{r}/cudadecode-35b-{arm}-r{r}.json"))
    return {int(x["target_prompt_tokens"]): x for x in d["rows"]}


R = {a: [rows(a, r) for r in (1, 2, 3)] for a in ("base", "dev")}
for key, name in (("prefill_tok_s", "prefill tok/s"), ("decode_tok_s", "decode tok/s"), ("ttft_s", "TTFT s")):
    print(f"### {name}")
    print("| prompt tokens | base r1, r2, r3 | mean | FP4 r1, r2, r3 | mean | delta | per round |")
    print("|---|---|---|---|---|---|---|")
    for s in (1024, 4096, 16384):
        b = [R["base"][i][s][key] for i in range(3)]
        f = [R["dev"][i][s][key] for i in range(3)]
        mb, mf = sum(b) / 3, sum(f) / 3
        per = ", ".join(f"{100 * (f[i] / b[i] - 1):+.1f}%" for i in range(3))
        print(f"| {s} | {', '.join(f'{x:g}' for x in b)} | **{mb:.4g}** | {', '.join(f'{x:g}' for x in f)} | **{mf:.4g}** | {100 * (mf / mb - 1):+.1f}% | {per} |")
    print()
