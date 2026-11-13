#!/usr/bin/env python3
"""ab35b: print the run 1 markdown tables from the replay summary.json files. usage: tables.py RAW_DIR"""
import json, sys, pathlib
raw = pathlib.Path(sys.argv[1])
arms = [("A", "A vLLM"), ("B-default", "B batch2 (--max-batch 2, MTP, no reuse)"), ("B-single", "B single (default engine, MTP, held-state reuse)"), ("B-store", "B store (--prefix-store on, plain decode)")]
for order, seed in (("interleaved", 0), ("sequential", 1)):
    print(f"\n#### {order} (seed {seed})\n")
    print("| turn | prompt tok | " + " | ".join(f"{n}: cached tok, TTFT s, decode tok/s" for _, n in arms) + " |")
    print("|---|---|" + "---|" * len(arms))
    S = {a: json.load(open(raw / f"{a}-{order}-seed{seed}.summary.json")) for a, _ in arms}
    for t in range(6):
        cells = []
        for a, _ in arms:
            r = S[a]["turns"][t]
            cells.append(f"{r['cached_tokens_median']:.0f} ({r['cached_frac_median']:.2f}), {r['ttft_s_median']} ({r['ttft_s_range'][0]}-{r['ttft_s_range'][1]}), {r['decode_tok_s_median']}")
        print(f"| {t+1} | {S['A']['turns'][t]['prompt_tokens_median']:.0f} | " + " | ".join(cells) + " |")
    print("\n| arm | requests | uncached prompt tok | sum TTFT s | median decode tok/s | wall s |\n|---|---|---|---|---|---|")
    for a, n in arms:
        T = S[a]["totals"]
        print(f"| {n} | {T['requests']} | {T['prompt_tokens']-T['cached_tokens']} of {T['prompt_tokens']} | {T['ttft_s_sum']} | {T['decode_tok_s_median']} | {S[a]['wall_s']} |")
