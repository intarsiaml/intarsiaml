#!/usr/bin/env python3
"""Lane c8k: ITL table of itl-window.sh. usage: itl-table.py RAWDIR(i1)"""
import json, os, statistics, sys
raw = sys.argv[1]
arms = [("i2048", "chunk 2048"), ("i8192", "chunk 8192"), ("i8192b2048", "chunk 8192 + --busy-chunk 2048")]
f = lambda xs: f"{statistics.median(xs):.0f} ({', '.join(f'{x:.0f}' for x in xs)})"
for size in ("4k", "16k"):
    print(f"\n{size} prompts, 4 streams (ms; median of rounds, per-round values in brackets)\n")
    print("| arm | scenario | pooled ITL p50 | p99 | max | agg decode tok/s |\n|---|---|---:|---:|---:|---:|")
    for a, name in arms:
        for sc in ("burst", "stagger"):
            v = {k: [] for k in ("p50", "p99", "max", "agg")}
            for r in (1, 2, 3):
                p = os.path.join(raw, f"itl{size}-{a}-r{r}.json")
                if os.path.exists(p):
                    s = json.load(open(p))["summary"].get(sc)
                    if s:
                        v["p50"].append(s["pooled_itl_p50_ms_median"]); v["p99"].append(s["pooled_itl_p99_ms_median"])
                        v["max"].append(s["pooled_itl_max_ms_median"]); v["agg"].append(s["aggregate_decode_tok_s_median"])
            if v["p99"]:
                print(f"| {name} | {sc} | {f(v['p50'])} | {f(v['p99'])} | {f(v['max'])} | {f(v['agg'])} |")
