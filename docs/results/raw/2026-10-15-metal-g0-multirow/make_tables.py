#!/usr/bin/env python3
"""Condensed per-shape tables for the report from summary.json (analyze_g0.py output). usage: make_tables.py <dir> [a8]"""
import json, sys, os
d = sys.argv[1]; a8 = sys.argv[2] if len(sys.argv) > 2 else "0"
S = json.load(open(os.path.join(d, "summary.json")))["summary"]
names = {"qo": "q / o (4096x4096)", "down": "down (K=12288, N=4096)", "kv": "k+v pair (2 x 1024x4096)", "gu": "gate+up pair (2 x 12288x4096)", "lm": "lm_head (151936x4096, f32 out)"}
print("| shape | R | T(1) serving us | T(R) existing us (path) | existing / T(1) | T(R) prototype us | prototype / T(1) | bitwise |")
print("|---|---|---|---|---|---|---|---|")
for case in ["qo", "down", "kv", "gu", "lm"]:
    s = S.get(f"{case}/{a8}")
    if not s: continue
    t1 = s["serving1"]["t_med"]
    for R in (2, 3, 4, 5, 6, 7, 8, 16):
        m = s.get(f"mr{R}"); ex = s.get(f"tiled{R}"); path = "tiled GEMM"
        if R == 16 and s.get("nax16"): ex, path = s["nax16"], "NAX GEMM"
        if R == 16 and case == "lm": path = "tiled GEMM (no NAX logits)"
        if not m: continue
        cfg = m["best_p"]
        print(f"| {names[case] if R == 2 else ''} | {R} | {t1*1e6:.1f} | " + (f"{ex['t_med']*1e6:.1f} ({path})" if ex else "-") + " | " + (f"{ex['t_med']/t1:.2f}x" if ex else "-") +
              f" | {m['t_med']*1e6:.1f} (cfg {cfg}) | **{m['ratio_med']:.3f}** | yes |")
