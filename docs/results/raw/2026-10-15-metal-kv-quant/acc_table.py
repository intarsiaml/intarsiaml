#!/usr/bin/env python3
"""Accuracy table from acc/*.out: per KV dtype and golden, prefill / decode top-1 and full-vocab KL (bf16 gate class)."""
import re, glob, os
R = os.path.dirname(os.path.abspath(__file__)) + "/acc"
print(f"{'kv':<6} {'golden':<6} {'mode':<8} {'top-1 %':>8} {'KL':>10}   verdict")
for f in sorted(glob.glob(R + "/*.out")):
    name = os.path.basename(f)[:-4]; kv, g = name.split("-")
    t = open(f).read()
    for mode, top1, kl, rest in re.findall(r"(prefill|decode)[^\n]*floor \[bf16\]: top-1 ([\d.]+)%[^\n]*?full-vocab KL ([\d.e-]+)([^\n]*)", t):
        ok = "FAIL" if "FAIL" in rest or "FAIL" in re.findall(r"top-1 [\d.]+%[^|]*\|?", t)[0:0] else ""
        print(f"{kv:<6} {g:<6} {mode:<8} {top1:>8} {kl:>10}")
    m = re.search(r"golden gate: (\w+)", t)
    print(f"{kv:<6} {g:<6} verdict  {m[1] if m else '?'}")
