#!/usr/bin/env python3
"""Context-aware byte floor and decode-step composition. Inputs: cb-mode traces (GPU busy per step), attn_bw in-stream split-KV times."""
import re, json, gzip, os
BW = 614e9
WEIGHTS = 15.1368e9                      # bytes streamed per token: 36 x (q 16.78M + k 4.19M + v 4.19M + o 16.78M + gate 50.33M + up 50.33M + down 50.33M) + lm_head 622.3M params, x2 B
KVB = 36 * 8 * 128 * 2 * 2               # per context position: layers x kv heads x head_dim x (K,V) x bf16 = 147,456 B
cb = {4096: "analysis/ctx4096-c2048.md", 16384: "analysis/ctx16384-c2048.md", 32764: "analysis/ctx32768-c2048-a2.md", 40000: "analysis/ctx40000-c2048.md"}
loc = {128: 36.3, 4113: 35.2, 16677: 31.8, 33926: 27.9}  # localeval, intarsia serve (chunk 2048), median of 3, generated 256
attn_us = {4096: 36.3, 16384: 122.0, 32764: 238.6, 40000: 285.1}  # attn_bw splitkv, ctx rounded to the measured point: 32764 measured at 32768
print("| context (mean visible keys over the 128 steps) | KV read/token | bytes/token | floor ms | floor tok/s | cb-mode GPU busy/step | step period | tok/s (cb run) | % of floor | attention 36 x chain us | busy minus attention | idle gap |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for ctx, f in cb.items():
    t = open(f).read()
    m = re.search(r"GPU busy/step ([\d.]+) ms.*?idle between steps ([\d.]+) ms; step period ([\d.]+) ms = ([\d.]+) tok/s", t)
    busy, gap, per, tps = (float(x) for x in m.groups())
    mean_ctx = ctx + 64
    b = WEIGHTS + KVB * mean_ctx
    fl = b / BW * 1e3
    a = attn_us[ctx] * mean_ctx / (ctx if ctx != 32764 else 32768) * 36 / 1e3
    print(f"| {ctx} (+{64}) | {KVB*mean_ctx/1e9:.2f} GB | {b/1e9:.2f} GB | {fl:.2f} | {1e3/fl:.1f} | {busy:.2f} ms | {per:.2f} ms | {tps:.1f} | {100*fl/per:.1f}% | {a:.2f} ms | {busy-a:.2f} ms | {gap:.2f} ms |")
print("\nlocaleval (served, chunk 2048):")
print("| prompt tok | mean ctx during 256 tokens | bytes/token | floor tok/s | decode tok/s | % of floor | % of weights-only floor (40.6) |\n|---:|---:|---:|---:|---:|---:|---:|")
for p, v in loc.items():
    mc = p + 128; b = WEIGHTS + KVB * mc
    print(f"| {p} | {mc} | {b/1e9:.2f} GB | {BW/b:.1f} | {v} | {100*v*b/BW:.1f}% | {100*v/(BW/WEIGHTS):.1f}% |")
