#!/usr/bin/env python3
"""Class table from a per-dispatch trace (disp2.py's step filter). usage: classes.py TRACE CTX SPAN_MS"""
import sys, json, collections, statistics as st
recs = sorted((json.loads(l) for l in open(sys.argv[1]) if l.strip()), key=lambda r: r["s"])
ctx, span = int(sys.argv[2]), float(sys.argv[3])
starts = [i for i, r in enumerate(recs) if r["op"].startswith("embedding")]
segs = [recs[a:b] for a, b in zip(starts, starts[1:])][2:]
nd0 = st.median([len(s) for s in segs]); segs = [s for s in segs if len(s) == nd0]
def cls(op, r):
    if op.startswith("logits"): op = "router_gemv"
    if op.startswith(("conv_mix", "loop_blend", "attn_res_mix")): return "controllers: ConvMix row + blend, LoopBlend, AttnResMix"
    if op.startswith(("ink_route", "router_gemv")): return "router (ink_route)"
    if op.startswith(("ink_attn_decode", "rel_sink_attn", "ink_kv_prologue")): return "attention core (fused decode; full visits split at 4k)"
    if op.startswith("moe_experts_situ"): return "experts NVFP4 (gate+up, down)"
    if op.startswith("moe_experts_mxfp8"): return "experts MXFP8"
    if op.startswith(("ink_in_proj", "matmul_p12_gemv", "matmul_bf16_gemv")): return "attention projections (in_proj, wo; P12)"
    if op.startswith(("matmul_fp8_gemv", "head_screen", "screen_argmax")): return "LM head (fp8 screen + gather + argmax)"
    return "norms, embedding, other"
# MB per token (dfuse roofline L, raw/L_roofline.txt; the head re-counted for the fp8 screen: 461 M weights x 1 B)
MB = {"controllers: ConvMix row + blend, LoopBlend, AttnResMix": 113, "router (ink_route)": 28,
      "attention core (fused decode; full visits split at 4k)": 48 * min(ctx, 1024) * 4096 / 1e6 if ctx <= 1024 else (40 * 1024 + 8 * ctx) * 4096 / 1e6,
      "experts NVFP4 (gate+up, down)": 1577, "experts MXFP8": 263, "attention projections (in_proj, wo; P12)": 1950,
      "LM head (fp8 screen + gather + argmax)": 461 + 8, "norms, embedding, other": 8}
agg = collections.defaultdict(float); nd = collections.defaultdict(float)
for s in segs:
    for r in s:
        c = cls(r["op"], r); agg[c] += (r["e"] - r["s"]) * 1e6 / len(segs); nd[c] += r["nd"] / len(segs)
tot = sum(agg.values()); sc = span * 1e3 / tot
print(f"steps {len(segs)}, per-dispatch busy {tot / 1e3:.3f} ms, normal-step span {span:.3f} ms (scale {sc:.3f}), floor at 560 GB/s")
print("| class | per-dispatch us / token | scaled to the real step | dispatches | MB / token | floor (ms) | excess (ms, scaled) |")
print("|---|---:|---:|---:|---:|---:|---:|")
F = E = 0
for c, v in sorted(agg.items(), key=lambda kv: -kv[1]):
    f = MB[c] / 560; F += f; E += v * sc / 1e3 - f
    print(f"| {c} | {v:.0f} | {v * sc:.0f} | {nd[c]:.0f} | {MB[c]:.0f} | {f:.3f} | **{v * sc / 1e3 - f:.3f}** |")
print(f"| **sum** | {tot:.0f} | {tot * sc:.0f} | {sum(nd.values()):.0f} | {sum(MB.values()):.0f} | {F:.3f} | **{E:.3f}** |")
