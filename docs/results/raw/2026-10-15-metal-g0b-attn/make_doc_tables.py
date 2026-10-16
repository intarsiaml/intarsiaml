#!/usr/bin/env python3
"""Tables of docs/results/2026-10-15-metal-g0b-attn.md from final/summary_g0b.json (markdown on stdout, sections split by '@@name')."""
import json, subprocess, sys
S = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "final/summary_g0b.json"))
print("@@attn")
print("| ctx | k | T(1) serving (us) | T(k) rows kernel (us) | T(k)/T(1) [p10-p90] | k sequential 1-row launches / T(1) | today's AttnPrefill (flash) / T(1) |")
print("|---|---|---|---|---|---|---|")
for ctx in (128, 1024, 4096, 16384):
    for k in range(2, 9):
        r = S[f"attn/{ctx}/rows{k}"]; s = S.get(f"attn/{ctx}/seq{k}"); f = S.get(f"attn/{ctx}/flash{k}")
        print(f"| {ctx} | {k} | {S[f'attn/{ctx}/serving1']['t_us']:.1f} | {r['t_us']:.1f} | **{r['ratio']:.2f}** [{r['p10']:.2f}-{r['p90']:.2f}] | {('%.2f' % s['ratio']) if s else '-'} | {('%.1f' % f['ratio']) if f else '-'} |")
print("@@small")
ops = [("norm_add", "RmsNormAdd [k,4096]", "2 per layer"), ("rmsnorm", "RmsNorm [k,4096]", "1 per step"), ("qkrope", "QkNormRopeKv T=k (32/8/128, `_pf` form)", "1 per layer"),
       ("silu", "SiluGlu [k,12288]", "1 per layer"), ("add", "Add [k,4096]", "0 (fused into RmsNormAdd)"), ("embedding", "Embedding [k] -> [k,4096]", "1 per step"), ("argmax", "Argmax f32 [k,151936]", "1 per step")]
print("| op | per decode step | T(1) us | T(2) | T(4) | T(6) | T(8) |")
print("|---|---|---|---|---|---|---|")
for key, label, use in ops:
    c = lambda k: S[f"small/0/{key}{k}"]
    print(f"| {label} | {use} | {c(1)['t_us']:.2f} | " + " | ".join(f"{c(k)['t_us']:.2f} ({c(k)['ratio']:.2f}x)" for k in (2, 4, 6, 8)) + " |")
