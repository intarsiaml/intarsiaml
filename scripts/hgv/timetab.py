#!/usr/bin/env python3
"""Lane hgv: markdown before/after tables from two gate.sh timing runs.
  timetab.py raw/gate-HHMM   (reads time-gemv-hgvbase.log / time-gemv-hgv.log, time-mxfp4-*, time-moe-*)"""
import re, sys, statistics as st

d = sys.argv[1]


def times(name):
    out = {}
    for tree in ("hgvbase", "hgv"):
        for line in open(f"{d}/{name}-{tree}.log", errors="replace"):
            m = re.match(r"TIME (\S+ \S+)\s+(\S+)\s+M=(\d+)\s+([\d.]+) us", line)
            if m:
                out[(tree, m[1], m[2], int(m[3]))] = float(m[4])
    return out


g = times("time-gemv")
print("### bf16 GEMV rows (us per launch, back-to-back, `matmul_bf16_gemv_rows`)\n\n| matrix | M | main | hgv | ratio |\n|---|---:|---:|---:|---:|")
for (tree, what, kern, m), v in sorted(g.items(), key=lambda kv: (kv[0][1], kv[0][3])):
    if tree == "hgvbase" and kern == "matmul_bf16_gemv_rows" and what.startswith("bf16"):
        h = g.get(("hgv", what, kern, m))
        if h:
            print(f"| {what} | {m} | {v:.2f} | {h:.2f} | {h / v:.2f} |")
x = times("time-mxfp4")
print("\n### lm_head 248320x5120 bf16 -> f32 (`logits_bf16_gemv_rows`, us)\n\n| M | main | hgv | ratio |\n|---:|---:|---:|---:|")
for (tree, what, kern, m), v in sorted(x.items(), key=lambda kv: kv[0][3]):
    if tree == "hgvbase" and what.startswith("lm_head") and kern == "logits_bf16_gemv_rows":
        h = x[("hgv", what, kern, m)]
        print(f"| {m} | {v:.1f} | {h:.1f} | {h / v:.3f} |")


def moe(tree):
    res, cur = {}, None
    for line in open(f"{d}/time-moe-{tree}.log", errors="replace"):
        m = re.match(r"moe_rows_timing T=(\d+)", line)
        if m:
            cur = int(m[1])
        m = re.match(r"\s+\((\w)\) ([\w +x=.,()-]+?)\s{2,}([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", line)
        if m and cur:
            res[(cur, m[1], m[2].strip())] = st.median(float(m[i]) for i in (3, 4, 5))
    return res


a, b = moe("hgvbase"), moe("hgv")
print("\n### MoE block pieces at the 35B shape (us per layer, median of 3 rounds, graph replay of 16 layers)\n\n| T | piece | main | hgv | ratio |\n|---:|---|---:|---:|---:|")
for (t, k, name), v in sorted(a.items()):
    if k in "cbg" and (t, k, name) in b:
        print(f"| {t} | ({k}) {name} | {v:.1f} | {b[(t, k, name)]:.1f} | {b[(t, k, name)] / v:.2f} |")
