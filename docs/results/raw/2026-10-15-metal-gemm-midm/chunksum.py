#!/usr/bin/env python3
"""Chunk 256 vs 2048, 4k prompt, in-engine kernel traces (chunk/k-4062-c<C>-r<n>.jsonl): GPU ms per kernel class (summed over the
prefill, median over rounds), TFLOPS of the NAX GEMMs, and the GPU clock (macmon, busy samples).  python3 chunksum.py"""
import glob, json, re, statistics, collections
cls = {(4096, 4096): "qo", (2048, 4096): "kv(pair)", (24576, 4096): "gate+up(pair)", (4096, 12288): "down"}
flops = {"qo": 4096 * 4096, "kv(pair)": 2048 * 4096, "gate+up(pair)": 24576 * 4096, "down": 4096 * 12288}
tot = collections.defaultdict(lambda: collections.defaultdict(list))   # chunk -> class -> [ms per run]
rows = collections.defaultdict(lambda: collections.defaultdict(int))   # chunk -> class -> rows processed (M summed)
clk = collections.defaultdict(list)
runs = collections.defaultdict(list)
for f in sorted(glob.glob("chunk/k-4062-c*-r*.jsonl")):
    m = re.fullmatch(r"chunk/k-4062-c(\d+)-r(\d+)\.jsonl", f)
    if not m: continue
    C, r = int(m.group(1)), int(m.group(2))
    per = collections.defaultdict(float); cnt = collections.defaultdict(int); rws = collections.defaultdict(int)
    for l in open(f):
        x = json.loads(l)
        us = (x["e"] - x["s"]) * 1e3
        op = x["op"]
        if op in ("matmul_bf16_nax", "matmul_pair_bf16_nax") and x["t"] > 1:
            c = cls.get((x["n"], x["k"]), "other-gemm")
            per["GEMM " + c] += us; rws["GEMM " + c] += x["t"]
        elif op.startswith("attn_prefill"):
            per["attention"] += us
        else:
            per["other (norms, rope, glu, logits)"] += us
    for k, v in per.items():
        tot[C][k].append(v); rows[C][k] = rws[k]
    runs[C].append(sum(per.values()))
    fr = []
    for l in open(f.replace(".jsonl", ".macmon.jsonl")):
        try: j = json.loads(l)
        except Exception: continue
        if j["gpu_usage"][1] >= 0.5: fr.append(j["gpu_usage"][0])
    clk[C] += fr
    if glob.glob(f.replace(".jsonl", ".contaminated")): print("CONTAMINATED", f)
Cs = sorted(tot)
print("class | " + " | ".join(f"c{C} ms (n={len(runs[C])})" for C in Cs) + " | c256/c2048 | TFLOPS " + " / ".join(f"c{C}" for C in Cs))
for k in sorted(set().union(*[tot[C].keys() for C in Cs]), key=lambda k: -statistics.median(tot[Cs[0]].get(k, [0]))):
    ms = {C: statistics.median(tot[C][k]) for C in Cs if k in tot[C]}
    tf = ""
    if k.startswith("GEMM ") and k[5:] in flops:
        n_k = flops[k[5:]]
        tf = " / ".join(f"{2 * rows[C][k] * n_k / (ms[C] * 1e-3) / 1e12:.1f}" for C in Cs)
    print(f"{k} | " + " | ".join(f"{ms.get(C, 0):.1f}" for C in Cs) + f" | {ms[Cs[0]] / ms[Cs[-1]]:.2f} | {tf}")
print("total kernel ms | " + " | ".join(f"{statistics.median(runs[C]):.1f} [{min(runs[C]):.0f}..{max(runs[C]):.0f}]" for C in Cs) + f" | {statistics.median(runs[Cs[0]]) / statistics.median(runs[Cs[-1]]):.3f}")
print("GPU MHz (busy samples) | " + " | ".join(f"{statistics.median(clk[C]) if clk[C] else 0:.0f} (n={len(clk[C])})" for C in Cs))
