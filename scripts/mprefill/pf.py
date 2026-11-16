#!/usr/bin/env python3
"""Prefill attribution from a cb trace with one command buffer per dispatch (pf.sh). Exclusive time = time after the previous
buffer ended. The prefill phase = every record up to the last one with t >= 64 rows (the 1..2 decode steps after it are
dropped). Prints a per-op table and a per-class table (GEMM, weight decode to scratch, activation prep, attention, GDN, MoE,
norms, other).  usage: pf.py trace.jsonl [top]"""
import json, sys, collections, re

import gzip
recs = [json.loads(l) for l in (gzip.open(sys.argv[1], "rt") if sys.argv[1].endswith(".gz") else open(sys.argv[1])) if l.strip().startswith("{")]
last = max(i for i, r in enumerate(recs) if r.get("t", 0) >= 64)
rs = recs[: last + 1]
# the three dispatches of a quantised NAX matmul: base = activation prep (mxfp4 / fp8) or weight decode (nvfp4), #1 ...
def klass(op, r):
    base, _, sub = op.partition("#")
    if base.endswith("_nax") and base.startswith("matmul_"):
        two = base == "matmul_nvfp4_nax"  # nvfp4: [w decode, GEMM]; mxfp4 / fp8: [x prep, w decode, GEMM]
        idx = int(sub) if sub else 0
        if two:
            return "weight decode -> bf16 scratch" if idx == 0 else "NAX GEMM (quantised weights)"
        return ["activation prep (x QDQ / copy)", "weight decode -> bf16 scratch", "NAX GEMM (quantised weights)"][idx]
    if re.search(r"attn|flash", base):
        return "attention"
    if re.search(r"gdn|conv", base):
        return "GDN (conv, delta rule, gates)"
    if base.startswith("moe"):
        return "MoE (router top-k, experts, combine)"
    if re.search(r"matmul|gemm|logits", base):
        return "other GEMM / GEMV (bf16, pair, lm_head)"
    if re.search(r"norm", base):
        return "norms"
    return "other (rope, kv append, silu, copies, embed)"

t = collections.Counter(); c = collections.Counter(); end = rs[0]["s"]; idle = 0.0
for r in rs:
    s = max(r["s"], end); idle += max(0.0, r["s"] - end); d = max(0.0, r["e"] - s)
    t[r["op"]] += d; c[r["op"]] += r["nd"]; end = max(end, r["e"])
tot = sum(t.values()); span = rs[-1]["e"] - rs[0]["s"]
rows = sum(r["t"] for r in rs if r["op"].startswith("attn_prefill") or r["op"].startswith("flash"))
print(f"{sys.argv[1].split('/')[-1]}: {len(rs)} buffers, GPU busy {tot*1e3:.0f} ms, idle {idle*1e3:.0f} ms, span {span*1e3:.0f} ms")
k = collections.Counter()
for o, v in t.items():
    k[klass(o, None)] += v
print("\n| class | ms | share of busy |\n|---|---:|---:|")
for o, v in k.most_common():
    print(f"| {o} | {v*1e3:.0f} | {100*v/tot:.1f} % |")
print("\n| op | launches | ms | share | us / launch |\n|---|---:|---:|---:|---:|")
for o, v in t.most_common(int(sys.argv[2]) if len(sys.argv) > 2 else 30):
    print(f"| {o} | {c[o]} | {v*1e3:.1f} | {100*v/tot:.1f} % | {v*1e6/c[o]:.1f} |")
# per weight shape for the quantised NAX matmuls: decode vs GEMM per call, and GEMM TFLOPS
sh = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0])  # calls, prep, decode, gemm, rows
ops = list(rs)
for i, r in enumerate(ops):
    base, _, sub = r["op"].partition("#")
    if not (base.startswith("matmul_") and base.endswith("_nax")):
        continue
    key = (base, r["n"], r["k"])
    d = max(0.0, r["e"] - r["s"])
    cls = klass(r["op"], r)
    e = sh[key]
    if cls.startswith("activation"):
        e[1] += d
    elif cls.startswith("weight"):
        e[2] += d
        e[0] += 1; e[4] += r["t"]
    else:
        e[3] += d
print("\n| matmul | N x K (unpacked) | calls | rows / call | x prep ms | w decode ms | GEMM ms | decode share | GEMM TFLOPS |\n|---|---|---:|---:|---:|---:|---:|---:|---:|")
for (b, n, kk), e in sorted(sh.items(), key=lambda x: -(x[1][1] + x[1][2] + x[1][3])):
    calls, prep, dec, gm, rws = e
    kr = kk * 2 if b in ("matmul_mxfp4_nax", "matmul_nvfp4_nax") else kk  # 4-bit weights: the trace's k is the packed K / 2
    tf = 2 * rws * n * kr / gm / 1e12 if gm else 0
    print(f"| {b} | {n} x {kr} | {calls} | {rws/max(calls,1):.0f} | {prep*1e3:.1f} | {dec*1e3:.1f} | {gm*1e3:.1f} | {100*dec/max(prep+dec+gm,1e-12):.0f} % | {tf:.1f} |")
