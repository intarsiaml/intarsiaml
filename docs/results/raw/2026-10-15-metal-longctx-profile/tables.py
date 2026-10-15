#!/usr/bin/env python3
"""Prefill class table (raw and at the reference clock) and the decode floor/composition table from the kernel / cb traces."""
import gzip, json, os, sys, collections
D, H, L, DK = 128, 32, 36, 36
def load(p):
    if not os.path.exists(p) and os.path.exists(p + ".gz"): p += ".gz"
    return [json.loads(l) for l in (gzip.open(p, "rt") if p.endswith(".gz") else open(p))]
CL = {"matmul": "matmul", "logits": "matmul", "attn": "attention", "rmsnorm": "norms/rope", "qk_norm": "norms/rope", "rope": "norms/rope"}
def cls(op):
    for k, v in CL.items():
        if op.startswith(k): return v
    return "other"
def passes(rs):
    out, cur = [], None
    for r in rs:
        if r["op"] == "embedding_bf16": cur = [r]; out.append(cur)
        elif cur is not None: cur.append(r)
    return out
traces = {4096: "kernel/ctx4096-c2048.jsonl", 16384: "kernel/ctx16384-c2048.jsonl", 32764: "s2/kernel/ctx32768-c2048-a2.jsonl", 40000: "kernel/ctx40000-c2048.jsonl"}
print("| context | chunks | matmul ms | attention ms | norms/rope ms | other ms | kernel total ms | attention share | matmul TFLOPS | attention TFLOPS | same, at the reference clock: matmul ms | attention ms | total ms | attention share | attention TFLOPS |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for ctx, p in traces.items():
    ps = [x for x in passes(load(p)) if x[0]["t"] > 1]
    c = collections.Counter(); pos0 = 0; mmfl = atfl = 0.0; mm_t = at_t = 0.0
    rows = []
    for x in ps:
        T = x[0]["t"]
        cc = collections.Counter()
        for r in x:
            d = r["e"] - r["s"]; k = cls(r["op"]); cc[k] += d; c[k] += d
            if k == "matmul" and r["t"] > 1: mmfl += 2.0 * r["t"] * r["n"] * r["k"]; mm_t += d
        fl = L * 4.0 * D * H * (T * pos0 + T * (T + 1) / 2)
        atfl += fl
        rows.append((cc, fl, sum(2.0 * r["t"] * r["n"] * r["k"] for r in x if r["op"].startswith("matmul")) , T)); pos0 += T
    best = min(cc["matmul"] / mf for cc, fl, mf, T in rows)
    n = collections.Counter()
    for cc, fl, mf, T in rows:
        f = (cc["matmul"] / mf) / best
        for k in cc: n[k] += cc[k] / f
    tot = sum(c.values()); totn = sum(n.values())
    print(f"| {ctx} | {len(ps)} | {c['matmul']*1e3:.0f} | {c['attention']*1e3:.0f} | {c['norms/rope']*1e3:.0f} | {c['other']*1e3:.0f} | {tot*1e3:.0f} | {100*c['attention']/tot:.0f}% | {mmfl/mm_t/1e12:.1f} | {atfl/c['attention']/1e12:.1f} | {n['matmul']*1e3:.0f} | {n['attention']*1e3:.0f} | {totn*1e3:.0f} | {100*n['attention']/totn:.0f}% | {atfl/n['attention']/1e12:.1f} |")
