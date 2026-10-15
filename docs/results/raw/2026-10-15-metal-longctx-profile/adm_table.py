#!/usr/bin/env python3
"""Per-class prefill time on the clock-admissible chunks only (chunk matmul rate >= 56.4 TFLOPS, the proxy for GPU clock >= 1450 MHz).
Admissible chunks of the 32k / 40k traces are the early (low pos0) ones, so their attention share is a lower bound for the whole prompt."""
import gzip, json, os, collections
D, H, L = 128, 32, 36
def load(p):
    return [json.loads(l) for l in (gzip.open(p, "rt") if p.endswith(".gz") else open(p))]
def cls(op):
    for k, v in (("matmul", "matmul"), ("logits", "matmul"), ("attn", "attention"), ("rmsnorm", "norms/rope"), ("qk_norm", "norms/rope"), ("rope", "norms/rope")):
        if op.startswith(k): return v
    return "other"
tr = [(4096, "kernel/ctx4096-c2048.jsonl.gz"), (16384, "kernel/ctx16384-c2048.jsonl.gz"), (32764, "s4/kernel/ctx32768-c2048-a2.jsonl.gz"), (40000, "kernel/ctx40000-c2048.jsonl.gz")]
print("| context | admissible chunks | pos0 range | matmul ms | attention ms | norms/rope ms | other ms | attention share | matmul TFLOPS | attention TFLOPS |\n|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")
for ctx, p in tr:
    rs = load(p); ps = []; cur = None
    for r in rs:
        if r["op"] == "embedding_bf16": cur = [r]; ps.append(cur)
        elif cur is not None: cur.append(r)
    pos0 = 0; c = collections.Counter(); n = 0; fl = 0.0; mf = 0.0; mt = 0.0; pr = []
    for x in ps:
        if x[0]["t"] <= 1: continue
        T = x[0]["t"]
        cc = collections.Counter(); f_mm = 0.0
        for r in x:
            cc[cls(r["op"])] += r["e"] - r["s"]
            if cls(r["op"]) == "matmul" and r["t"] > 1: f_mm += 2.0 * r["t"] * r["n"] * r["k"]
        t_mm = sum(r["e"] - r["s"] for r in x if r["op"].startswith("matmul"))
        rate = f_mm / t_mm / 1e12
        if rate >= 56.4:
            n += 1; pr.append(pos0)
            for k in cc: c[k] += cc[k]
            fl += L * 4.0 * D * H * (T * pos0 + T * (T + 1) / 2); mf += f_mm; mt += t_mm
        pos0 += T
    tot = sum(c.values())
    print(f"| {ctx} | {n} of {len([x for x in ps if x[0]['t']>1])} | {min(pr)}-{max(pr)} | {c['matmul']*1e3:.0f} | {c['attention']*1e3:.0f} | {c['norms/rope']*1e3:.0f} | {c['other']*1e3:.0f} | {100*c['attention']/tot:.0f}% | {mf/mt/1e12:.1f} | {fl/c['attention']/1e12:.1f} |")
