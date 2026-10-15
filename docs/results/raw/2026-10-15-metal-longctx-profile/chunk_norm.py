#!/usr/bin/env python3
"""Clock-normalised prefill attention from a kernel-mode trace. The matmuls of a chunk are identical in every chunk, so their time
per chunk is a clock/power-state thermometer. Attention time is scaled by (best matmul chunk time / this chunk's matmul time).
usage: chunk_norm.py <kernel.jsonl[.gz]> <chunk>   -> markdown table + totals"""
import sys, gzip, os, json
D, H, L = 128, 32, 36
p, chunk = sys.argv[1], int(sys.argv[2])
if not os.path.exists(p) and os.path.exists(p + ".gz"): p += ".gz"
rs = [json.loads(l) for l in (gzip.open(p, "rt") if p.endswith(".gz") else open(p))]
passes, cur = [], None
for r in rs:
    if r["op"] == "embedding_bf16": cur = [r]; passes.append(cur)
    elif cur is not None: cur.append(r)
pre = [x for x in passes if x[0]["t"] > 1]
rows = []; pos0 = 0
for x in pre:
    T = x[0]["t"]
    mm = sum(r["e"] - r["s"] for r in x if r["op"].startswith("matmul"))
    mmfl = sum(2.0 * r["t"] * r["n"] * r["k"] for r in x if r["op"].startswith("matmul"))
    at = sum(r["e"] - r["s"] for r in x if r["op"].startswith("attn_prefill"))
    ot = sum(r["e"] - r["s"] for r in x if not (r["op"].startswith("matmul") or r["op"].startswith("attn_prefill") or r["op"].startswith("logits")))
    fl = L * 4.0 * D * H * (T * pos0 + T * (T + 1) / 2)
    rows.append((pos0, T, mm, mmfl, at, ot, fl)); pos0 += T
full = [r for r in rows if r[1] == chunk]
best = min(r[2] / r[3] for r in rows)  # s per FLOP of the fastest chunk
print(f"matmul best {1/best/1e12:.1f} TFLOPS (reference clock state)\n")
print("| pos0 | matmul ms | matmul TFLOPS | attention ms | attention TFLOPS raw | attention TFLOPS at reference clock |\n|---:|---:|---:|---:|---:|---:|")
sa = sf = 0.0; na = 0; tn = 0.0; ta = 0.0; tm = 0.0; tmn = 0.0; tfl = 0.0
for pos0, T, mm, mmfl, at, ot, fl in rows:
    f = (mm / mmfl) / best  # slowdown vs reference (>= 1)
    an = at / f
    adm = (mmfl / mm / 1e12) >= 56.4
    print(f"| {pos0} | {mm*1e3:.0f} | {mmfl/mm/1e12:.1f} | {at*1e3:.0f} | {fl/at/1e12:.1f}{' (clock >= 1450 proxy)' if adm else ''} | {fl/an/1e12:.1f} |")
    if adm: sa += at; sf += fl; na += 1
    ta += at; tn += an; tm += mm; tmn += mm / f; tfl += fl
print(f"\nTotals: attention {ta*1e3:.0f} ms raw, {tn*1e3:.0f} ms at the reference clock ({tfl/tn/1e12:.1f} TFLOPS); matmul {tm*1e3:.0f} ms raw, {tmn*1e3:.0f} ms at the reference clock.")

print(f"Admissible chunks (matmul rate >= 56.4 TFLOPS, the proxy for a GPU clock >= 1450 MHz): {na} of {len(rows)}; attention there {sf/sa/1e12:.1f} TFLOPS raw (no rescaling)." if na else "No admissible chunk.")
