#!/usr/bin/env python3
"""Lane srv16: whole-burst wall-time attribution from an nsys cuda_gpu_trace csv (prof.sh). The burst is the last cluster of GPU ops
(>= 1 s idle splits clusters); a pass is a run of ops between idle gaps >= 300 us. A pass that runs a prefill kernel (the W4A16
expert GEMM, a cuBLASLt nvjet / cutlass GEMM, the chunked GDN conv) is prefill, else decode. Prints the burst wall split into
decode passes, prefill passes and idle gaps, and the decode pass span by burst segment.
  timeline.py trace.csv"""
import csv, sys
PF = ("moe_w4a16_kernel", "nvjet", "cutlass", "gdn_conv_kernel", "flash_prefill", "sm90_", "sm80_")
ev = []
for r in csv.DictReader(open(sys.argv[1])):
    try: ev.append((float(r["Start (ns)"]), float(r["Duration (ns)"]), r.get("Name") or ""))
    except (KeyError, ValueError): pass
ev.sort()
cl, cur, last = [], [], None
for e in ev:
    if last is not None and e[0] - last >= 1e9: cl.append(cur); cur = []
    cur.append(e); last = max(last or 0, e[0] + e[1])
cl.append(cur); q = cl[-1]
passes, cur, end = [], [], None
for e in q:
    if end is not None and e[0] - end >= 300e3: passes.append(cur); cur = []
    cur.append(e); end = max(end or 0, e[0] + e[1])
passes.append(cur)
t0 = q[0][0]; t1 = max(e[0] + e[1] for e in q); wall = t1 - t0
rows = []
for p in passes:
    s, e = p[0][0], max(x[0] + x[1] for x in p)
    pf = any(x[2].startswith(PF) or any(k in x[2] for k in PF) for x in p)
    rows.append((s, e, pf, sum(x[1] for x in p), len(p)))
gap = sum(b[0] - a[1] for a, b in zip(rows, rows[1:]))
dec = [r for r in rows if not r[2]]; pre = [r for r in rows if r[2]]
ms = lambda x: x / 1e6
print(f"burst wall {ms(wall):.0f} ms, {len(rows)} passes, GPU kernel time {ms(sum(r[3] for r in rows)):.0f} ms")
for nm, rs in (("decode", dec), ("prefill", pre)):
    sp = sum(r[1] - r[0] for r in rs)
    print(f"  {nm:8s} passes {len(rs):4d}  span {ms(sp):7.0f} ms ({100 * sp / wall:5.1f} %)  mean {ms(sp / max(1, len(rs))):6.2f} ms")
print(f"  idle gaps (>= 300 us) {ms(gap):7.0f} ms ({100 * gap / wall:5.1f} %)  mean {ms(gap / max(1, len(rows) - 1)) * 1e3:.0f} us")
# first prefill to last prefill: how long the burst prefills, and what decode did meanwhile
if pre:
    a, b = pre[0][0], pre[-1][1]
    d_in = [r for r in dec if a <= r[0] <= b]
    print(f"prefill phase {ms(b - a):.0f} ms: prefill {ms(sum(r[1] - r[0] for r in pre)):.0f} ms, decode {len(d_in)} passes {ms(sum(r[1] - r[0] for r in d_in)):.0f} ms")
    after = [r for r in dec if r[0] > b]
    if after:
        n = len(after); th = [after[: n // 3], after[n // 3: 2 * n // 3], after[2 * n // 3:]]
        print("decode after the last prefill, mean span by thirds (ms):", " / ".join(f"{ms(sum(r[1] - r[0] for r in t) / max(1, len(t))):.2f}" for t in th), f"({n} passes)")
    spans = sorted(ms(r[1] - r[0]) for r in pre)
    print("prefill pass spans ms:", " ".join(f"{x:.0f}" for x in spans))
