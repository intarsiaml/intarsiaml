#!/usr/bin/env python3
"""Lane c27: per-pass time by kernel class for the decode window of a `generate` nsys trace (cuda_gpu_trace csv), memcpy rows included.
    breakdown.py <trace.csv> <decode_wall_seconds> <passes>
Classes: body MXFP4 GEMV, lm_head fp8 rows (verify), MTP layer bf16 GEMV, draft head fp8 GEMV, attention, GDN, D2D memcpy, other kernels, idle (span minus union busy)."""
import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1])))
wall, passes = float(sys.argv[2]), float(sys.argv[3])
ev = []
for r in rows:
    s = float(r['Start (ns)']); ev.append((s, s + float(r['Duration (ns)']), r.get('Name') or '', float(r.get('Bytes (MB)') or 0)))
ev.sort()
cut = max(e[1] for e in ev) - wall * 1e9
dec = [e for e in ev if e[0] >= cut]
def cls(n):
    if n.startswith('[CUDA memcpy Device-to-Device]'): return 'D2D memcpy (GDN state rows)'
    if n.startswith('[CUDA'): return 'other memops'
    if 'mxfp4_gemv' in n: return 'body MXFP4 GEMV (verify rows)'
    if 'fp8_gemv_rows' in n: return 'lm_head fp8 rows (verify)'
    if 'fp8_gemv' in n: return 'draft head fp8 GEMV'
    if 'gemv_pair' in n or n.split('(')[0].strip().endswith('gemv_kernel<') or 'gemv_kernel<' in n: return 'MTP layer bf16 GEMV'
    if 'splitkv' in n or 'combine_kernel' in n or 'attn' in n: return 'attention'
    if 'gdn' in n: return 'GDN (delta, conv, state)'
    return 'norms, rope, kv append, glue'
t = collections.Counter(); c = collections.Counter()
for s, e, n, b in dec:
    k = cls(n); t[k] += e - s; c[k] += 1
busy = 0; end = None; st = None
for s, e, n, b in dec:
    if end is None or s > end:
        if end is not None: busy += end - st
        st, end = s, e
    else: end = max(end, e)
busy += end - st
span = dec[-1][1] - dec[0][0]
tot = sum(t.values())
print(f"decode window {wall*1e3:.0f} ms, {passes:.0f} passes: {wall*1e3/passes:.2f} ms per pass; launches per pass {len(dec)/passes:.0f}")
print(f"{'class':36s} {'ms/pass':>8s} {'share':>6s} {'calls/pass':>10s} {'us/call':>8s}")
for k, v in sorted(t.items(), key=lambda kv: -kv[1]):
    print(f"{k:36s} {v/1e6/passes:8.2f} {100*v/1e6/(wall*1e3):5.1f}% {c[k]/passes:10.1f} {v/1e3/c[k]:8.1f}")
print(f"{'idle (span - union busy)':36s} {(span-busy)/1e6/passes:8.2f} {100*(span-busy)/1e6/(wall*1e3):5.1f}%")
print(f"sum of class times {tot/1e6/passes:.2f} ms/pass (concurrent side-stream work counts twice)")
