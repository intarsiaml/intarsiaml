#!/usr/bin/env python3
"""M4: time spent on the misaligned (8 mod 16, shard 1) weight path at B=8, from a per-launch trace.
Weights of layers 0..~7 sit at 8 mod 16 (Qwen3-8B shard 1, 9,336-byte data offset) and take the slower 8-byte-load multi-row
GEMV at R=8; the other layers take the 16-byte one. For each GEMV class the launches are ordered by layer; the launches
slower than 1.08x the median of the last 20 layers are the misaligned ones, and their excess over that median is what a
perfect remedy could remove.   usage: analyze_align.py TRACE.jsonl TRACE.marks STEP_MS_FOR_SHARE"""
import json, sys, statistics as st, os, gzip

def opn(f):
    return gzip.open(f + '.gz', 'rt') if os.path.exists(f + '.gz') else open(f)

trace, marks, step_ms = sys.argv[1], sys.argv[2], float(sys.argv[3])
recs = [json.loads(l) for l in opn(trace)]
ms = [json.loads(l) for l in open(marks)]
ms = [m for m in ms if not m['warm']]

def cls(r):
    if r['op'].startswith('matmul_pair'): return 'gate+up' if r['n'] > 10000 else 'k+v'
    if r['op'].startswith('matmul_bf16'): return 'down' if r['k'] > 8000 else 'q/o'
    return None

per = {}
for m in ms:
    rs = sorted((r for r in recs if r['s'] >= m['s'] and r['e'] <= m['e']), key=lambda r: r['s'])
    seqs = {}
    for r in rs:
        c = cls(r)
        if c: seqs.setdefault(c, []).append((r['e'] - r['s']) * 1e6)
    for c, v in seqs.items():
        per.setdefault(c, []).append(v)
tot_excess = 0.0
print('| class | launches | slow launches | aligned median us | misaligned median us | excess ms/step |')
print('|---|---:|---:|---:|---:|---:|')
for c, runs in per.items():
    n = len(runs[0])
    med = [st.median(r[i] for r in runs) for i in range(n)]
    base = st.median(med[-20:])
    slow = [i for i, x in enumerate(med) if x > 1.08 * base]
    ex = sum(med[i] - base for i in slow) / 1e3
    tot_excess += ex
    print(f'| {c} | {n} | {len(slow)} | {base:.1f} | {st.median(med[i] for i in slow) if slow else float("nan"):.1f} | {ex:.3f} |')
print(f'\ntotal excess {tot_excess:.3f} ms/step = {100 * tot_excess / step_ms:.2f}% of a {step_ms:.2f} ms step; '
      f'aggregate tok/s gain if removed {100 * (step_ms / (step_ms - tot_excess) - 1):.2f}%')
