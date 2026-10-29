#!/usr/bin/env python3
"""Lane vprof: per-step kernel buckets of a vLLM B=16 decode burst from an nsys cuda_gpu_trace csv(.gz) (tail.py output).
Buckets use the names of scripts/step16/buckets.py so the table lines up 1:1 with the engine's.
  vsteps.py trace.csv.gz <cluster_index> <step_from> <step_to> [out_prefix]
A decode step = from its embedding kernel (indexSelect*) to the next step's. Steps are found by the 80 marlin_moe_wna16 (grid 144)
launches per step (gate_up + down x 40 layers)."""
import csv, gzip, re, sys, statistics as st
path, ci, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rows = list(csv.DictReader(gzip.open(path, 'rt') if path.endswith('.gz') else open(path)))
for r in rows: r['s'] = float(r['Start (ns)']); r['d'] = float(r['Duration (ns)'])
rows.sort(key=lambda r: r['s'])
cl, cur, last = [], [], None
for r in rows:
    if last is not None and r['s'] - last >= 1e9: cl.append(cur); cur = []
    cur.append(r); last = max(last or 0, r['s'] + r['d'])
cl.append(cur)
big = [c for c in cl if len(c) > 200000]
c = big[ci]
def name(r): return re.sub(r'^void ', '', r['Name'])
def bucket(r, aux):
    n = name(r); g = (r['GrdX'], r['GrdY'], r['GrdZ'])
    if aux: return 'shared expert + shared gate (aux stream)'
    if n.startswith('marlin_moe_wna16'): return 'routed MoE (gate_up, down)'
    if 'gated_delta_rule' in n or 'causal_conv1d' in n: return 'GDN (delta, conv)'
    if 'GemmUniversal' in n:
        if g[1] in ('192', '144'): return 'fp8 K=2048 (GDN qkvz, attn qkv)'
        if g[1] == '32': return 'fp8 K=4096 (GDN out, o_proj)'
        return 'fp8 other'
    if n.startswith('marlin::Marlin'): return 'lm_head nvfp4 (Marlin)'
    if 'flashinfer' in n or 'reshape_and_cache' in n: return 'attention'
    if 'cutlass_80_wmma' in n or 'splitKreduce' in n or 'gemvx' in n: return 'bf16 GEMVs, main stream (router, GDN b/a)'
    return 'norms, top-k, glue'
mi = [i for i, r in enumerate(c) if name(r).startswith('marlin_moe_wna16') and r['GrdX'] == '144']
first = mi[0::80]
starts = []
for j in first:
    k = j
    while k > 0 and 'indexSelect' not in c[k]['Name']: k -= 1
    starts.append(k)
steps = [(starts[i], starts[i + 1]) for i in range(len(starts) - 1)]
sel = steps[lo:hi]
buckets, kern = {}, {}
walls, sums, busys, auxs = [], [], [], []
for a, b in sel:
    seg = c[a:b]; w = (c[b]['s'] - c[a]['s']) / 1e6
    iv = sorted((r['s'], r['s'] + r['d']) for r in seg)
    busy, cs, ce = 0.0, None, None
    for s, e in iv:
        if ce is None or s > ce:
            if ce is not None: busy += ce - cs
            cs, ce = s, e
        else: ce = max(ce, e)
    busy += ce - cs
    walls.append(w); busys.append(busy / 1e6); sums.append(sum(r['d'] for r in seg) / 1e6)
    main = max(set(r['Strm'] for r in seg), key=lambda s: sum(1 for r in seg if r['Strm'] == s))
    auxs.append(sum(r['d'] for r in seg if r['Strm'] != main) / 1e6)
    for r in seg:
        b_ = bucket(r, r['Strm'] != main); x = buckets.setdefault(b_, [0.0, 0, 0.0]); x[0] += r['d']; x[1] += 1
        if r['Strm'] != main: x[2] += r['d']
        k = (re.sub(r'\(.*', '', name(r))[:70], f"{r['GrdX']}x{r['GrdY']}x{r['GrdZ']}", f"{r['BlkX']}x{r['BlkY']}", 'aux' if r['Strm'] != main else 'main')
        y = kern.setdefault(k, [0, 0.0]); y[0] += 1; y[1] += r['d']
n = len(sel)
print(f'cluster {ci}: {len(steps)} steps found, using steps [{lo},{hi}) = {n}')
print(f'step wall   mean {st.mean(walls):.2f} ms  median {st.median(walls):.2f}  min {min(walls):.2f} max {max(walls):.2f}')
print(f'kernel sum  mean {st.mean(sums):.2f} ms (all streams)   union-busy mean {st.mean(busys):.2f} ms   aux-stream kernel time {st.mean(auxs):.2f} ms')
print(f'idle (wall - union busy) {st.mean(walls) - st.mean(busys):.2f} ms')
print('\n| bucket | ms/step | calls/step | of which on aux stream (ms) |\n|---|---:|---:|---:|')
for k, (t, cnt, ax) in sorted(buckets.items(), key=lambda x: -x[1][0]):
    print(f'| {k} | {t / n / 1e6:.2f} | {cnt / n:.1f} | {ax / n / 1e6:.2f} |')
print(f'| **sum** | **{sum(v[0] for v in buckets.values()) / n / 1e6:.2f}** | | |')
print('\nkernel | grid | block | stream | calls/step | us/call | ms/step')
for k, (cnt, t) in sorted(kern.items(), key=lambda x: -x[1][1])[:int(sys.argv[5]) if len(sys.argv) > 5 else 30]:
    print(f'{k[0]} | {k[1]} | {k[2]} | {k[3]} | {cnt / n:.1f} | {t / cnt / 1e3:.1f} | {t / n / 1e6:.2f}')
print('\nper-step wall ms (first 40 of selection):', [round(w, 1) for w in walls[:40]])
