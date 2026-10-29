#!/usr/bin/env python3
"""Lane moe16b: per-bucket kernel ms per step from two prof.sh shapes.txt files (run in the raw dir): buckets.py base-b16 dev-b16"""
import re,sys
def load(a):
    rows=[]
    for l in open(f'{a}/shapes.txt'):
        m=re.search(r'grid (\S+)\s+blk (\S+)\s+([\d.]+) calls/step\s+([\d.]+) us\s+([\d.]+) ms/step',l)
        if not m: continue
        rows.append((l.split()[0].replace('<unnamed>::',''),m.group(1),float(m.group(3)),float(m.group(4)),float(m.group(5))))
    return rows
def bucket(n,grid):
    if 'moe_dec_gate_up' in n or 'moe_dec_down' in n: return 'routed MoE (gate_up + down_v2)'
    if 'gdn_delta' in n or 'gdn_conv' in n: return 'GDN (delta, conv)'
    if 'fp8_gemv_rows_q' in n and grid.startswith('96x'): return 'fp8 K=2048 (GDN qkv / z, attn q)'
    if 'fp8_gemv_rows_q' in n and grid.startswith('48x'): return 'fp8 K=4096 (GDN out, o_proj)'
    if 'fp8_gemv_rows_q' in n and grid.startswith('32x'): return 'fp8 k / v 512x2048'
    if 'nvfp4_gemv_full' in n: return 'lm_head nvfp4'
    if 'nvfp4_gemv_split' in n or 'nvfp4_gemv_tile' in n: return 'shared expert nvfp4 (gate, up, down)'
    if 'gemv_split' in n: return 'bf16 small GEMVs (router, GDN b / a, shared gate)'
    if 'splitkv' in n or 'attn' in n or 'kv_append' in n: return 'attention'
    return 'norms, top-k, glue'
out={a:{} for a in sys.argv[1:]}
for a in out:
    for name,grid,calls,us,ms in load(a):
        b=bucket(name,grid); out[a][b]=out[a].get(b,0)+ms
A,B=sys.argv[1],sys.argv[2]
keys=sorted(set(out[A])|set(out[B]),key=lambda k:-out[A].get(k,0))
print('| bucket (ms per step, B=16) | base | dev | dev - base |\n|---|---:|---:|---:|')
for k in keys:
    x,y=out[A].get(k,0),out[B].get(k,0); print(f'| {k} | {x:.2f} | {y:.2f} | {y-x:+.2f} |')
x,y=sum(out[A].values()),sum(out[B].values()); print(f'| **sum of kernel durations** | **{x:.2f}** | **{y:.2f}** | **{y-x:+.2f}** |')
