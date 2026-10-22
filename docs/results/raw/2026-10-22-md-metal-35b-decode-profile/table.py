#!/usr/bin/env python3
"""md lane: final per-phase table (markdown). usage: table.py data"""
import sys
sys.argv = ['analyze.py', sys.argv[1] if len(sys.argv) > 1 else 'data']
__name__ = 'analyze'
exec(open('analyze.py').read())
ROOF = 560.0  # GB/s practical (lm_head streaming here: 586; quant_bw asymptote 564; spec 614)
_, kr, cr = report()
_, S = cb_report('cb1')
cbusy = st.median(sum(x['e'] - x['s'] for x in s) * 1e6 for s in S)
csum = sum(v[0] for v in cr.values())
c = (csum - cbusy) / 764.0
MB = 1e6
W = dict(  # weight (+KV) bytes per launch-group per token, in MB; fp8 1 B/w, nvfp4 0.5625 B/w (fp4 + 1 fp8 scale / 16), bf16 2 B/w
    qkvz=30 * (8192 * 2048 + 4096 * 2048) / MB, gdnout=30 * 2048 * 4096 / MB, qg=10 * 8192 * 2048 / MB,
    kv=10 * 2 * 512 * 2048 / MB, o=10 * 2048 * 4096 / MB,
    experts_gu=40 * 8 * 2 * 512 * 2048 * .5625 / MB, experts_dn=40 * 8 * 2048 * 512 * .5625 / MB,
    shared=40 * 3 * 512 * 2048 * .5625 / MB, lm=248320 * 2048 * .5625 / MB,
    delta=30 * 2 * 32 * 128 * 128 * 4 / MB, router=40 * 257 * 2048 * 2 / MB, pairab=30 * 2 * 32 * 2048 * 2 / MB, kvread=10 * 2 * 2 * 256 * 1000 * 2 / MB)
groups = [  # name, roles, launches, bytes MB
    ('fp8 GEMV gdn qkv 8192x2048 + z 4096x2048', ['gdn qkv fp8 8192x2048', 'gdn z fp8 4096x2048'], 60, W['qkvz']),
    ('fp8 GEMV gdn out 2048x4096', ['gdn out fp8 2048x4096'], 30, W['gdnout']),
    ('fp8 GEMV attn q+gate 8192x2048 + k, v 512x2048', ['attn q+gate fp8 8192x2048', 'attn k fp8 512x2048', 'attn v fp8 512x2048'], 30, W['qg'] + W['kv']),
    ('fp8 GEMV attn o 2048x4096', ['attn o_proj fp8 2048x4096'], 10, W['o']),
    ('MoE experts gate/up (8 of 256, nvfp4)', ['moe experts gate/up'], 40, W['experts_gu']),
    ('MoE experts down (8 of 256, nvfp4)', ['moe experts down'], 40, W['experts_dn']),
    ('shared expert (gate, up, silu_glu, down; nvfp4)', ['shared expert gate', 'shared expert up', 'shared silu_glu', 'shared expert down'], 160, W['shared']),
    ('lm_head nvfp4 248320x2048', ['lm_head nvfp4'], 1, W['lm']),
    ('MoE router+shared-gate pair (bf16 257x2048)', ['moe router+gate pair'], 40, W['router']),
    ('MoE topk', ['moe topk'], 40, 0), ('MoE combine', ['moe combine'], 40, 0),
    ('attn_decode split-KV (1k ctx)', ['attn_decode (split-KV)'], 10, W['kvread']),
    ('gdn delta_rule (f32 state in + out)', ['gdn delta_rule'], 30, W['delta']), ('gdn conv', ['gdn conv'], 30, 0),
    ('gdn a/b pair bf16', ['gdn a/b pair bf16'], 30, W['pairab']), ('gdn gated rmsnorm + silu_glu', ['gdn gated rmsnorm', 'gdn silu_glu'], 60, 0),
    ('rmsnorm / rmsnorm_add (all layers)', ['rmsnorm_add (post)', 'rmsnorm_add (gdn)', 'rmsnorm_add (attn)', 'rmsnorm (first)'], 81, 0),
    ('attn qk_norm_rope_kv + gate_split + sigmoid_mul', ['attn qk_norm_rope_kv', 'attn gate_split', 'attn sigmoid_mul'], 30, 0),
    ('embedding + argmax', ['embedding', 'argmax'], 2, 0)]
STEP = 8064.0  # us: cb1 GPU busy 7837 + idle between steps 227 (served trace); generate wall 8.05 ms
rows = []; tot_l = 0; tot_us = 0; tot_mb = 0
for name, roles, n, mb in groups:
    us = sum(cr[r][0] for r in roles) - c * n; k = sum(kr[r][0] for r in roles)
    rows.append((name, n, us, k, mb)); tot_l += n; tot_us += us; tot_mb += mb
print(f'launch correction c = {c:.2f} us (ce1 attributed sum {csum:.0f} - cb1 GPU busy {cbusy:.0f}) / 764; roofline {ROOF:.0f} GB/s')
print('| phase | launches / token | us / token | % of step (8064 us) | weight+KV MB / token | achieved GB/s | at-roofline us | excess us |')
print('|---|---:|---:|---:|---:|---:|---:|---:|')
for name, n, us, k, mb in sorted(rows, key=lambda r: -r[2]):
    gbs = f'{mb / us * 1e3:.0f}' if mb and us > 0 else '-'
    ideal = mb / ROOF * 1e3 if mb else 0
    print(f'| {name} | {n} | {us:.0f} | {us / STEP * 100:.1f} % | {mb:.1f} | {gbs} | {ideal:.0f} | {us - ideal:.0f} |')
print(f'| total | {tot_l} | {tot_us:.0f} | {tot_us / STEP * 100:.1f} % | {tot_mb:.0f} | {tot_mb / tot_us * 1e3:.0f} | {tot_mb / ROOF * 1e3:.0f} | {tot_us - tot_mb / ROOF * 1e3:.0f} |')
print(f'| GPU idle between steps (feedback off: recurrent model) | - | 227 | {227 / STEP * 100:.1f} % | - | - | - | 227 |')
print(f'| Host encode on the critical path (710-820 us per step overlapped with GPU) | - | ~0 | ~0 % | - | - | - | - |')
