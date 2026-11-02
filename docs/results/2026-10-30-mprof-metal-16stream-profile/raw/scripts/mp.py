#!/usr/bin/env python3
"""MPROF trace helpers: load a real-traffic trace (w1), split it into the per-B phases and per-step record lists."""
import gzip, json, collections
BS = [1, 4, 8, 9, 12, 16]
WARM = 4

def load(f):
    op = gzip.open if f.endswith('.gz') else open
    R = [json.loads(l) for l in op(f, 'rt')]
    R.sort(key=lambda r: r['s'])
    return R

def phases(R, gap=1.0):
    ph = [[R[0]]]
    for a, b in zip(R, R[1:]):
        if b['s'] - a['e'] > gap:
            ph.append([])
        ph[-1].append(b)
    return ph

def exclusive(P):
    """ce1: charge each record only the time after the previous one ended (consecutive buffers can overlap)."""
    pe = P[0]['s']
    for r in P:
        r['dur'] = max(0.0, r['e'] - max(r['s'], pe))
        pe = max(pe, r['e'])
    return P

def steps(P, first='embedding_bf16'):
    S = []
    for r in P:
        if r['op'] == first:
            S.append([])
        if S:
            S[-1].append(r)
    return S

def by_b(f, mode, order=None):
    """-> {B: [step record lists] without warm-up}; mode ce1 uses exclusive durations, kernel uses e-s."""
    R = load(f)
    ph = phases(R)
    order = order or BS
    assert len(ph) == 1 + len(order), [len(p) for p in ph]
    out = {}
    for b, P in zip(order, ph[1:]):
        if mode == 'ce1':
            exclusive(P)
        else:
            for r in P:
                r['dur'] = r['e'] - r['s']
        out[b] = steps(P)[WARM:]
    return out

# ---- grouping -------------------------------------------------------------------------------------------------------------
def kind(r, b):
    """-> (group, sub) of one trace record of a B-row decode step (label op, t = rows of the first input, n / k = weight shape
    from the label; nvfp4 weights are stored packed, so their k is half the true K)."""
    op = r['op'].split('#')[0]
    sfx = r['op'][len(op):]
    n, k, t = r['n'], r['k'], r['t']
    if op.startswith('moe_experts'):
        return ('MoE routed experts', {'': 'experts gate-up (fused silu*up)', '#1': 'experts down', '#2': 'expert gather / grouping'}.get(sfx, 'other' + sfx))
    if op.startswith('moe_topk'):
        return ('router + top-k', 'top-k')
    if op.startswith('matmul_pair_bf16') and n == 257:
        return ('router + top-k', 'router + shared-gate GEMV (257x2048)')
    if op.startswith('moe_combine'):
        return ('MoE routed experts', 'combine (weighted sum + shared)')
    if op.startswith('matmul_nvfp4') and n == 248320:
        return ('lm_head + argmax', 'lm_head nvfp4 248320x2048 -> f32')
    if op.startswith('argmax'):
        return ('lm_head + argmax', 'argmax')
    if op.startswith('matmul_nvfp4'):
        return ('shared expert', 'nvfp4 gate / up 512x2048' if n == 512 else 'nvfp4 down 2048x512')
    if op.startswith('silu_glu'):
        return ('shared expert', 'silu_glu') if t == b else ('GDN layers', 'gated norm: silu_glu')
    if op.startswith('rmsnorm_add'):
        return ('norms / embedding', 'rmsnorm_add (2 per layer)')
    if op.startswith('rmsnorm'):
        return ('norms / embedding', 'final rmsnorm') if t == b else ('GDN layers', 'gated norm: rmsnorm')
    if op.startswith('embedding'):
        return ('norms / embedding', 'embedding')
    if op.startswith('matmul_fp8'):
        sub = {(8192, 2048): 'fp8 8192x2048 (GDN qkv x30, attn q+gate x10)', (4096, 2048): 'fp8 4096x2048 (GDN z x30)',
               (2048, 4096): 'fp8 2048x4096 (GDN out x30, attn o x10)', (512, 2048): 'fp8 512x2048 (attn k, v x10 each)'}.get((n, k), f'fp8 {n}x{k}')
        return ('dense fp8 projections', sub)
    if op.startswith('matmul_pair_bf16') and n == 64:
        return ('GDN layers', 'in_a + in_b bf16 pair GEMV (64x2048)')
    if op.startswith('gdn_conv'):
        return ('GDN layers', 'conv1d + state')
    if op.startswith('gdn_delta'):
        return ('GDN layers', 'delta rule (recurrent state)')
    if op.startswith('attn_decode'):
        return ('full attention (core)', 'attention decode split-KV')
    if op.startswith('qk_norm_rope_kv'):
        return ('full attention (core)', 'qk_norm_rope_kv')
    if op.startswith('gate_split') or op.startswith('sigmoid_mul'):
        return ('full attention (core)', op.split('_bf16')[0])
    return ('other', r['op'])
