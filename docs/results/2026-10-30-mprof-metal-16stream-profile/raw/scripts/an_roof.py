#!/usr/bin/env python3
"""an_roof.py : per-kernel roofline table of the 35B batched decode step. Time = ce1 exclusive ms / step (mean of the two real-traffic
runs w1 asc / w1b desc); bytes and FLOPs are modelled from the launch labels (weight shape), the measured distinct experts
(kernel-mode labels of the same run), the context lengths of the harness rows and the model config. Prints markdown.
usage: an_roof.py B [B...]"""
import sys, collections, statistics
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import mp

RUNS = [('w1/ce1.jsonl.gz', 'w1/kernel.jsonl.gz', [1, 4, 8, 9, 12, 16]), ('w1b/ce1.jsonl.gz', 'w1b/kernel1.jsonl.gz', [16, 12, 9, 8, 4, 1])]
ROOF_BW = 580.0      # GB/s attainable read bandwidth measured here (scripts/roof, roof.txt: 580 max / 566 median; nominal 614)
STATE_B = 32 * 128 * 128 * 4   # GDN recurrent state per layer per row, f32
KV_B_PER_TOK = 2 * 2 * 256 * 2  # K+V, 2 kv heads, head_dim 256, bf16 (per layer)
E_GU, E_DN = 1.18e6, 0.59e6    # bytes per expert: gate-up 1024x2048 fp4 (+scales), down 2048x512 fp4 (+scales)

def ctx_rows(order, b, phase_steps=36):
    """context length of each of the b rows (mean over the measured steps of phase b) in a run with the given phase order."""
    out = []
    for i in range(b):
        before = sum(phase_steps for p in order[:order.index(b)] if i < p)
        out.append(1000 + 7 * i + before + (phase_steps + mp.WARM) / 2)
    return out

def experts(kf, order):
    D = mp.by_b(kf, 'kernel', order); r = {}
    for b in order:
        ms = [x for s in D[b] for x in s if x['op'].startswith('moe_experts') and '#' not in x['op']]
        r[b] = (statistics.mean(x['n'] for x in ms), statistics.mean(x['k'] for x in ms))
    return r

def model(b, sub, cnt, rec, dist, pairs, ctx, order):
    """-> (stream bytes per step, unique bytes per step, flops per step) for one (group, sub) at B=b.
    rec = a representative record (label), cnt = launches per step."""
    ch = 2 if b > 8 else 1
    w = rec.get('bytes', 0)
    if sub.startswith('fp8'):
        n, k = rec['n'], rec['k']
        M = b / ch   # rows per launch (the two chunks are ceil / floor of b/2)
        return w * cnt, w * cnt / ch, 2 * n * k * b * cnt / ch
    if sub.startswith('nvfp4') or sub.startswith('lm_head'):
        n, kp = rec['n'], rec['k']
        wb = n * kp * 1.125
        return wb * cnt, wb * cnt / ch, 2 * n * (2 * kp) * b * cnt / ch
    if sub.startswith('router') or sub.startswith('in_a'):
        n, k = rec['n'], rec['k']
        return w * cnt, w * cnt / ch, 2 * n * k * b * cnt / ch
    if sub.startswith('experts gate-up'):
        return dist * E_GU * 40, dist * E_GU * 40, pairs * 4.19e6 * 40
    if sub.startswith('experts down'):
        return dist * E_DN * 40, dist * E_DN * 40, pairs * 2.10e6 * 40
    if sub.startswith('delta rule'):
        # read + write the state in place; q/k/v/g/beta traffic is negligible
        return 2 * b * 30 * STATE_B, 2 * b * 30 * STATE_B, b * 30 * 4.2e6
    if sub.startswith('attention decode'):
        return sum(ctx) * KV_B_PER_TOK * 10, sum(ctx) * KV_B_PER_TOK * 10, sum(ctx) * 16384 * 10
    return 0.0, 0.0, 0.0

def main():
    Bs = [int(x) for x in sys.argv[1:]] or [8, 16]
    acc = {b: collections.defaultdict(lambda: collections.defaultdict(list)) for b in Bs}
    for cf, kf, order in RUNS:
        D = mp.by_b(cf, 'ce1', order); E = experts(kf, order)
        for b in Bs:
            n = len(D[b]); ctx = ctx_rows(order, b)
            agg = collections.defaultdict(lambda: [0.0, 0.0, None])
            for s in D[b]:
                for r in s:
                    g, sub = mp.kind(r, b); a = agg[(g, sub)]
                    a[0] += r['dur'] * 1e3 / n; a[1] += 1 / n
                    if r['bytes'] and a[2] is None: a[2] = r
            for key, (ms, cnt, rec) in agg.items():
                sb, ub, fl = model(b, key[1], cnt, rec or {}, E[b][0], E[b][1], ctx, order) if (rec or key[1].startswith(('experts', 'delta', 'attention'))) else (0, 0, 0)
                acc[b][key]['ms'].append(ms); acc[b][key]['cnt'].append(cnt)
                acc[b][key]['sb'].append(sb); acc[b][key]['ub'].append(ub); acc[b][key]['fl'].append(fl)
    for b in Bs:
        print(f'\n### B={b}\n')
        print('| kernel (ce1, in-stream) | ms / step | launches | us / launch | stream MB / step | unique MB / step | GB/s (stream) | GB/s (unique) | % of 580 (unique) | TFLOPS |')
        print('|---|---|---|---|---|---|---|---|---|---|')
        tot = sum(statistics.mean(v['ms']) for v in acc[b].values())
        for key, v in sorted(acc[b].items(), key=lambda kv: -statistics.mean(kv[1]['ms'])):
            ms = statistics.mean(v['ms']); cnt = statistics.mean(v['cnt']); sb = statistics.mean(v['sb']); ub = statistics.mean(v['ub']); fl = statistics.mean(v['fl'])
            if ms < 0.04: continue
            gs = sb / ms / 1e6 if sb else 0; gu = ub / ms / 1e6 if ub else 0
            print(f'| {key[1]} | {ms:.3f} | {cnt:.0f} | {ms / cnt * 1e3:.1f} | {sb / 1e6:.0f} | {ub / 1e6:.0f} | {gs:.0f} | {gu:.0f} | {gu / ROOF_BW * 100:.0f}% | {fl / ms / 1e9:.2f} |' if sb else f'| {key[1]} | {ms:.3f} | {cnt:.0f} | {ms / cnt * 1e3:.1f} | - | - | - | - | - | - |')
        print(f'\ntotal {tot:.2f} ms / step')
main()
