#!/usr/bin/env python3
"""Cut kernel-mode / cb-mode traces of batch_prof into decode steps and tabulate where a step goes.
usage: analyze.py OUT_DIR   (reads kernel-c{ctx}-b{B}.{jsonl,marks}, cb-*, plain-*.marks, macmon.jsonl)"""
import json, sys, statistics as st, collections, os, glob, time, datetime

D = sys.argv[1]
CTXS = (1024, 4096)
HEAD_DIM, NKV = 128, 8
C_LIST = (0.0, 1.8, 3.1)   # assumed fixed cost per kernel-mode command buffer, us (see report)

import gzip
def opn(f):
    return gzip.open(f + '.gz', 'rt') if os.path.exists(f + '.gz') else open(f)

def load(f):
    return [json.loads(l) for l in opn(f)]

def classify(r):
    op = r['op']
    if op.startswith('attn_decode') or op.startswith('qk_norm_rope_kv'):
        return 'attn'
    if op.startswith('logits'):
        return 'head'
    if op.startswith('matmul_pair'):
        return 'gate+up' if r['n'] > 10000 else 'k+v'
    if op.startswith('matmul'):
        return 'down' if r['k'] > 8000 else 'q/o'
    return 'norm/elementwise'

def steps_of(kind, ctx, b):
    marks = [m for m in load(f'{D}/{kind}-c{ctx}-b{b}.marks') if not m['warm']]
    recs = load(f'{D}/{kind}-c{ctx}-b{b}.jsonl')
    out = []
    for m in marks:
        rs = [r for r in recs if r['s'] >= m['s'] and r['e'] <= m['e']]
        out.append((m, rs))
    return out

def kernel_table(ctx, b, c):
    per = collections.defaultdict(list)  # class -> per-step totals (us)
    cnt = {}
    attn_parts = collections.defaultdict(list)
    nd, tot = [], []
    for m, rs in steps_of('kernel', ctx, b):
        agg = collections.defaultdict(float)
        for r in rs:
            d = (r['e'] - r['s']) * 1e6 - c
            agg[classify(r)] += d
            if classify(r) == 'attn':
                attn_parts[r['op']].append(d)
        for k, v in agg.items():
            per[k].append(v)
        cnt = collections.Counter(classify(r) for r in rs)
        nd.append(len(rs)); tot.append(sum(agg.values()))
    return {k: st.median(v) for k, v in per.items()}, st.median(tot), st.median(nd), cnt, attn_parts

def cb_step(ctx, b):
    res = []
    for m, rs in steps_of('cb', ctx, b):
        busy = sum(r['e'] - r['s'] for r in rs) * 1e3
        span = (max(r['e'] for r in rs) - min(r['s'] for r in rs)) * 1e3
        res.append((m['wall_ms'], span, busy, sum(r['nd'] for r in rs), len(rs),
                    (min(r['s'] for r in rs) - m['s']) * 1e3, (m['e'] - max(r['e'] for r in rs)) * 1e3))
    return [st.median(x[i] for x in res) for i in range(7)]

def plain(ctx, b):
    w = []
    for f in sorted(glob.glob(f'{D}/plain-c{ctx}-b{b}-r*.marks')):
        w.append(st.median(m['wall_ms'] for m in load(f) if not m['warm']))
    return w

def clock(kind_files):
    off = time.time() - time.clock_gettime(time.CLOCK_UPTIME_RAW)
    mm = []
    for l in opn(f'{D}/macmon.jsonl'):
        x = json.loads(l)
        mm.append((datetime.datetime.fromisoformat(x['timestamp']).timestamp(), x['gpu_usage'][0], x['gpu_power']))
    res = {}
    for f in kind_files:
        ms = [m for m in load(f) if not m['warm']]
        t0, t1 = ms[0]['s'] + off, ms[-1]['e'] + off
        w = [x for x in mm if t0 <= x[0] <= t1]
        res[os.path.basename(f)] = (len(w), st.mean(x[1] for x in w) if w else float('nan'), st.mean(x[2] for x in w) if w else float('nan'))
    return res

P = print
P('# analysis output (analyze.py)\n')
for ctx in CTXS:
    P(f'## ctx {ctx}\n')
    for b in (1, 8):
        P(f'### B={b}: plain step wall medians per run (ms): {[round(x, 2) for x in plain(ctx, b)]}')
        w = cb_step(ctx, b)
        P(f'cb mode: wall {w[0]:.2f} ms, GPU span {w[1]:.2f}, GPU busy {w[2]:.2f}, dispatches {w[3]:.0f}, command buffers {w[4]:.0f}, lead {w[5]:.2f} ms, tail {w[6]:.2f} ms')
        for c in C_LIST:
            tab, tot, nd, cnt, ap = kernel_table(ctx, b, c)
            P(f'\nkernel mode, c={c} us per launch removed: sum {tot / 1e3:.2f} ms over {nd:.0f} launches')
            P('| class | launches | GPU ms | share of kernel sum |'); P('|---|---:|---:|---:|')
            for k, v in sorted(tab.items(), key=lambda x: -x[1]):
                P(f'| {k} | {cnt[k]} | {v / 1e3:.2f} | {100 * v / tot:.1f}% |')
            if c == C_LIST[1]:
                for op, v in ap.items():
                    P(f'  attn kernel {op}: mean {st.mean(v):.2f} us  min {min(v):.2f}  n/step {len(v) / len(steps_of("kernel", ctx, b)):.0f}')
        P('')
