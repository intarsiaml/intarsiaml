#!/usr/bin/env python3
"""INTARSIA_METAL_COMMIT_EVERY=1 cb-mode trace (one command buffer per dispatch, committed back to back, not waited
on): cut into decode steps and attribute the GPU timeline to launch classes.
For record i: own = e_i - s_i; serial = e_i - max(s_i, e_(i-1)) (what it adds to the timeline, overlap with the previous
buffer removed); gap = max(0, s_i - e_(i-1)) (GPU idle before it).  sum(serial)+sum(gap) = step GPU span.
usage: analyze_ce1.py DIR"""
import json, sys, statistics as st, collections, os, gzip

def opn(f):
    return gzip.open(f + '.gz', 'rt') if os.path.exists(f + '.gz') else open(f)

D = sys.argv[1]

def load(f):
    return [json.loads(l) for l in opn(f)]

def classify(r):
    op = r['op']
    if op.startswith('attn_decode'): return 'attn_decode (per row)'
    if op.startswith('qk_norm_rope_kv'): return 'qk_norm_rope_kv (per row)'
    if op.startswith('logits'): return 'head (lm_head GEMV)'
    if op.startswith('matmul_pair'): return 'gate+up' if r['n'] > 10000 else 'k+v'
    if op.startswith('matmul'): return 'down' if r['k'] > 8000 else 'q/o'
    return 'norm/elementwise'

for ctx in (1024, 4096):
    for b in (8, 1):
        f = f'{D}/ce1-c{ctx}-b{b}'
        marks = [m for m in load(f + '.marks') if not m['warm']]
        recs = load(f + '.jsonl')
        per = collections.defaultdict(list); own = collections.defaultdict(list); gaps = collections.defaultdict(list)
        spans, walls, nds = [], [], []
        for m in marks:
            rs = [r for r in recs if r['s'] >= m['s'] and r['e'] <= m['e']]
            rs.sort(key=lambda r: r['s'])
            ser = collections.defaultdict(float); ow = collections.defaultdict(float); gp = collections.defaultdict(float)
            prev = None
            for r in rs:
                c = classify(r)
                ow[c] += (r['e'] - r['s']) * 1e6
                if prev is not None:
                    ser[c] += (r['e'] - max(r['s'], prev)) * 1e6
                    gp[c] += max(0.0, r['s'] - prev) * 1e6
                else:
                    ser[c] += (r['e'] - r['s']) * 1e6
                prev = r['e']
            for k in set(ser) | set(gp):
                per[k].append(ser[k]); own[k].append(ow[k]); gaps[k].append(gp[k])
            spans.append((rs[-1]['e'] - rs[0]['s']) * 1e3); walls.append(m['wall_ms']); nds.append(len(rs))
        span = st.median(spans)
        print(f'\n### ctx {ctx} B={b}: wall median {st.median(walls):.2f} ms, GPU span {span:.2f} ms, {st.median(nds):.0f} launches/step, {len(marks)} steps')
        print('| class | launches | serial GPU ms | gap-before ms | own (kernel+cb) ms | share of span (serial+gap) |')
        print('|---|---:|---:|---:|---:|---:|')
        cnt = collections.Counter(classify(r) for r in recs[:0])
        n_per = {}
        m0 = marks[len(marks) // 2]
        rs0 = [r for r in recs if r['s'] >= m0['s'] and r['e'] <= m0['e']]
        n_per = collections.Counter(classify(r) for r in rs0)
        tot = 0
        for k in sorted(per, key=lambda k: -(st.median(per[k]) + st.median(gaps[k]))):
            s_, g_, o_ = st.median(per[k]) / 1e3, st.median(gaps[k]) / 1e3, st.median(own[k]) / 1e3
            print(f'| {k} | {n_per[k]} | {s_:.2f} | {g_:.2f} | {o_:.2f} | {100 * (s_ + g_) / span:.1f}% |')
