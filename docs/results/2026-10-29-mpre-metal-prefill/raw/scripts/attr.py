#!/usr/bin/env python3
"""MPRE attribution: per-class GPU time from kernel-mode traces, idle gaps from cb-mode traces.
usage: attr.py DIR  (reads m{35,8}-{kernel,cb}-{1k,4k,16k}.jsonl.gz and m*-plain-*.err)"""
import gzip, json, re, sys, collections, os

D = sys.argv[1]

def cls(op):
    o = op.split('#')[0]
    if o.startswith('moe_experts'): return 'moe experts'
    if o.startswith('moe_'): return 'moe route/combine'
    if o.startswith('gdn_delta'): return 'gdn delta rule'
    if o.startswith('gdn_conv'): return 'gdn conv'
    if o.startswith('attn_'): return 'attention'
    if o.startswith('logits') or o.startswith('argmax'): return 'lm_head/argmax'
    if o.startswith('matmul_fp8'): return 'dense gemm fp8'
    if o.startswith('matmul_nvfp4'): return 'dense gemm nvfp4'
    if o.startswith('matmul') or o.startswith('gemm'): return 'dense gemm bf16'
    if o.startswith('fp8_') or 'quant' in o: return 'quant'
    return 'norm/elementwise'

def load(p):
    if not os.path.exists(p):
        q = p[:-3]
        return [json.loads(l) for l in open(q)] if os.path.exists(q) else []
    return [json.loads(l) for l in gzip.open(p, 'rt')]

def plain(m, c):
    out = []
    for ab in 'ab':
        try: t = open(f'{D}/m{m}-plain-{c}-{ab}.err').read()
        except OSError: continue
        x = re.findall(r'prefill (\d+) ms \(([\d.]+) tok/s\)', t)
        n = re.findall(r'prompt (\d+) tokens', t)
        if x: out.append((int(x[-1][0]), float(x[-1][1]), n[-1] if n else '?'))
    return out

for m in ('35', '8'):
    for c in ('1k', '4k', '16k'):
        k = load(f'{D}/m{m}-kernel-{c}.jsonl.gz')
        cb = load(f'{D}/m{m}-cb-{c}.jsonl.gz')
        pl = plain(m, c)
        print(f'\n=== {m}B {c}: plain prefill {pl}')
        if not k: continue
        # prefill records: everything before the first decode-shaped (t == 1) attention / logits after prefill;
        # keep it simple: drop records with t == 1 that come after the last record with t > 1
        last = max(i for i, r in enumerate(k) if r['t'] > 1)
        pre = k[:last + 2]  # include the prefill's logits row
        tot = sum(r['e'] - r['s'] for r in pre)
        by = collections.defaultdict(lambda: [0.0, 0, 0.0, 0])
        byop = collections.defaultdict(lambda: [0.0, 0, 0.0])
        for r in pre:
            d = r['e'] - r['s']
            c_ = cls(r['op'])
            fl = 2.0 * r['t'] * r['n'] * r['k'] if r['n'] else 0.0
            by[c_][0] += d; by[c_][1] += 1; by[c_][2] += fl; by[c_][3] += r['bytes']
            o = r['op'].split('#')[0]
            byop[o][0] += d; byop[o][1] += 1; byop[o][2] += fl
        print(f'kernel-mode GPU sum {tot*1e3:.0f} ms over {len(pre)} records')
        for c_, (d, n, fl, b) in sorted(by.items(), key=lambda x: -x[1][0]):
            tf = f'{fl/d/1e12:6.1f} TF' if fl else '        '
            gb = f'{b/d/1e9:6.0f} GB/s(w)' if b else ''
            print(f'  {c_:20s} {d*1e3:8.1f} ms {100*d/tot:5.1f}%  n={n:5d} {tf} {gb}')
        print('  -- top ops')
        for o, (d, n, fl) in sorted(byop.items(), key=lambda x: -x[1][0])[:14]:
            tf = f'{fl/d/1e12:6.1f} TF' if fl else ''
            print(f'     {o:34s} {d*1e3:8.1f} ms n={n:5d} {d/n*1e6:8.0f} us {tf}')
        if cb:
            # cb mode: prefill = buffers until the gap before decode; span vs busy
            lastc = max(i for i, r in enumerate(cb) if r['nd'] > 50) if any(r['nd'] > 50 for r in cb) else len(cb) - 1
            pc = cb[:lastc + 1]
            span = pc[-1]['e'] - pc[0]['s']
            busy = sum(r['e'] - r['s'] for r in pc)
            host = pc[-1]['e'] - pc[0]['ho']
            print(f'cb-mode: {len(pc)} buffers, first-start..last-end {span*1e3:.0f} ms, GPU busy {busy*1e3:.0f} ms, '
                  f'idle {(span-busy)*1e3:.0f} ms ({100*(span-busy)/span:.1f}%), host-open..end {host*1e3:.0f} ms')
