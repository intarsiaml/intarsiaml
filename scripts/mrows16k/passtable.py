#!/usr/bin/env python3
"""mrows16k: split an INTARSIA_METAL_PROFILE trace (kernel: or ce1 = cb + COMMIT_EVERY=1) into verify passes (embedding .. argmax_f32), classify
each launch by op class, and print the median per-pass time per class for each pass row count. usage: passtable.py <trace.jsonl.gz>..."""
import sys, json, gzip, statistics as st, collections, os
EFF = os.environ.get('EFF') == '1'  # ce1 traces: attribute only the non-overlapped part of each launch, so the classes sum to the pass span

def cls(r):
    op = r['op'].split('#')[0]
    n, k = r['n'], r['k']
    if op.startswith('attn_prefill'): return 'attention kernel (rows splitkv)'
    if op in ('qk_norm_rope_kv_bf16', 'gate_split_bf16', 'sigmoid_mul_bf16'): return 'attention glue (qk-norm/rope/kv write, gate split, sigmoid mul)'
    if op in ('gdn_conv_bf16', 'gdn_delta_rule_bf16'): return 'GDN recurrent (conv + delta rule, 1 row each)'
    if op == 'other': return 'other / unlabelled (t=0; between GDN rows: state copies)'
    if op in ('rmsnorm_bf16', 'silu_glu_bf16') and r['t'] >= 32: return 'GDN gated norm + silu (all rows at once)'
    if op.startswith('moe_experts'): return 'MoE routed experts (gate/up, down)'
    if op in ('moe_topk_bf16', 'moe_combine_bf16'): return 'MoE topk + combine'
    if op.startswith('matmul_nvfp4') and n == 248320: return 'lm_head (nvfp4 248320x2048)'
    if op.startswith('matmul_nvfp4'): return 'shared expert (nvfp4 gate/up/down)'
    if op.startswith('matmul_fp8'): return 'dense fp8 GEMM (GDN qkv/z/out, attn q/k/v/o)'
    if op.startswith('matmul_pair'): return 'small pair GEMM (GDN a/b 64, router + shared gate 257)'
    if op in ('rmsnorm_add_bf16', 'rmsnorm_bf16', 'embedding_bf16', 'silu_glu_bf16', 'argmax_f32'): return 'norms / embedding / argmax / silu_glu'
    return 'UNCLASSIFIED ' + op

def passes(R):
    cur = None
    for r in R:
        op = r['op'].split('#')[0]
        if op == 'embedding_bf16' and r['t'] > 1: cur = {'rows': r['t'], 'recs': []}
        if cur is not None: cur['recs'].append(r)
        if op == 'argmax_f32' and cur is not None:
            yield cur; cur = None

for path in sys.argv[1:]:
    R = [json.loads(l) for l in gzip.open(path, 'rt')]
    by = collections.defaultdict(list)
    for p in passes(R): by[p['rows']].append(p)
    print(f'## {path.split("/")[-1]}: passes by rows', {k: len(v) for k, v in sorted(by.items())})
    for rows, ps in sorted(by.items()):
        if len(ps) < 3: continue
        per = collections.defaultdict(list); tot = []; span = []; ln = []
        for p in ps:
            d = collections.defaultdict(float); n = collections.Counter()
            prev_end = p['recs'][0]['s']
            for r in p['recs']:
                c = cls(r); n[c] += 1
                if EFF:  # overlap-free attribution: a launch gets the time from the later of its start and the previous end
                    d[c] += max(0.0, r['e'] - max(r['s'], prev_end)) * 1e3; prev_end = max(prev_end, r['e'])
                else:
                    d[c] += (r['e'] - r['s']) * 1e3
            for c in set(d) | set(per): per[c].append((d.get(c, 0.0), n.get(c, 0)))
            tot.append(sum(d.values())); ln.append(len(p['recs']))
            span.append((p['recs'][-1]['e'] - p['recs'][0]['s']) * 1e3)
        print(f'### {rows}-row passes (n={len(ps)}): sum of launch GPU ms median {st.median(tot):.2f}, first-start..last-end span {st.median(span):.2f} ms, launches {st.median(ln):.0f}')
        for c, v in sorted(per.items(), key=lambda kv: -st.median(x[0] for x in kv[1])):
            print(f'  {st.median(x[0] for x in v):8.3f} ms  {st.median(x[1] for x in v):6.0f} launches  {c}')
