#!/usr/bin/env python3
"""md lane analysis: decode step breakdown from kernel / ce1 / cb traces. usage: analyze.py data/"""
import gzip, json, sys, statistics as st, collections
D = sys.argv[1] if len(sys.argv) > 1 else 'data'
def load(n): return [json.loads(l) for l in gzip.open(f'{D}/{n}.jsonl.gz', 'rt')]
def steps_of(r, nsteps=128):
    """split records into decode steps at argmax_f32; return the last nsteps steps"""
    idx = [i for i, x in enumerate(r) if x['op'] == 'argmax_f32']
    out = []
    for a, b in zip(idx[-nsteps - 1:-1], idx[-nsteps:]):
        out.append(r[a + 1:b + 1])
    return out
def dur_ce1(step, prev_end):
    d = []
    for x in step:
        s = max(x['s'], prev_end); d.append(max(x['e'] - s, 0)); prev_end = x['e']
    return d, prev_end

# ---- roles (offset within a 19-record layer; records 0-1 and 762-763 are outside layers)
def role(pos, ops):
    if pos < 2: return {0: 'embedding', 1: 'rmsnorm (first)'}[pos]
    if pos >= 762: return {762: 'lm_head nvfp4', 763: 'argmax'}[pos]
    L, o = divmod(pos - 2, 19)
    attn = (L % 4 == 3)
    if o >= 9:
        m = {9: 'moe router+gate pair', 10: 'moe topk', 11: 'moe experts gate/up', 12: 'moe experts down',
             13: 'shared expert gate', 14: 'shared expert up', 15: 'shared silu_glu', 16: 'shared expert down',
             17: 'moe combine', 18: 'rmsnorm_add (post)'}
        return m[o]
    if attn:
        return {0: 'attn q+gate fp8 8192x2048', 1: 'attn gate_split', 2: 'attn k fp8 512x2048', 3: 'attn v fp8 512x2048',
                4: 'attn qk_norm_rope_kv', 5: 'attn_decode (split-KV)', 6: 'attn sigmoid_mul', 7: 'attn o_proj fp8 2048x4096',
                8: 'rmsnorm_add (attn)'}[o]
    return {0: 'gdn qkv fp8 8192x2048', 1: 'gdn z fp8 4096x2048', 2: 'gdn a/b pair bf16', 3: 'gdn conv', 4: 'gdn delta_rule',
            5: 'gdn gated rmsnorm', 6: 'gdn silu_glu', 7: 'gdn out fp8 2048x4096', 8: 'rmsnorm_add (gdn)'}[o]

ROLE_ORDER = None
def per_role(durs_by_step, ops):
    """durs_by_step: list of per-step lists (seconds). returns {role: median over steps of summed us}"""
    tot = collections.defaultdict(lambda: [0.0] * len(durs_by_step)); cnt = collections.Counter()
    for si, d in enumerate(durs_by_step):
        for p, v in enumerate(d): tot[role(p, ops)][si] += v * 1e6
    for p in range(len(ops)): cnt[role(p, ops)] += 1
    return {k: (st.median(v), cnt[k]) for k, v in tot.items()}

def report():
    out = []
    # kernel mode
    r = load('kernel1'); S = steps_of(r); ops = [x['op'] for x in S[0]]
    kd = [[x['e'] - x['s'] for x in s] for s in S]
    kr = per_role(kd, ops); ksum = st.median(sum(d) for d in kd) * 1e6
    # ce1
    r = load('ce1'); S2 = steps_of(r); pe = S2[0][0]['s']; cd = []
    for s in S2:
        d, pe = dur_ce1(s, pe); cd.append(d)
    cr = per_role(cd, ops); csum = st.median(sum(d) for d in cd) * 1e6
    span = [(s[-1]['e'] - s[0]['s']) * 1e6 for s in S2]
    out.append(f'kernel-mode step sum (serialised, incl. per-cb cost): median {ksum:.0f} us')
    out.append(f'ce1 step sum of attributed durations: median {csum:.0f} us; first-start..last-end span median {st.median(span):.0f} us')
    gaps = []
    for a, b in zip(S2, S2[1:]): gaps.append((b[0]['s'] - a[-1]['e']) * 1e6)
    out.append(f'ce1 inter-step GPU idle (argmax end -> next embedding start): median {st.median(gaps):.0f} us')
    out.append('')
    out.append('role | launches/token | kernel-mode us/token | ce1 us/token')
    order = sorted(kr, key=lambda k: -kr[k][0])
    for k in order:
        out.append(f'{k} | {kr[k][1]} | {kr[k][0]:.0f} | {cr[k][0]:.0f}')
    return out, kr, cr

def cb_steps(r, nsteps=128, nd_step=764):
    """group cb records into decode steps by walking back from the end, nd_step dispatches per step"""
    steps = []; cur = []; acc = 0
    for x in reversed(r):
        cur.append(x); acc += x['nd']
        if acc == nd_step: steps.append(list(reversed(cur))); cur = []; acc = 0
        elif acc > nd_step: raise SystemExit(f'step boundary mismatch at record {x["i"]} acc={acc}')
        if len(steps) == nsteps: break
    return list(reversed(steps))

def cb_report(name):
    r = load(name); S = cb_steps(r); out = []
    nrec = collections.Counter(len(s) for s in S)
    busy = []; span = []; enc = []; commit = []; idle_in = []; lead = []; wall = []; prev_hr = None
    firstgap = []
    for si, s in enumerate(S):
        busy.append(sum(x['e'] - x['s'] for x in s) * 1e6)
        span.append((s[-1]['e'] - s[0]['s']) * 1e6)
        idle_in.append(sum(max(b['s'] - a['e'], 0) for a, b in zip(s, s[1:])) * 1e6)
        enc.append(sum(x['hc'] - x['ho'] for x in s) * 1e6)
        commit.append(sum(x['hr'] - x['hc'] for x in s) * 1e6)
        wall.append((s[-1]['e'] - s[0]['ho']) * 1e6)
        lead.append((s[0]['s'] - s[0]['ho']) * 1e6)
    gaps = [(b[0]['s'] - a[-1]['e']) * 1e6 for a, b in zip(S, S[1:])]
    m = lambda v: st.median(v)
    out.append(f'{name}: {len(S)} decode steps, cb records/step {dict(nrec)}')
    out.append(f'  GPU busy (sum e-s) {m(busy):.0f} us; first-start..last-end {m(span):.0f} us; GPU idle between cbs inside step {m(idle_in):.0f} us')
    out.append(f'  host encode (open->commit entered, summed over cbs) {m(enc):.0f} us; commit call {m(commit):.0f} us')
    out.append(f'  first encode -> last GPU end {m(wall):.0f} us; first-cb encode start -> its GPU start {m(lead):.0f} us')
    out.append(f'  GPU idle step-to-step (last end -> next first start) {m(gaps):.0f} us (min {min(gaps):.0f}, max {max(gaps):.0f})')
    return out, S

if __name__ == '__main__':
    o, kr, cr = report(); print('\n'.join(o))
    for n in sys.argv[2:]: print('\n'.join(cb_report(n)[0]))
