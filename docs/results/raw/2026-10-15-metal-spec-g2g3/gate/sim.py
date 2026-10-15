#!/usr/bin/env python3
"""Offline gate simulator. Greedy streams are identical with spec on/off, so the accepted count of every pass is
known exactly from the plain (spec off) stream: replay the engine loop (drafter + gate) over [prompt ids + stream].
Cost model: plain step 1, verify pass with m drafts 1 + C*m (C fitted on the G1 open-ended trace: 106 passes of 4 drafts
~ 0.93-0.95x)."""
import json, re, glob, sys, os
from tokenizers import Tokenizer
T = Tokenizer.from_file('/Users/josef/models/Qwen3-8B/tokenizer.json')
HERE = os.path.dirname(os.path.abspath(__file__))
G1P = HERE + '/../../2026-10-15-metal-spec-g1/prompts/'
G3P = HERE + '/../g3/prompts/'
CHAT = lambda p: '<|im_start|>user\n' + p + '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
C = float(os.environ.get("SIM_C", "0.06"))

def ids_of(err):
    t = open(err).read()
    m = re.search(r'^ids: (\[.*\])', t, re.M)
    return json.loads(m[1]), int(re.search(r'^prompt: (\d+) tokens', t, re.M)[1])

def load_cases():
    cases = {}
    gd = HERE + '/'
    for name, pf, chat in [('open-256','open',1),('grounded-256','grounded',1),('golden-docs-256','golden-docs',0),
                           ('golden-code-256','golden-code',0),('4k-256','4k',0),('short-256','short',0)]:
        g, n = ids_of(gd + f'1-off-{name}.err')
        p = open(G1P + pf + '.txt').read().rstrip('\n')
        pi = T.encode(CHAT(p) if chat else p, add_special_tokens=False).ids
        assert len(pi) == n, (name, len(pi), n)
        cases['sweep-' + name] = (pi, g)
    for f in sorted(glob.glob(HERE + '/../g3/ids/off-*-1.err')):
        row = re.search(r'off-(.*)-1\.err', f)[1]
        g, n = ids_of(f)
        pi = T.encode(CHAT(open(G3P + row + '.txt').read().rstrip('\n')), add_special_tokens=False).ids
        if len(pi) != n: print('prompt mismatch', row, len(pi), n, file=sys.stderr)
        cases['g3-' + row] = (pi, g)
    return cases

NMAX = 3
def propose(ctx, k, min_n):
    L = len(ctx)
    for n in range(NMAX, max(min_n, 1) - 1, -1):
        if L < n + 1: continue
        suf = ctx[L - n:]
        best = None
        for i in range(L - n - 1, -1, -1):
            if ctx[i:i + n] == suf:
                got = min(k, L - (i + n))
                if got == k: return ctx[i + n:i + n + k]
                if best is None or got > best[1]: best = (i + n, got)
        if best: return ctx[best[0]:best[0] + best[1]]
    return []

def simulate(prompt, g, k, fails, skip0, skip_max, min_n, gate_kind='streak', **kw):
    n = 1  # g[0] comes from prefill
    ctx = list(prompt) + [g[0]]
    streak = 0; skip = 0; nxt = skip0
    cost = 0.0; passes = 0; acc = 0; plain = 0
    ewma = kw.get('e0', 1.0)
    while n < len(g):
        if skip > 0:
            skip -= 1; allowed = False
        else:
            allowed = True
        if gate_kind == 'ewma' and ewma < kw['theta'] and not allowed_probe(kw, state=None): pass
        d = propose(ctx, k, min_n) if allowed else []
        if d:
            truth = g[n:n + len(d) + 1]
            j = 0
            while j < len(d) and j < len(truth) and d[j] == truth[j]: j += 1
            emit = g[n:n + j + 1]
            cost += 1 + C * len(d); passes += 1; acc += j
            n += len(emit); ctx += emit
            if fails:
                if j > 0: streak = 0; nxt = skip0
                else:
                    streak += 1
                    if streak >= fails:
                        skip = nxt; nxt = min(nxt * 2, skip_max)
        else:
            cost += 1; plain += 1; n += 1; ctx.append(g[n - 1])
    toks = len(g) - 1
    return toks / cost, passes, acc

def allowed_probe(*a, **k): return True

if __name__ == '__main__':
    cases = load_cases()
    configs = [tuple(map(int, c.split(','))) for c in sys.argv[1:]] or [(0,4,64,1),(2,4,64,1),(3,4,64,1)]
    print('| cfg(fails,skip0,max,min_n) | ' + ' | '.join(c.replace('sweep-','s-').replace('grounded-','g-').replace('open-','o-') for c in cases) + ' | mean(grounded) | mean(open) |')
    for cfg in configs:
        row = []; gs = []; os_ = []
        for name, (pi, g) in cases.items():
            sp, p, a = simulate(pi, g, 4, *cfg)
            row.append(f'{sp:.2f}')
            if name.startswith('g3-grounded'): gs.append(sp)
            if name.startswith('g3-open') or name == 'sweep-open-256': os_.append(sp)
        print(f'| {cfg} | ' + ' | '.join(row) + f' | {sum(gs)/max(len(gs),1):.2f} | {sum(os_)/max(len(os_),1):.3f} |')
