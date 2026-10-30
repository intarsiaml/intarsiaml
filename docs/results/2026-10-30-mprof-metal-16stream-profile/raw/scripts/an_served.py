#!/usr/bin/env python3
"""an_served.py TRACE.jsonl.gz [W2 ce1 trace] : served 16-stream burst (localeval speed --concurrency 1,8,16) decomposed from the
`ce1` trace: steps are cut at each embedding launch (t = rows of the step: <= 16 decode, larger = a prefill chunk); the
8- and 16-stream bursts are the segments around the long D8 / D16 runs. Prints wall decomposition and per-group ms / step."""
import sys, collections, statistics
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import mp
R = mp.load(sys.argv[1])
mp.exclusive(R)
S = []
for r in R:
    if r['op'] == 'embedding_bf16': S.append([])
    if S: S[-1].append(r)
info = []
for s in S:
    t = s[0]['t']
    info.append(dict(t=t, kind='P' if t > 16 else 'D', s=s[0]['s'], e=s[-1]['e'], recs=s))
# runs of the same row count among decode steps
def burst(rows):
    """index range [a, b) of the burst with `rows` streams: from the first prefill step before the run of D<rows> steps whose
    previous burst ended, to the last decode step with <= rows rows after the run."""
    idx = [i for i, x in enumerate(info) if x['kind'] == 'D' and x['t'] == rows]
    # longest contiguous run
    best = (0, 0); a = None
    for i in idx:
        if a is None or i != prev + 1: a = i
        prev = i
        if prev - a + 1 > best[1] - best[0]: best = (a, prev + 1)
    a, b = best
    # extend the end over the tail (decode steps with fewer rows, ends when a prefill or a 1-row run longer than 20 starts)
    e = b
    while e < len(info) and info[e]['kind'] == 'D' and info[e]['t'] < rows and info[e]['t'] >= 1 and (e - b) < rows: e += 1
    # extend the start back over the ramp: prefill steps and decode steps with fewer rows, until the previous long run
    st = a
    n_p = 0
    while st > 0:
        st -= 1
        if info[st]['kind'] == 'P':
            n_p += 1
            if n_p == rows: break
    return st, a, b, e
out = []
for rows in (8, 16):
    st, a, b, e = burst(rows)
    seg = info[st:e]
    t_all = info[e - 1]['e'] - info[st]['s']
    prefill = [x for x in info[st:a] if x['kind'] == 'P']
    ramp_d = [x for x in info[st:a] if x['kind'] == 'D']
    steady = info[a:b]
    tail = info[b:e]
    sd = lambda xs: sum(x['e'] - x['s'] for x in xs)
    toks = rows * 512
    out.append(f'\n#### {rows}-stream burst (ce1 trace)\n')
    out.append(f'* wall {t_all:.2f} s for {toks} tokens = {toks / t_all:.1f} tok/s aggregate (ce1 mode)')
    out.append(f'* prefill steps: {len(prefill)} x {sd(prefill) / max(len(prefill), 1) * 1e3:.0f} ms = {sd(prefill):.2f} s ({sd(prefill) / t_all * 100:.1f} % of wall)')
    out.append(f'* ramp decode steps (rows 1..{rows - 1}, between the prefills): {len(ramp_d)} steps, {sd(ramp_d):.2f} s ({sd(ramp_d) / t_all * 100:.1f} %)')
    gap_ramp = (info[a]['s'] - info[st]['s']) - sd(prefill) - sd(ramp_d)
    out.append(f'* ramp idle between steps: {gap_ramp * 1e3:.0f} ms')
    out.append(f'* steady D{rows}: {len(steady)} steps, {sd(steady):.2f} s ({sd(steady) / t_all * 100:.1f} %), {sd(steady) / len(steady) * 1e3:.2f} ms / step (GPU span first->last record), wall per step {(steady[-1]["e"] - steady[0]["s"]) / len(steady) * 1e3:.2f} ms = {rows * len(steady) / (steady[-1]["e"] - steady[0]["s"]):.1f} tok/s decode-only')
    out.append(f'* tail (draining streams): {len(tail)} steps, {sd(tail):.2f} s ({sd(tail) / t_all * 100:.1f} %)')
    # per-group ms per steady step
    n = len(steady); g = collections.defaultdict(float)
    for x in steady[4:-4]:
        for r in x['recs']:
            g[mp.kind(r, rows)[0]] += r['dur'] * 1e3 / (n - 8)
    out.append('* ce1 ms / steady step by group: ' + ', '.join(f'{k} {v:.2f}' for k, v in sorted(g.items(), key=lambda kv: -kv[1])) + f'; total {sum(g.values()):.2f}')
    ops = collections.Counter()
    for x in steady[4:-4]:
        for r in x['recs']: ops[r['op']] += 1
    out.append('* launches per steady step: ' + ', '.join(f'{k} {v / (n - 8):.1f}' for k, v in ops.most_common()))
print('\n'.join(out))
