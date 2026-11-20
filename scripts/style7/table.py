#!/usr/bin/env python3
"""style7: summary table of the first-token files (raw/ft-*.json) against vLLM's three passes (mean over passes). Run with deepeval's venv (tokenizers)."""
import json, math, re, statistics as st, sys
from tokenizers import Tokenizer
R = '/opt/code/intarsiaml/.worktrees/style7/docs/results/2026-11-20-style7/raw/'
tok = Tokenizer.from_file('/opt/code/intarsiaml/.worktrees/style7/scratch/model/tokenizer.json')
def L(n): return {r['id']: r for r in json.load(open(R + f'ft-{n}.json'))}
def isd(t):
    s = tok.decode([t]).strip(); return bool(s) and any(c.isdigit() for c in s) and bool(re.fullmatch(r"[\d$,.-]+", s))
def pd(r): return sum(math.exp(l) for t, l in r['top'] if isd(t))
def lg(p): p = min(max(p, 1e-6), 1 - 1e-6); return math.log(p / (1 - p))
def kl(p, q):
    U = set(p) | set(q)
    def dist(d):
        rest = max(1e-9, 1 - sum(math.exp(v) for v in d.values())); per = rest / 150000
        return {t: (math.exp(d[t]) if t in d else per) for t in U}, rest
    P, rp = dist(p); Q, rq = dist(q)
    return sum(P[t] * math.log(P[t] / Q[t]) for t in U) + rp * math.log(rp / rq)
V = [L(f'vllm-p{k}') for k in (1, 2, 3)]
rows = []
for name in sys.argv[1:]:
    E = L(name); ids = sorted(E)
    d1 = sum(isd(E[i]['top'][0][0]) for i in ids)
    pdm = st.mean(pd(E[i]) for i in ids)
    if name.startswith('vllm'):
        others = [v for v in V if v is not E]
    else:
        others = V
    kls = st.mean(st.mean(kl(dict(map(tuple, E[i]['top'])), dict(map(tuple, v[i]['top']))) for i in ids) for v in others)
    sd = st.mean(sum(isd(E[i]['top'][0][0]) != isd(v[i]['top'][0][0]) for i in ids) for v in others)
    sh = st.mean(st.mean(lg(pd(v[i])) - lg(pd(E[i])) for i in ids) for v in others)
    rows.append({'config': name, 'digit_top1': d1, 'mean_p_digit': round(pdm, 4), 'mean_KL_vs_vllm_passes': round(kls, 4), 'style_differs_vs_vllm_passes_mean': round(sd, 1), 'mean_logit_shift_vllm_minus_this': round(sh, 3)})
for r in rows: print(r)
json.dump(rows, open(R + 'first-token-summary.json', 'w'), indent=1)
