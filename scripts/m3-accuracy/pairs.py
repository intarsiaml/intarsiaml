"""Mean KL21(ref||cand) for pairs of golden/dump files on one prompt, plus user-turn / rest split.
usage: pairs.py PROMPT refA:candA [refB:candB ...]   (paths may be aliases defined below)"""
import sys, json
import numpy as np
sys.path.insert(0, '/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/m3')
from split import load, kl21_rows, IM_END, IM_START
D = '/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/m3/'
A = {'gold': '/opt/code/intarsiaml/golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors'}
cache = {}
def get(k):
    p = A.get(k, k if k.startswith('/') else D + k)
    if p not in cache:
        cache[p] = load(p)
    return cache[p]
name = sys.argv[1]
for pair in sys.argv[2:]:
    r, c = pair.split(':')
    ids, rid, rlp = get(r)[name]
    cids, cid, clp = get(c)[name]
    assert (ids == cids).all()
    kl, pir, pic, _ = kl21_rows(rid, rlp, cid, clp)
    n = len(ids)
    sp = [t for t in range(n) if ids[t] in (IM_START, IM_END)]
    if len(sp) >= 4:
        u = np.zeros(n, bool); u[sp[2]:sp[3]] = True  # the long user turn [<|im_start|> user .. <|im_end|>)
    else:
        u = np.zeros(n, bool)
    top1 = np.mean(rid[:, 0] == cid[:, 0])
    print(f'{name:14s} {r:>34s} || {c:<34s} mean KL21 {kl.mean():.3e}  user-turn rows {kl[u].mean() if u.any() else 0:.3e} (n={u.sum()})  other rows {kl[~u].mean():.3e} (n={(~u).sum()})  top1 agree {top1:.4f}  max {kl.max():.2f}')
