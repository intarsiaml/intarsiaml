import sys, json
import numpy as np
from safetensors import safe_open
f = safe_open(sys.argv[1], 'np')
md = json.loads(f.metadata()['prompts'])
B = int(sys.argv[2]) if len(sys.argv) > 2 else 512
for i, p in enumerate(md):
    ids = f.get_tensor(f'ids_{i}')
    tk = f.get_tensor(f'topk_ids_{i}')
    tl = f.get_tensor(f'topk_logprobs_{i}').astype(np.float64)
    n = len(ids)
    # log-prob of the actual next token if in top-k, else (cap) the k-th logprob
    lp = np.empty(n - 1)
    intop = np.zeros(n - 1, bool)
    for t in range(n - 1):
        m = np.where(tk[t] == ids[t + 1])[0]
        if len(m):
            lp[t] = tl[t, m[0]]; intop[t] = True
        else:
            lp[t] = tl[t, -1]
    top1 = (tk[:-1, 0] == ids[1:])
    print('%-14s n=%5d  top1(actual)=%.3f  in-top20=%.3f  meanNLL(capped)=%.3f' % (p['name'], n, top1.mean(), intop.mean(), -lp.mean()))
    if p['name'].startswith('chat') or '-v' in sys.argv:
        for b in range(0, n - 1, B):
            e = min(b + B, n - 1)
            print('    %5d-%5d top1 %.2f in20 %.2f nll %.2f' % (b, e, top1[b:e].mean(), intop[b:e].mean(), -lp[b:e].mean()))
