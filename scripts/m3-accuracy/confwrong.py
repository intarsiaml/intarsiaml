import sys, json
import numpy as np
from safetensors import safe_open
f = safe_open(sys.argv[1], 'np')
md = json.loads(f.metadata()['prompts'])
for i, p in enumerate(md):
    ids = f.get_tensor(f'ids_{i}')
    tk = f.get_tensor(f'topk_ids_{i}')
    tl = f.get_tensor(f'topk_logprobs_{i}').astype(np.float64)
    n = len(ids)
    p1 = np.exp(tl[:-1, 0])
    intop = np.array([ids[t + 1] in tk[t] for t in range(n - 1)])
    cw = (p1 > 0.5) & ~intop
    cw9 = (p1 > 0.9) & ~intop
    print('%-14s n=%5d  confident(>0.5)&actual-not-in-top20: %4d (%.2f%%)   >0.9: %4d (%.2f%%)' % (p['name'], n, cw.sum(), 100 * cw.mean(), cw9.sum(), 100 * cw9.mean()))
