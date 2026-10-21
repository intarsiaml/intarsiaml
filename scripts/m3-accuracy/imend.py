import sys, json
import numpy as np
from safetensors import safe_open
IM_END = 248046
for path in sys.argv[1:]:
    f = safe_open(path, 'np')
    md = json.loads(f.metadata()['prompts'])
    print(path.split('/')[-1])
    for i, p in enumerate(md):
        ids = f.get_tensor(f'ids_{i}'); tk = f.get_tensor(f'topk_ids_{i}'); tl = f.get_tensor(f'topk_logprobs_{i}').astype(np.float64)
        n = len(ids)
        is_top = (tk[:-1, 0] == IM_END)
        actual = (ids[1:] == IM_END)
        wrong = is_top & ~actual
        pm = np.array([np.exp(tl[t][tk[t] == IM_END][0]) if (tk[t] == IM_END).any() else 0.0 for t in range(n - 1)])
        print('  %-14s n=%5d  im_end top1 & not actual: %4d (%.1f%%)  mean P(im_end) on non-im_end rows %.3f' % (
            p['name'], n, wrong.sum(), 100 * wrong.mean(), pm[~actual].mean()))
