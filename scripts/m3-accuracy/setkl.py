"""Per-prompt and aggregate offline KL21(ref||cand) over a whole golden set, candidates truncated to top-20 for fairness.
usage: setkl.py REF CAND [CAND ...]"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split import load, kl21_rows, IM_END, IM_START
ref = load(sys.argv[1])
T = int(os.environ.get('TRUNC', '20'))
for cpath in sys.argv[2:]:
    cand = load(cpath)
    tot, n_all, tot_nochat, n_nochat = 0.0, 0, 0.0, 0
    parts = []
    for name, (ids, rid, rlp) in ref.items():
        if name not in cand:
            continue
        cids, cid, clp = cand[name]
        assert (ids == cids).all(), name
        kl, *_ = kl21_rows(rid, rlp, cid[:, :T], clp[:, :T])
        n = len(ids)
        sp = [t for t in range(n) if ids[t] in (IM_START, IM_END)]
        u = np.zeros(n, bool)
        for a, b in zip(sp[0::2], sp[1::2]):
            # rows inside system/user turns: <|im_start|>role ... <|im_end|> where the role token is not 'assistant'
            u[a:b] = True
        # keep assistant turns out of the mask: the role token follows <|im_start|>
        for a, b in zip(sp[0::2], sp[1::2]):
            if a + 1 < n and ids[a + 1] == 74455:  # 'assistant'
                u[a:b] = False
        parts.append(f'{name}={kl.mean():.3e}')
        tot += kl.sum(); n_all += n
        tot_nochat += kl[~u].sum(); n_nochat += (~u).sum()
    print(f'{os.path.basename(cpath):40s} ALL {tot / n_all:.3e} (n={n_all})  excl. system/user-turn rows {tot_nochat / n_nochat:.3e} (n={n_nochat})  | ' + ' '.join(parts))
