#!/usr/bin/env python3
"""Compare two intarsia-golden-topk-v1 goldens over the same ids (e.g. HF transformers vs vLLM of one checkpoint): per prompt top-1 agreement
and KL(A || B) on A's top-K support, B's log-probs taken where B lists the token, else floored at B's K-th log-prob (a bound, not
exact). usage: golden_cmp.py A.safetensors B.safetensors"""
import json, sys
import numpy as np
from safetensors import safe_open

a, b = safe_open(sys.argv[1], "np"), safe_open(sys.argv[2], "np")
meta = json.loads(a.metadata()["prompts"])
tot, n_all, agree_all = 0.0, 0, 0
for j, m in enumerate(meta):
    ia, ib = a.get_tensor(f"topk_ids_{j}"), b.get_tensor(f"topk_ids_{j}")
    la, lb = a.get_tensor(f"topk_logprobs_{j}").astype(np.float64), b.get_tensor(f"topk_logprobs_{j}").astype(np.float64)
    assert (a.get_tensor(f"ids_{j}") == b.get_tensor(f"ids_{j}")).all()
    n = ia.shape[0]
    kls, agree = [], 0
    for t in range(n):
        agree += int(ia[t, 0] == ib[t, 0])
        d = {int(i): l for i, l in zip(ib[t], lb[t]) if i >= 0}
        floor = lb[t][lb[t] > -1e29].min()
        pa = np.exp(la[t])
        lbv = np.array([d.get(int(i), floor) for i in ia[t]])
        mask = ia[t] >= 0
        kls.append(float((pa[mask] * (la[t][mask] - lbv[mask])).sum() / pa[mask].sum()))
    print(f"{m.get('name', j):18s} n={n:4d} top1-agree {100 * agree / n:6.2f}%  mean KL(A||B, supp) {np.mean(kls):.3e}")
    tot += sum(kls); n_all += n; agree_all += agree
print(f"OVERALL n={n_all} top1-agree {100 * agree_all / n_all:.2f}%  mean KL {tot / n_all:.3e}")
