import sys, json, numpy as np
from safetensors import safe_open
A, B = safe_open(sys.argv[1], "np"), safe_open(sys.argv[2], "np")
n = len(json.loads(A.metadata()["prompts"]))
bins = {}
for i in range(n):
    ia, la = A.get_tensor(f"topk_ids_{i}"), A.get_tensor(f"topk_logprobs_{i}")
    ib, lb = B.get_tensor(f"topk_ids_{i}"), B.get_tensor(f"topk_logprobs_{i}")
    for t in range(len(ia)):
        ma = ia[t] >= 0
        pa = dict(zip(ia[t][ma].tolist(), la[t][ma].tolist())); pb = dict(zip(ib[t].tolist(), lb[t].tolist()))
        lo = min(pb.values())
        kl = sum(np.exp(l) * (l - pb.get(k, lo)) for k, l in pa.items())
        bins.setdefault(t // 50, []).append(kl)
for k in sorted(bins): print(f"pos {k*50:4d}-{k*50+49:4d}  n={len(bins[k]):5d}  meanKL {np.mean(bins[k]):.4f}  p99 {np.percentile(bins[k],99):.3f}")
