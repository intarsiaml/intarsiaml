"""KL21 (ref top-20 support + other bucket) of self-golden B against self-golden A, prefill and decode rows, plus confident-row misses.
usage: klpair.py A.safetensors B.safetensors"""
import sys, json
import numpy as np
from safetensors import safe_open
a, b = safe_open(sys.argv[1], "np"), safe_open(sys.argv[2], "np")
n = len(json.loads(a.metadata()["prompts"]))
for pre in ("", "dec_"):
    tot = cnt = rows = miss = top1 = 0
    for j in range(n):
        k = f"{pre}topk_ids_{j}"
        if k not in a.keys() or k not in b.keys():
            continue
        ai, al = a.get_tensor(k)[:, :20], a.get_tensor(f"{pre}topk_logprobs_{j}")[:, :20].astype(np.float64)
        bi, bl = b.get_tensor(k), b.get_tensor(f"{pre}topk_logprobs_{j}").astype(np.float64)
        if pre and not np.array_equal(a.get_tensor(f"dec_pos_{j}"), b.get_tensor(f"dec_pos_{j}")):
            print("decode positions differ", j); continue
        for t in range(ai.shape[0]):
            d = dict(zip(bi[t].tolist(), bl[t].tolist())); floor = bl[t][-1]
            pr = np.exp(al[t]); pc = np.exp([d.get(int(i), floor) for i in ai[t]])
            rr, rc = max(1 - pr.sum(), 1e-12), max(1 - pc.sum(), 1e-12)
            tot += float((pr * (np.log(pr) - np.log(np.maximum(pc, 1e-30)))).sum() + rr * (np.log(rr) - np.log(rc))); cnt += 1
            hit = bi[t][0] == ai[t][0]; top1 += hit
            if pr[0] - pr[1] >= 0.5:
                rows += 1; miss += (not hit)
    print(f"{pre or 'prefill'}: KL21 {tot/max(cnt,1):.3e} over {cnt} rows, top-1 agree {100*top1/max(cnt,1):.2f} %, confident misses {miss}/{rows}")
