import sys, json, numpy as np
from safetensors import safe_open
def kl(A, B, n, last=False):
    out = []
    for i in range(n):
        ia, la = A.get_tensor(f"topk_ids_{i}"), A.get_tensor(f"topk_logprobs_{i}")
        ib, lb = B.get_tensor(f"topk_ids_{i}"), B.get_tensor(f"topk_logprobs_{i}")
        rows = [len(ia)-1] if last else range(len(ia))
        for t in rows:
            m = ia[t] >= 0
            pb = dict(zip(ib[t].tolist(), lb[t].tolist())); lo = min(pb.values())
            # 21-way: top-20 of A plus tail
            pa = np.exp(la[t][m]); qa = np.array([np.exp(pb.get(k, lo)) for k in ia[t][m].tolist()])
            ta, tb = max(1-pa.sum(), 1e-12), max(1-qa.sum(), 1e-12)
            out.append(float((pa*np.log(pa/qa)).sum() + ta*np.log(ta/tb)))
    return np.mean(out)
fs = sys.argv[1:]; F = [safe_open(f, "np") for f in fs]
n = min(len(json.loads(f.metadata()["prompts"])) for f in F)
names = [f.split("/")[-1].split("-gsm")[0] for f in fs]
for a in range(len(F)):
    for b in range(len(F)):
        if a != b: print(f"KL21({names[a]} || {names[b]}) all rows {kl(F[a],F[b],n):.4f}  last row {kl(F[a],F[b],n,True):.4f}")
