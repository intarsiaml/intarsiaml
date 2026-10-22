import sys, json, numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/Users/josef/Models/Qwen3.6-35B-A3B-NVFP4")
files = sys.argv[1:]
F = [safe_open(f, "np") for f in files]
n = len(json.loads(F[0].metadata()["prompts"]))
cache = {}
def isdig(t):
    if t not in cache:
        s = tok.decode([int(t)]).strip()
        cache[t] = bool(s) and (s[0].isdigit() or s[0] in "$-")
    return cache[t]
P = np.zeros((len(F), n)); T = [[None]*n for _ in F]
for fi, f in enumerate(F):
    for i in range(n):
        ids = f.get_tensor(f"topk_ids_{i}")[-1]; lp = f.get_tensor(f"topk_logprobs_{i}")[-1]
        m = ids >= 0
        P[fi, i] = sum(np.exp(l) for t, l in zip(ids[m], lp[m]) if isdig(t))
        T[fi][i] = tok.decode([int(ids[0])])
for fi, f in enumerate(files):
    print(f.split('/')[-1], "mean P(digit-start)", P[fi].mean().round(4), "argmax digit", sum(isinstance(x,str) and x.strip()[:1].isdigit() for x in T[fi]))
for a in range(len(F)):
    for b in range(a+1, len(F)):
        d = P[a]-P[b]; print(f"paired {a}-{b}: mean diff {d.mean():+.4f} sd {d.std(ddof=1):.4f} se {d.std(ddof=1)/np.sqrt(n):.4f}; argmax agree {sum(x==y for x,y in zip(T[a],T[b]))}/{n}")
np.save(sys.argv[1] + ".P.npy", P)
