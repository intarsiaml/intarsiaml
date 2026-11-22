import sys, json, numpy as np
from safetensors import safe_open
g, a = safe_open(sys.argv[1], "np"), safe_open(sys.argv[2], "np")
meta = json.loads(g.metadata()["prompts"])
def kl21(gi, gl, oi, ol):
    gp = np.exp(gl.astype(np.float64)); rest_g = max(1 - gp.sum(), 1e-12)
    om = dict(zip(oi.tolist(), np.exp(ol.astype(np.float64)).tolist()))
    op = np.array([om.get(int(t), 0.0) for t in gi]); rest_o = max(1 - op.sum(), 1e-12); op = np.maximum(op, 1e-12)
    P = np.append(gp, rest_g); Q = np.append(op, rest_o)
    return float((P * np.log(P / Q)).sum())
tot = []; 
for i, m in enumerate(meta):
    gi, gl = g.get_tensor(f"topk_ids_{i}"), g.get_tensor(f"topk_logprobs_{i}")
    pos = a.get_tensor(f"dec_pos_{i}"); oi, ol = a.get_tensor(f"dec_topk_ids_{i}"), a.get_tensor(f"dec_topk_logprobs_{i}")
    ks = [kl21(gi[p], gl[p], oi[j], ol[j]) for j, p in enumerate(pos)]
    tot += ks
    if m["name"] in sys.argv[3:]:
        print(m["name"], "pos", int(pos[0]), "..", int(pos[-1]))
        print("   ", " ".join(f"{k:.3f}" for k in ks))
s = sorted(tot, reverse=True)
print(f"all rows n={len(tot)} mean {np.mean(tot):.3e}; top-1 share {s[0]/sum(tot):.0%}, top-3 {sum(s[:3])/sum(tot):.0%}, top-5 {sum(s[:5])/sum(tot):.0%}; mean without top-5 {np.mean(s[5:]):.3e}")
