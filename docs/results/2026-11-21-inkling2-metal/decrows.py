"""Per-row 21-way KL of decode rows (self-golden dec_*) vs the golden, two self-goldens side by side. usage: decrows.py GOLDEN OLD NEW"""
import sys, json, numpy as np
from safetensors import safe_open
g, a, b = (safe_open(p, "np") for p in sys.argv[1:4])
meta = json.loads(g.metadata()["prompts"])
def kl21(gi, gl, oi, ol):
    # 21-way: golden top-20 + "rest"; ours mapped onto the golden ids (missing -> rest)
    gp = np.exp(gl.astype(np.float64)); rest_g = max(1 - gp.sum(), 1e-12)
    om = dict(zip(oi.tolist(), np.exp(ol.astype(np.float64)).tolist()))
    op = np.array([om.get(int(t), 0.0) for t in gi]); rest_o = max(1 - op.sum(), 1e-12); op = np.maximum(op, 1e-12)
    P = np.append(gp, rest_g); Q = np.append(op, rest_o)
    return float((P * np.log(P / Q)).sum()), float(gp[0] - (gp[1] if len(gp) > 1 else 0))
for i, m in enumerate(meta):
    ids = g.get_tensor(f"ids_{i}"); gi_all, gl_all = g.get_tensor(f"topk_ids_{i}"), g.get_tensor(f"topk_logprobs_{i}")
    pos = a.get_tensor(f"dec_pos_{i}")
    rows = []
    for j, p in enumerate(pos):
        r = []
        for s in (a, b):
            r.append(kl21(gi_all[p], gl_all[p], s.get_tensor(f"dec_topk_ids_{i}")[j], s.get_tensor(f"dec_topk_logprobs_{i}")[j]))
        rows.append((int(p), r[0][0], r[1][0], r[0][1]))
    big = [x for x in rows if max(x[1], x[2]) > 0.02]
    print(f"{m['name']:16s} mean old {np.mean([x[1] for x in rows]):.3e} new {np.mean([x[2] for x in rows]):.3e}  median old {np.median([x[1] for x in rows]):.2e} new {np.median([x[2] for x in rows]):.2e}")
    for p, o, n, mg in big:
        print(f"    pos {p}: old {o:.3f} new {n:.3f}  golden p1-p2 {mg:.3f}")
