"""Write a perturbed copy of a top-k golden (same ids and prompts, noisy log-probs, re-sorted) for cross-checking the
Rust `intarsia ref-noise` against scripts/m3-accuracy/setkl.py. Usage: uv run --no-project --with numpy --with safetensors python scripts/a2-ref-noise/perturb_topk_golden.py SRC.safetensors DST.safetensors"""
import sys, json
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

src, dst = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(7)
out = {}
with safe_open(src, "np") as f:
    md = f.metadata()
    for k in f.keys():
        t = f.get_tensor(k)
        if k.startswith("topk_ids_"):
            out[k] = t.copy()
        elif k.startswith("topk_logprobs_"):
            out[k] = t.copy()
        else:
            out[k] = t
    n = 0
    while f"topk_ids_{n}" in out:
        ids, lp = out[f"topk_ids_{n}"], out[f"topk_logprobs_{n}"].astype(np.float64)
        noisy = lp + rng.normal(0, 0.15, lp.shape)
        # drop the 3 lowest-ranked ids from the candidate and replace them by unrelated ids at a low probability
        noisy[:, -3:] = np.log(1e-4)
        ids = ids.copy()
        ids[:, -3:] = np.array([100001, 100002, 100003])
        order = np.argsort(-noisy, axis=1, kind="stable")
        out[f"topk_ids_{n}"] = np.take_along_axis(ids, order, 1)
        out[f"topk_logprobs_{n}"] = np.take_along_axis(noisy, order, 1).astype(np.float32)
        n += 1
save_file(out, dst, metadata=md)
print("wrote", dst, n, "prompts")
