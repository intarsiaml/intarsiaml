"""Final normed hidden states h (the unembed input) of the W4A16 oracle (inkling2_ref.py) on a golden's prompts, for
inkling2_unembed_screen.py (lane ink-fp8, docs/results/2026-11-21-inkling2-fp8.md). Run from tools/:
  INK_MODEL=<checkpoint dir> uv run --project . python inkling2_dump_hidden.py GOLDEN OUT.npz [max_prompts]"""
import sys, json, os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np
from safetensors import safe_open
import inkling2_ref as R

ref = R.build(Path(os.environ.get("INK_MODEL", str(R.SNAPSHOT))), torch.float32)
hs = []
orig = ref.head
def head(blocks, partial):
    c, w = ref.c, ref.w
    h = ref.attn_res(blocks, partial, ref.out_q, w["norm.weight"][None])[0]
    hs.append(h.detach().to(torch.float32).numpy().copy())
    return orig(blocks, partial)
ref.head = head
f = safe_open(sys.argv[1], "pt")
n = sum(1 for k in f.keys() if k.startswith("ids_"))
n = min(n, int(sys.argv[3])) if len(sys.argv) > 3 else n
out = {}
for j in range(n):
    ids = f.get_tensor(f"ids_{j}").long()
    hs.clear()
    logits = ref.forward(ids, R.SeqState(ref.c))
    out[f"h_{j}"] = np.concatenate(hs, 0)
    print(j, out[f"h_{j}"].shape, flush=True)
np.savez(sys.argv[2], **out)
