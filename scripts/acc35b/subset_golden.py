#!/usr/bin/env python3
"""Write a golden restricted to some prompts (the accuracy CLI has no prompt filter).

    uv run --project tools python scripts/acc35b/subset_golden.py IN.safetensors OUT.safetensors name1,name2,...

Copies ids_i / topk_* tensors of the chosen prompts under new consecutive indices and rewrites the `prompts`
metadata (and any per-prompt list metadata of the same length) in the same order. Everything else is kept.
"""
import json, sys
from safetensors import safe_open
from safetensors.numpy import save_file

src, dst, names = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
f = safe_open(src, "np")
meta = dict(f.metadata())
prompts = json.loads(meta["prompts"])
idx = {p["name"]: i for i, p in enumerate(prompts)}
missing = [n for n in names if n not in idx]
if missing:
    sys.exit(f"unknown prompts {missing}; have {list(idx)}")
keys = list(f.keys())
per = {}
other = {}
for k in keys:
    base, _, num = k.rpartition("_")
    if num.isdigit():
        per.setdefault(int(num), []).append((base, k))
    else:
        other[k] = f.get_tensor(k)
out = dict(other)
for new, n in enumerate(names):
    for base, k in per[idx[n]]:
        out[f"{base}_{new}"] = f.get_tensor(k)
meta["prompts"] = json.dumps([prompts[idx[n]] for n in names])
meta["subset_of"] = src.rsplit("/", 1)[-1]
save_file(out, dst, metadata=meta)
print(f"wrote {dst}: {names}")
