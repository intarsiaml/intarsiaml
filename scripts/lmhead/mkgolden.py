#!/usr/bin/env python3
"""Prompts-only top-k golden for `intarsia accuracy --write-self-golden` (lane lmhead-fp8).

    uv run --with tokenizers --with numpy --with safetensors scripts/lmhead/mkgolden.py \
        MODEL_DIR OUT.safetensors NAME=WRAPPED.txt[:GEN.ids] ...

Each argument is a prompt name, a text file (already ChatML-wrapped) and optionally the file of greedy ids the bf16-head engine
generated for it (`ids: 1,2,3` as printed by `generate --show-ids`, or bare comma-separated). The golden holds tokens(text) +
generated ids, so the last positions (the teacher-forced decode window of `accuracy`) lie on the model's own distribution.
The reference rows are placeholders (a uniform top-20): the file is only read for its prompts, by a self-golden write.
"""
import json
import re
import sys

import numpy as np
from safetensors.numpy import save_file
from tokenizers import Tokenizer

model, out, *specs = sys.argv[1:]
tok = Tokenizer.from_file(f"{model}/tokenizer.json")
tensors, prompts = {}, []
for i, spec in enumerate(specs):
    name, rest = spec.split("=", 1)
    path, _, gen = rest.partition(":")
    text = open(path).read()
    ids = tok.encode(text, add_special_tokens=False).ids
    n_prompt = len(ids)
    if gen and gen.endswith(".txt"):  # generated TEXT (HIP server): the golden tokenises prompt + text
        ids = tok.encode(text + open(gen).read(), add_special_tokens=False).ids
    elif gen:
        m = re.search(r"ids:\s*([0-9, ]+)", open(gen).read())
        ids += [int(x) for x in m.group(1).replace(" ", "").strip(",").split(",") if x]
    k = 20
    tensors[f"ids_{i}"] = np.array(ids, dtype=np.int32)
    tensors[f"topk_ids_{i}"] = np.tile(np.arange(k, dtype=np.int32), (len(ids), 1))
    tensors[f"topk_logprobs_{i}"] = np.full((len(ids), k), -np.log(k), dtype=np.float32)
    prompts.append({"name": name, "n_tokens": len(ids), "n_prompt_tokens": n_prompt})
save_file(tensors, out, metadata={"format": "intarsia-golden-topk-v1", "topk": "20", "prompts": json.dumps(prompts),
                                  "note": "prompts only; placeholder reference rows (lane lmhead-fp8)"})
print(f"wrote {out}: " + ", ".join(f"{p['name']}={p['n_tokens']}" for p in prompts))
