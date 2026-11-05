#!/usr/bin/env python3
"""An ids-only topk-v1 golden (dummy log-probs) for prompts.jsonl, tokenised exactly as tools/goldenlib does: the CPU oracle
(`intarsia accuracy --backend cpu --write-self-golden`) reads its ids and writes the real reference. usage: MODEL_DIR PROMPTS.jsonl OUT"""
import json, sys, time
sys.path.insert(0, "/opt/code/intarsiaml/.worktrees/m35-peel/tools")
import numpy as np
from safetensors.numpy import save_file
from tokenizers import Tokenizer
import goldenlib as gl
model, pj, out = sys.argv[1:4]
tok = Tokenizer.from_file(model + "/tokenizer.json")
tokenize = lambda t: tok.encode(t, add_special_tokens=True).ids
vocab = json.load(open(model + "/config.json")); vocab = vocab.get("text_config", vocab)["vocab_size"]
T, mp = {}, []
for i, p in enumerate(gl.load_prompts([pj])):
    ids, text = gl.prompt_ids(p, tokenize)
    n = len(ids)
    T[f"ids_{i}"] = np.array(ids, dtype=np.int32)
    T[f"topk_ids_{i}"] = np.tile(np.arange(20, dtype=np.int32), (n, 1))
    T[f"topk_logprobs_{i}"] = np.full((n, 20), -3.0, dtype=np.float32)
    mp.append(gl.meta_prompt(p, text, n, n, None)); print(p["name"], n)
meta = {"format": gl.FORMAT_TOPK, "topk": "20", "model": model, "model_revision": "not in the model path", "source": "ids-only",
        "engine": "ids-only (dummy log-probs; ids for the CPU oracle)", "engine_version": "-", "device": "cpu", "dtype": "float32",
        "kv_dtype": "f32", "template": "chatml", "add_special_tokens": "true", "sampling": json.dumps({"continuation": "none"}),
        "prompt_files": json.dumps([pj.split("/")[-1]]), "vocab": str(vocab), "prompts": json.dumps(mp), "built_unix": str(int(time.time()))}
save_file(T, out, metadata=meta); print("wrote", out)
