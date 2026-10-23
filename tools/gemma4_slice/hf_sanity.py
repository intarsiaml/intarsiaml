#!/usr/bin/env python3
"""HF-side sanity of the full Gemma-4-26B-A4B-NVFP4 reference (tools/gemma4_nvfp4_golden.py loader): greedy continuations of the
quick-tier sanity prompts and the mean negative log-likelihood of a plain text, so a wrong REFERENCE is told apart from a wrong
engine. usage: TORCH_DISABLE_NATIVE_JIT=1 python hf_sanity.py MODEL_DIR [--device cuda] [--bos]"""
import sys, pathlib, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import gemma4_nvfp4_golden as g
from transformers import AutoTokenizer

d = sys.argv[1]
BOS = [2] if "--bos" in sys.argv else []  # this checkpoint's tokenizer.json adds no BOS; raw text needs one (docs/results)
g.DEV = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda"
tok = AutoTokenizer.from_pretrained(d)
model = g.build_model(d)
prompts = ["The capital of France is", "The sky is", "Water is made of hydrogen and", "def fibonacci(n):"]
for p in prompts:
    ids = torch.tensor([BOS + tok(p).input_ids]).to(g.DEV)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=24, do_sample=False)
    print(repr(p), "->", repr(tok.decode(out[0, ids.shape[1]:])), flush=True)
msgs = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
t = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
ids = tok(t, return_tensors="pt", add_special_tokens=False).input_ids.to(g.DEV)
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=24, do_sample=False)
print("chat ->", repr(tok.decode(out[0, ids.shape[1]:])), flush=True)
text = open(pathlib.Path(__file__).resolve().parent.parent / "prompts" / "ppl.txt").read()[:6000]
ids = torch.tensor([BOS + tok(text).input_ids])[:, :1024].to(g.DEV)
with torch.no_grad():
    lg = model(ids).logits[0].float()
nll = torch.nn.functional.cross_entropy(lg[:-1], ids[0, 1:]).item()
print(f"ppl over {ids.shape[1] - 1} tokens: {torch.tensor(nll).exp().item():.2f}")
