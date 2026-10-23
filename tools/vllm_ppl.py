#!/usr/bin/env python3
"""Perplexity of a vLLM server on tools/prompts/ppl.txt (prompt_logprobs; stdlib only), with or without a leading BOS id:
tells a wrong reference implementation from a model that is just poor on raw text. usage: vllm_ppl.py BASE_URL MODEL TOKENIZER_DIR [--bos ID]"""
import json, math, pathlib, sys, urllib.request

base, model, tokdir = sys.argv[1:4]
bos = int(sys.argv[sys.argv.index("--bos") + 1]) if "--bos" in sys.argv else None
text = (pathlib.Path(__file__).resolve().parent / "prompts" / "ppl.txt").read_text()[:6000]
def post(path, body):
    r = urllib.request.Request(base.rsplit("/v1", 1)[0] + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r))
ids = post("/tokenize", {"model": model, "prompt": text, "add_special_tokens": False})["tokens"]
if bos is not None:
    ids = [bos] + ids
ids = ids[:1024]
r = post("/v1/completions", {"model": model, "prompt": ids, "max_tokens": 1, "prompt_logprobs": 1, "temperature": 0})
pl = r["choices"][0]["prompt_logprobs"]
nll = [-pl[j][str(ids[j])]["logprob"] for j in range(1, len(ids))]
print(f"vllm ppl over {len(nll)} tokens (bos={bos}): {math.exp(sum(nll) / len(nll)):.2f}")
