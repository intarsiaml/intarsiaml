#!/usr/bin/env python3
"""Generate golden logprobs from the Hugging Face transformers reference.

For each prompt in the prompt set the model runs ONE bf16 forward over the whole token
sequence; the logits are upcast to fp32, log_softmax'ed over the vocab, stored as fp16.
Format and usage: docs/accuracy.md. Consumed by `intarsia accuracy --golden FILE`.

    uv run golden.py --model DIR --out ../golden/NAME.safetensors [--device cpu|cuda]

Only torch, transformers, safetensors and numpy are needed (runs inside the vLLM image).
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = pathlib.Path(__file__).resolve().parent
MIN_TOK, MAX_TOK = 8, 256


def chatml(messages, add_generation_prompt=True):
    """Plain ChatML (Qwen family). Fixed text, independent of the model's own template."""
    s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return s + ("<|im_start|>assistant\n" if add_generation_prompt else "")


def prompt_text(p, tok, template):
    if p["kind"] == "raw":
        return p["text"]
    if template == "chatml":
        return chatml(p["messages"])
    return tok.apply_chat_template(p["messages"], tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF model dir (config.json, tokenizer, safetensors)")
    ap.add_argument("--out", required=True, help="output .safetensors file")
    ap.add_argument("--prompts", default=str(HERE / "prompts" / "golden.jsonl"))
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--template", default="chatml", choices=["chatml", "hf"],
                    help="chat prompts: fixed ChatML text (default) or the model's own chat template")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"],
                    help="reference precision (float32 only for diagnosing how much bf16 noise the reference has)")
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.eval().to(args.device)
    prompts = [json.loads(l) for l in open(args.prompts) if l.strip()]

    tensors, meta_prompts = {}, []
    t0 = time.time()
    for i, p in enumerate(prompts):
        text = prompt_text(p, tok, args.template)
        ids = tok(text, add_special_tokens=True, return_tensors="pt").input_ids
        n = ids.shape[1]
        if not MIN_TOK <= n <= MAX_TOK:
            sys.exit(f"prompt {i} ({p['name']}) has {n} tokens, want {MIN_TOK}..{MAX_TOK}")
        with torch.no_grad():
            logits = model(ids.to(args.device)).logits[0].float()  # [n, vocab]
        lp = torch.log_softmax(logits, dim=-1).cpu()
        tensors[f"ids_{i}"] = ids[0].to(torch.int32).numpy()
        tensors[f"logprobs_{i}"] = lp.numpy().astype(np.float16)
        meta_prompts.append({"name": p["name"], "kind": p["kind"], "n_tokens": n, "text": text})
        print(f"[{i}] {p['name']:<16} {n:4d} tokens  vocab={lp.shape[1]}  argmax_last={int(lp[-1].argmax())}", flush=True)

    meta = {
        "format": "intarsia-golden-v1",
        "model": str(args.model),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "device": args.device,
        "dtype": args.dtype,
        "template": args.template,
        "vocab": str(tensors["logprobs_0"].shape[1]),
        "prompts": json.dumps(meta_prompts),
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out), metadata=meta)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
