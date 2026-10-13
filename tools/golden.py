#!/usr/bin/env python3
"""Generate golden logprobs from the Hugging Face transformers reference.

For each prompt in the prompt set the model runs ONE bf16 forward over the whole token
sequence; the logits are upcast to fp32, log_softmax'ed over the vocab, stored as fp16
(full-vocab format `intarsia-golden-v1`) or as the top-K log-probs per position
(`--topk K`, format `intarsia-golden-topk-v1`).
Format and usage: docs/accuracy.md. Consumed by `intarsia accuracy --golden FILE`.

    uv run golden.py --model DIR --out ../golden/NAME.safetensors [--device cpu|cuda]
                     [--prompts a.jsonl b.jsonl] [--topk 20]

Default prompt sets: prompts/golden.jsonl (8 short prompts) + prompts/golden_long.jsonl (2 prompts
of 300 and 700 tokens, so default 256-token prefill chunks are crossed). Only torch, transformers,
safetensors and numpy are needed (runs inside the vLLM image).
"""
import argparse
import json
import pathlib
import time

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

import goldenlib as gl

HERE = pathlib.Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF model dir (config.json, tokenizer, safetensors)")
    ap.add_argument("--out", required=True, help="output .safetensors file")
    ap.add_argument("--prompts", nargs="+",
                    default=[str(HERE / "prompts" / "golden.jsonl"), str(HERE / "prompts" / "golden_long.jsonl")])
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--template", default="chatml", choices=["chatml", "hf"],
                    help="chat prompts: fixed ChatML text (default) or the model's own chat template")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"],
                    help="reference precision (float32 only for diagnosing how much bf16 noise the reference has)")
    ap.add_argument("--topk", type=int, default=0,
                    help="store only the top-K log-probs per position (format intarsia-golden-topk-v1); "
                         "0 = full vocabulary as fp16 (format intarsia-golden-v1)")
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.eval().to(args.device)
    prompts = gl.load_prompts(args.prompts)

    def apply_template(messages):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    tensors, meta_prompts = {}, []
    t0 = time.time()
    vocab = 0
    for i, p in enumerate(prompts):
        ids, text = gl.prompt_ids(
            p, lambda t: tok(t, add_special_tokens=True).input_ids, args.template, apply_template)
        n = len(ids)
        with torch.no_grad():
            logits = model(torch.tensor([ids], device=args.device)).logits[0].float()  # [n, vocab]
        lp = torch.log_softmax(logits, dim=-1).cpu()
        vocab = lp.shape[1]
        tensors[f"ids_{i}"] = np.array(ids, dtype=np.int32)
        if args.topk > 0:
            v, ix = lp.topk(args.topk, dim=-1)
            tensors[f"topk_ids_{i}"] = ix.to(torch.int32).numpy()
            tensors[f"topk_logprobs_{i}"] = v.numpy().astype(np.float32)
        else:
            tensors[f"logprobs_{i}"] = lp.numpy().astype(np.float16)
        meta_prompts.append({"name": p["name"], "kind": p["kind"], "n_tokens": n, "text": text})
        print(f"[{i}] {p['name']:<16} {n:4d} tokens  vocab={vocab}  argmax_last={int(lp[-1].argmax())}", flush=True)

    meta = {
        "format": gl.FORMAT_TOPK if args.topk > 0 else gl.FORMAT_FULL,
        "model": str(args.model),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "device": args.device,
        "dtype": args.dtype,
        "template": args.template,
        "vocab": str(vocab),
        "prompts": json.dumps(meta_prompts),
    }
    if args.topk > 0:
        meta["topk"] = str(args.topk)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out), metadata=meta)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
