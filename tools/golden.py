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
of 300 and 700 tokens, so default 256-token prefill chunks are crossed). Long sets: prompts/golden_4k.jsonl
(~4k-token prompts, each followed by 64 tokens of this model's own greedy continuation, see goldenlib.py);
use `--topk 20` for them (a full-vocabulary 4k golden is over a GB per prompt). The metadata records the
reference engine (transformers + torch versions), model revision (snapshot hash of the path), dtype and the
KV dtype the reference attention used (its activation dtype), the template and the prompt files.
Only torch, transformers, safetensors and numpy are needed (runs inside the vLLM image).
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
    # a rendered chat template that already starts with the BOS token (Llama 3: <|begin_of_text|>) must not get a second one
    tokenize = lambda t: tok(t, add_special_tokens=not (tok.bos_token and t.startswith(tok.bos_token))).input_ids
    for i, p in enumerate(prompts):
        ids, text = gl.prompt_ids(p, tokenize, args.template, apply_template)
        n_prompt = len(ids)
        split = gl.reuse_split(p, ids, tokenize) if args.template == "chatml" else None
        want = int(p.get("continue", 0))
        if want:
            with torch.no_grad():
                gen = model.generate(torch.tensor([ids], device=args.device), attention_mask=torch.ones(1, n_prompt, device=args.device, dtype=torch.long),
                                     max_new_tokens=want, min_new_tokens=want, do_sample=False,
                                     pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else 0)
            ids = ids + [int(t) for t in gen[0, n_prompt:].tolist()]
        n = len(ids)
        with torch.no_grad():
            logits = model(torch.tensor([ids], device=args.device)).logits[0].float()  # [n, vocab]
        lp = torch.log_softmax(logits, dim=-1).cpu()
        del logits
        vocab = lp.shape[1]
        tensors[f"ids_{i}"] = np.array(ids, dtype=np.int32)
        if args.topk > 0:
            v, ix = lp.topk(args.topk, dim=-1)
            tensors[f"topk_ids_{i}"] = ix.to(torch.int32).numpy()
            tensors[f"topk_logprobs_{i}"] = v.numpy().astype(np.float32)
        else:
            tensors[f"logprobs_{i}"] = lp.numpy().astype(np.float16)
        meta_prompts.append(gl.meta_prompt(p, text, n, n_prompt, split))
        print(f"[{i}] {p['name']:<16} {n:5d} tokens ({n_prompt}+{n - n_prompt})  vocab={vocab}  "
              f"argmax_last={int(lp[-1].argmax())}  {time.time() - t0:.0f}s", flush=True)
        del lp

    kv_names = {"bfloat16": "bf16", "float32": "f32", "float16": "f16"}
    meta = {
        "format": gl.FORMAT_TOPK if args.topk > 0 else gl.FORMAT_FULL,
        "model": str(args.model),
        "model_revision": gl.snapshot_hash(args.model) or "not in the model path",
        "source": "hf",
        "engine": "hf-transformers",
        "engine_version": "transformers " + __import__("transformers").__version__ + ", torch " + torch.__version__,
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "device": args.device,
        "dtype": args.dtype,
        "kv_dtype": kv_names.get(args.dtype, args.dtype),
        "template": args.template,
        "add_special_tokens": "true",
        "sampling": json.dumps({"forward": "one full-sequence forward", "continuation": "greedy (generate, do_sample=False)"}),
        "prompt_files": json.dumps([pathlib.Path(x).name for x in args.prompts]),
        "vocab": str(vocab),
        "prompts": json.dumps(meta_prompts),
        "built_unix": str(int(time.time())),
    }
    if args.topk > 0:
        meta["topk"] = str(args.topk)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out), metadata=meta)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
