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
With `--dtype float32` (the golden an engine is judged against) a second bfloat16 pass over the same ids records, in the
metadata key `ref_spread`, how far HF's own bf16 forward is from the fp32 rows (top-1 agreement, mean KL): `intarsia accuracy`
derives its top-1 floor from it. `--no-bf16-spread` skips the pass.
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


def bf16_spread(args, tensors, ref_argmax):
    """How far the same model in bfloat16 is from the fp32 golden just built: top-1 agreement and mean KL over all rows.

    The fp32 golden is the reference `intarsia accuracy` judges an engine against, but the engine is a bf16 model: even
    HF's own bf16 forward does not reproduce the fp32 argmax on every row. That spread is what a bf16 engine can at best be
    expected to match, so the accuracy gate derives its top-1 floor from it (metadata `ref_spread`, accuracy/top1ref.rs).
    """
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model.eval().to(args.device)
    topk = args.topk > 0
    rows = flips = 0
    kl_sum = 0.0
    for i, ref_top in enumerate(ref_argmax):
        ids = tensors[f"ids_{i}"]
        with torch.no_grad():
            lp = torch.log_softmax(model(torch.tensor([ids.tolist()], device=args.device)).logits[0].float(), dim=-1).cpu()
        flips += int((lp.argmax(dim=-1).numpy() != ref_top).sum())
        rows += len(ids)
        if topk:
            # 21-way KL, the gate's metric for a top-k golden: the reference's top-k plus one "everything else" bucket
            v = torch.from_numpy(tensors[f"topk_logprobs_{i}"])
            ix = torch.from_numpy(tensors[f"topk_ids_{i}"]).long()
            pr, pb = v.exp(), lp.gather(1, ix).exp()
            orr, ob = (1 - pr.sum(-1)).clamp_min(1e-12), (1 - pb.sum(-1)).clamp_min(1e-12)
            kl = (pr * (v - pb.clamp_min(1e-30).log())).sum(-1) + orr * (orr / ob).log()
        else:
            ref = torch.from_numpy(tensors[f"logprobs_{i}"].astype(np.float32))
            kl = (ref.exp() * (ref - lp)).sum(-1)
        kl_sum += float(kl.sum())
        del lp
    del model
    return {
        "format": "intarsia-ref-spread-v1",
        "method": "hf_bf16_vs_fp32",
        "top1_agree_pct": 100.0 * (rows - flips) / rows,
        "top1_flips": flips,
        "n": rows,
        "kl_kind": "21-way" if topk else "full-vocab",
        "kl_mean": kl_sum / rows,
        "date": time.strftime("%Y-%m-%d", time.gmtime()),
    }


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
    ap.add_argument("--no-bf16-spread", action="store_true",
                    help="with --dtype float32: skip the second pass that records how far this model's own bf16 forward "
                         "is from the fp32 golden (metadata `ref_spread`, which `intarsia accuracy` turns into its top-1 floor)")
    args = ap.parse_args()
    dtype = getattr(torch, args.dtype)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.eval().to(args.device)
    prompts = gl.load_prompts(args.prompts)

    def apply_template(messages):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    tensors, meta_prompts, ref_argmax = {}, [], []
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
        ref_argmax.append(lp.argmax(dim=-1).numpy())
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
    if args.dtype == "float32" and not args.no_bf16_spread:
        del model
        t1 = time.time()
        spread = bf16_spread(args, tensors, ref_argmax)
        spread["how"] = (f"tools/golden.py second pass: the same HF model in bfloat16 on {args.device}, same token ids, one forward per prompt; "
                         f"top-1 = bf16 argmax equals the fp32 golden's argmax (all rows); KL = {spread['kl_kind']} KL(fp32 || bf16), mean over all rows")
        meta["ref_spread"] = json.dumps(spread)
        print(f"ref_spread: bf16 vs fp32 top-1 agreement {spread['top1_agree_pct']:.2f}% ({spread['top1_flips']} flips in {spread['n']} rows), "
              f"{spread['kl_kind']} KL {spread['kl_mean']:.3e}  ({time.time() - t1:.0f}s)", flush=True)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out), metadata=meta)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
