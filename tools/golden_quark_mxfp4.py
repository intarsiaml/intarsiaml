#!/usr/bin/env python3
"""Golden log-probs for an AMD Quark OCP-MXFP4 Qwen3.5-family checkpoint (e.g. amd/Qwen3.8-27B-Quark-AWQ-MXFP4), made
with HF transformers' own Qwen3.5 decoder layers and Quark's own activation-scale rule, streaming one layer at a time.

Why a separate tool: tools/golden.py loads the whole model with AutoModelForCausalLM, which needs amd-quark (and its
scipy/... dependencies) for a Quark checkpoint and the full bf16 model in memory (~54 GB for 27B); the radiance server
that serves the same checkpoint ignores vLLM's prompt_logprobs, so tools/golden_vllm.py cannot use it either. This
script keeps peak memory at a few GB:

* every prompt's hidden states are pushed through one `Qwen3_5DecoderLayer` (HF modeling code, sdpa, the torch
  GatedDeltaNet fallback) at a time; the layer's MXFP4 weights are dequantised to bf16 (exact: e2m1 * 2^e) when it is
  built and dropped after it ran;
* every quantised projection gets its input QDQ'd to dynamic MXFP4 (W4A4, the checkpoint's `input_tensors` spec) by a
  forward pre-hook; the shared scale is Quark's `even_round`, extracted from the amd-quark wheel's source with `ast`
  (`--quark-site`), the e2m1 rounding is nearest-even with saturation at +-6 (independent of the Rust code; the same
  helpers as tools/gen_mxfp4_fixture.py). `--act none` skips the QDQ (a W4A16 golden);
* modules in the checkpoint's `exclude` list (lm_head, mtp, visual) stay bf16 and unquantised; the MTP head and the
  vision tower are not run.

Output: the `intarsia-golden-topk-v1` format of tools/golden.py `--topk K` (default 20), same prompt sets, same
metadata keys (engine `hf-transformers+quark-emulation`).

    tools/.venv/bin/python -I tools/golden_quark_mxfp4.py --model /models/Qwen3.8-27B-Quark-AWQ-MXFP4 \\
        --out golden/qwen3.8-27b-mxfp4-hfquark-top20.safetensors --quark-site <amd-quark site-packages>
"""
import argparse
import ast
import json
import pathlib
import struct
import sys
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file
from transformers import AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import goldenlib as gl  # noqa: E402

GRID = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)


def quark_even_round(site: pathlib.Path):
    src = (site / "quark/torch/quantization/utils.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "even_round")
    # Dtype only appears in an annotation; get_dtype_params is Quark's fp4 row (ebits 2, mbits 1, emax 2)
    ns = {"torch": torch, "Dtype": type("Dtype", (), {}), "get_dtype_params": lambda d: (2, 1, 2)}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "quark_even_round", "exec"), ns)
    return ns["even_round"]


def round_e2m1(v):
    a = v.abs().clamp(max=6.0)
    d = (a.unsqueeze(-1) - GRID).abs()
    cand = d == d.min(-1, keepdim=True).values
    idx = torch.arange(8)
    big = torch.full_like(idx, 99)
    even = torch.where(cand & (idx % 2 == 0), idx, big).min(-1).values
    first = torch.where(cand, idx, big).min(-1).values
    code = torch.where(even < 99, even, first)
    return torch.where(v < 0, -GRID[code], GRID[code])


def make_qdq(even_round):
    def qdq(x):  # [..., K] -> same dtype, QDQ'd per 32 along K
        shp, dt = x.shape, x.dtype
        b = x.float().reshape(-1, shp[-1] // 32, 32)
        scale = even_round(b.abs().max(-1, keepdim=True).values.clone(), "fp4").float()
        out = torch.empty_like(b)
        for s in range(0, b.shape[0], 64):  # bounded memory for the [rows, groups, 32, 8] distance tensor
            out[s:s + 64] = round_e2m1(b[s:s + 64] / scale[s:s + 64]) * scale[s:s + 64]
        return out.reshape(shp).to(dt)
    return qdq


def dequant(w, s):
    lo, hi = (w & 0x0F).to(torch.int32), ((w >> 4) & 0x0F).to(torch.int32)
    codes = torch.stack([lo, hi], -1).reshape(w.shape[0], -1)
    bits = ((codes & 0x07) << 22) | ((codes & 0x08) << 28)
    vals = bits.view(torch.float32) * (2.0 ** 126)
    sc = torch.exp2(s.float() - 127.0)
    return (vals * sc.repeat_interleave(32, dim=1)).to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", nargs="+",
                    default=[str(HERE / "prompts" / "golden.jsonl"), str(HERE / "prompts" / "golden_long.jsonl")])
    ap.add_argument("--template", default="chatml", choices=["chatml", "hf"])
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--act", default="dynamic", choices=["dynamic", "none"], help="MXFP4 activations: dynamic QDQ (W4A4) or none")
    ap.add_argument("--quark-site", required=True, help="site-packages dir holding the amd-quark wheel's quark/ sources")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--layers", type=int, default=0, help="debug: run only the first N layers")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    model = pathlib.Path(args.model)
    cfg = json.load(open(model / "config.json"))
    qc = cfg.get("quantization_config") or {}
    assert qc.get("quant_method") == "quark", "not a Quark checkpoint"
    exclude = {e[:-len(".weight")] if e.endswith(".weight") else e for e in qc.get("exclude", [])}
    tc = dict(cfg.get("text_config", cfg))
    tc.pop("quantization_config", None)
    config = Qwen3_5TextConfig(**tc)
    config._attn_implementation = "sdpa"
    qdq = make_qdq(quark_even_round(pathlib.Path(args.quark_site))) if args.act == "dynamic" else None

    files = sorted(model.glob("*.safetensors"))
    handles = [safe_open(str(f), "pt") for f in files]
    where = {k: h for h in handles for k in h.keys()}
    get = lambda k: where[k].get_tensor(k)

    tok = AutoTokenizer.from_pretrained(str(model))
    prompts = gl.load_prompts(args.prompts)
    tokenize = lambda t: tok(t, add_special_tokens=True).input_ids
    apply_template = lambda m: tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    items = []
    for p in prompts:
        assert not int(p.get("continue", 0)), "continuation prompts need a generate loop (not supported here)"
        ids, text = gl.prompt_ids(p, tokenize, args.template, apply_template)
        split = gl.reuse_split(p, ids, tokenize) if args.template == "chatml" else None
        items.append((p, ids, text, split))

    t0 = time.time()
    P = "model.language_model."
    emb = get(P + "embed_tokens.weight")
    hs = [emb[torch.tensor(ids)].unsqueeze(0) for _, ids, _, _ in items]
    del emb
    rot = hf.Qwen3_5TextRotaryEmbedding(config)
    pe = [rot(h, torch.arange(h.shape[1]).view(1, 1, -1)) for h in hs]
    n_layers = args.layers or config.num_hidden_layers
    nq = 0
    with torch.no_grad():
        for li in range(n_layers):
            layer = hf.Qwen3_5DecoderLayer(config, li).to(torch.bfloat16)
            pre = f"{P}layers.{li}."
            sd = {}
            for name, _ in layer.named_parameters():
                key = pre + name
                t = get(key)
                if t.dtype == torch.uint8:  # quantised linear: e2m1 codes next to e8m0 `weight_scale`
                    mod = key[:-len(".weight")]
                    t = dequant(t, get(mod + ".weight_scale"))
                    if qdq is not None and mod not in exclude:
                        layer.get_submodule(name[:-len(".weight")]).register_forward_pre_hook(
                            lambda m, a: (qdq(a[0]),) + tuple(a[1:]))
                        nq += 1
                sd[name] = t
            layer.load_state_dict(sd, strict=True)
            layer.eval()
            for i in range(len(hs)):
                hs[i] = layer(hs[i], position_embeddings=pe[i], attention_mask=None, position_ids=None)
            del layer, sd
            if li % 8 == 7 or li == n_layers - 1:
                print(f"layer {li + 1}/{n_layers}  {time.time() - t0:.0f}s", flush=True)
        norm = hf.Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(torch.bfloat16)
        norm.load_state_dict({"weight": get(P + "norm.weight")})
        head = get("lm_head.weight")
        tensors, meta_prompts = {}, []
        for i, (p, ids, text, split) in enumerate(items):
            logits = (norm(hs[i])[0] @ head.T).float()
            lp = torch.log_softmax(logits, dim=-1)
            v, ix = lp.topk(args.topk, dim=-1)
            tensors[f"ids_{i}"] = np.array(ids, dtype=np.int32)
            tensors[f"topk_ids_{i}"] = ix.to(torch.int32).numpy()
            tensors[f"topk_logprobs_{i}"] = v.numpy().astype(np.float32)
            meta_prompts.append(gl.meta_prompt(p, text, len(ids), len(ids), split))
            print(f"[{i}] {p['name']:<16} {len(ids):5d} tokens  argmax_last={int(lp[-1].argmax())}", flush=True)
    import transformers
    meta = {
        "format": gl.FORMAT_TOPK,
        "topk": str(args.topk),
        "model": str(model),
        "model_revision": gl.snapshot_hash(str(model)) or "not in the model path",
        "source": "hf",
        "engine": "hf-transformers+quark-emulation",
        "engine_version": f"transformers {transformers.__version__}, torch {torch.__version__}, layer-streamed, "
                          f"MXFP4 weights dequantised to bf16, activations {args.act} (Quark even_round), {nq} QDQ'd linears",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": "cpu",
        "dtype": "bfloat16",
        "kv_dtype": "bf16",
        "template": args.template,
        "add_special_tokens": "true",
        "sampling": json.dumps({"forward": "one full-sequence forward per prompt, layer by layer", "continuation": "none"}),
        "prompt_files": json.dumps([pathlib.Path(x).name for x in args.prompts]),
        "vocab": str(head.shape[0]),
        "prompts": json.dumps(meta_prompts),
        "built_unix": str(int(time.time())),
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out), metadata=meta)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s, {nq} QDQ'd linears")


if __name__ == "__main__":
    main()
