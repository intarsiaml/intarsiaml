#!/usr/bin/env python3
"""Golden log-probs for a compressed-tensors NVFP4A16 (`nvfp4-pack-quantized`) Qwen3.5-family checkpoint (e.g.
ThinkingCap-Qwen3.8-27B-NVFP4), made with HF transformers' own Qwen3.5 decoder layers and the compressed-tensors package's own
dequantisation, streaming one layer at a time (peak memory a few GB, no GPU).

This is the independent reference for the engine's compressed-tensors loader (`quant/compressed_tensors.rs`): the weights are
dequantised by `compressed_tensors`' `unpack_fp4_from_uint8` + `dequantize` (`w = e2m1 * (weight_scale / weight_global_scale)`,
float32, then rounded to the layer dtype), not by anything in this repository. Activations stay 16-bit (NVFP4A16), so there is
no activation QDQ. Modules the checkpoint leaves unquantised (lm_head, norms, `in_proj_a/b`, the vision tower, the MTP head)
are read as stored; the MTP head and the vision tower are not run.

Output: the `intarsia-golden-topk-v1` format of tools/golden.py `--topk K` (default 20), same prompt sets, same metadata keys
(engine `hf-transformers+compressed-tensors`).

    uv run --with compressed-tensors==0.17.1 --with psutil --with transformers --with torch --with safetensors --with numpy \\
        tools/golden_ct_nvfp4.py --model /models/ThinkingCap-Qwen3.8-27B-NVFP4 --out golden/thinkingcap-27b-nvfp4-hfct-top20.safetensors
"""
import argparse
import json
import pathlib
import struct
import sys
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file
from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
from compressed_tensors.quantization.lifecycle.forward import dequantize
from transformers import AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf_dense
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf_moe
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import goldenlib as gl  # noqa: E402

def dequant(packed, scale, gscale):
    """compressed-tensors' own reference: unpack the e2m1 codes and w = code * (scale / global_scale), in float32."""
    q = unpack_fp4_from_uint8(packed, packed.shape[0], packed.shape[1] * 2, dtype=torch.float32)
    return dequantize(x_q=q, scale=scale.to(torch.float32), global_scale=gscale.to(torch.float32), dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", nargs="+",
                    default=[str(HERE / "prompts" / "golden.jsonl"), str(HERE / "prompts" / "golden_long.jsonl")])
    ap.add_argument("--template", default="chatml", choices=["chatml", "hf"])
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"],
                    help="layer / residual precision (float32 only to measure how much bf16 noise the bf16 reference has)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--layers", type=int, default=0, help="debug: run only the first N layers")
    args = ap.parse_args()
    dt = getattr(torch, args.dtype)
    if args.threads:
        torch.set_num_threads(args.threads)
    model = pathlib.Path(args.model)
    cfg = json.load(open(model / "config.json"))
    qc = cfg.get("quantization_config") or {}
    assert qc.get("quant_method") == "compressed-tensors", "not a compressed-tensors checkpoint"
    tc = dict(cfg.get("text_config", cfg))
    tc.pop("quantization_config", None)
    moe = int(tc.get("num_experts") or 0) > 0  # MoE layers keep per-expert tensors; HF wants them fused [E, 2I, H] / [E, H, I]
    hf = hf_moe if moe else hf_dense
    config = (Qwen3_5MoeTextConfig if moe else Qwen3_5TextConfig)(**tc)
    prefix = "Qwen3_5Moe" if moe else "Qwen3_5"
    Layer, Rotary, Norm = (getattr(hf, prefix + n) for n in ("DecoderLayer", "TextRotaryEmbedding", "RMSNorm"))
    config._attn_implementation = "sdpa"

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
    hs = [emb[torch.tensor(ids)].unsqueeze(0).to(dt) for _, ids, _, _ in items]
    del emb
    rot = Rotary(config)
    pe = [rot(h, torch.arange(h.shape[1]).view(1, 1, -1)) for h in hs]
    n_layers = args.layers or config.num_hidden_layers
    nq = 0
    with torch.no_grad():
        for li in range(n_layers):
            layer = Layer(config, li).to(dt)
            pre = f"{P}layers.{li}."
            sd = {}
            for name, _ in layer.named_parameters():
                key = pre + name
                mod = key[:-len(".weight")]
                if name in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"):
                    e = pre + "mlp.experts."
                    one = lambda x, proj: dequant(get(f"{e}{x}.{proj}.weight_packed"), get(f"{e}{x}.{proj}.weight_scale"),
                                                  get(f"{e}{x}.{proj}.weight_global_scale"))
                    if name.endswith("gate_up_proj"):
                        t = torch.stack([torch.cat([one(x, "gate_proj"), one(x, "up_proj")], 0) for x in range(config.num_experts)])
                    else:
                        t = torch.stack([one(x, "down_proj") for x in range(config.num_experts)])
                    t = t.to(dt)
                    nq += 3 * config.num_experts
                elif key not in where and mod + ".weight_packed" in where:  # quantised linear
                    t = dequant(get(mod + ".weight_packed"), get(mod + ".weight_scale"), get(mod + ".weight_global_scale")).to(dt)
                    nq += 1
                else:
                    t = get(key)
                sd[name] = t
            layer.load_state_dict(sd, strict=True)
            layer.eval()
            for i in range(len(hs)):
                hs[i] = layer(hs[i], position_embeddings=pe[i], attention_mask=None, position_ids=None)
            del layer, sd
            if li % 8 == 7 or li == n_layers - 1:
                print(f"layer {li + 1}/{n_layers}  {time.time() - t0:.0f}s", flush=True)
        norm = Norm(config.hidden_size, eps=config.rms_norm_eps).to(dt)
        norm.load_state_dict({"weight": get(P + "norm.weight")})
        head = get("lm_head.weight")
        tensors, meta_prompts = {}, []
        for i, (p, ids, text, split) in enumerate(items):
            logits = (norm(hs[i])[0] @ head.to(dt).T).float()
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
        "engine": "hf-transformers+compressed-tensors",
        "engine_version": f"transformers {transformers.__version__}, torch {torch.__version__}, layer-streamed, "
                          f"compressed-tensors NVFP4 weights dequantised (float32) to {args.dtype}, 16-bit activations, {nq} quantised linears",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": "cpu",
        "dtype": args.dtype,
        "kv_dtype": {"bfloat16": "bf16", "float32": "f32"}[args.dtype],
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
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s, {nq} quantised linears")


if __name__ == "__main__":
    main()
