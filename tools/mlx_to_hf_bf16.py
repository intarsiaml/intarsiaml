"""Dequantise an MLX affine checkpoint (mlx-community/*, mlx_lm.convert -q) into a plain bf16 HF-layout directory.

An independent implementation (numpy) of what crates/intarsia-core/src/quant/mlx_affine.rs does at load, used as the
reference side when no bf16 original of an MLX checkpoint is at hand: HF transformers loads the result, and
tools/golden.py makes the golden from it. Same four steps as the engine:
  1. w = scale * q + bias per group (f32, round to nearest even to bf16), q from the little-endian bit stream of the u32 words
  2. names: language_model.model.* -> model.language_model.*, vision_tower.* -> model.visual.*
  3. conv1d [C, K, 1] -> [C, 1, K]; vision patch_embed [O, T, H, W, C] -> [O, C, T, H, W]
  4. zero-centred norms (Qwen3.5 / Qwen3-Next): mlx_lm bakes the "1 +" into the stored weight; give w - 1 back when
     layer 0's input_layernorm has mean > 0.5

  python tools/mlx_to_hf_bf16.py --src <mlx dir> --out <new dir>
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def bf16_bits(x32):
    u = np.ascontiguousarray(x32, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def f32_of(t):
    """torch tensor (bf16 / f16 / f32) -> float32 numpy."""
    return t.to(torch.float32).numpy()


def dequant(w_u32, scales, biases, group, bits):
    rows = w_u32.shape[0]
    cols = scales.shape[1] * group
    raw = np.ascontiguousarray(w_u32).astype("<u4").view(np.uint8).reshape(rows, -1)
    stream = np.unpackbits(raw, axis=1, bitorder="little")
    q = np.zeros((rows, cols), np.uint32)
    for j in range(bits):
        q |= stream[:, j : cols * bits : bits].astype(np.uint32) << j
    out = q.astype(np.float32) * np.repeat(f32_of(scales), group, axis=1) + np.repeat(f32_of(biases), group, axis=1)
    return torch.from_numpy(bf16_bits(out)).view(torch.bfloat16)


def map_name(n):
    if n.startswith("language_model.model."):
        return "model.language_model." + n[len("language_model.model."):]
    if n.startswith("language_model.lm_head."):
        return "lm_head." + n[len("language_model.lm_head."):]
    if n.startswith("vision_tower."):
        return "model.visual." + n[len("vision_tower."):]
    return n


def shifted_norm(n):
    return n in ("model.norm.weight", "model.language_model.norm.weight") or any(
        n.endswith(s) for s in (".input_layernorm.weight", ".post_attention_layernorm.weight", ".self_attn.q_norm.weight", ".self_attn.k_norm.weight")
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    q = cfg["quantization"]
    assert q.get("mode", "affine") == "affine"
    tensors = {}
    for f in sorted(glob.glob(os.path.join(a.src, "*.safetensors"))):
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                tensors[k] = sf.get_tensor(k)
    out = {}
    nq = 0
    for k, t in tensors.items():
        if k.endswith(".scales") or k.endswith(".biases"):
            continue
        m = k[: -len(".weight")] if k.endswith(".weight") else None
        if m and m + ".scales" in tensors:
            g, b = q.get(m, q)["group_size"], q.get(m, q)["bits"]
            t = dequant(t.numpy(), tensors[m + ".scales"], tensors[m + ".biases"], g, b)
            nq += 1
        n = map_name(k)
        if n.endswith("conv1d.weight") and t.ndim == 3 and t.shape[2] == 1:
            t = t.reshape(t.shape[0], 1, t.shape[1]).contiguous()
        if n.endswith("patch_embed.proj.weight") and t.ndim == 5:
            t = t.permute(0, 4, 1, 2, 3).contiguous()
        out[n] = t
    probe = next((n for n in out if n.endswith("layers.0.input_layernorm.weight")), None)
    tm = (cfg.get("text_config") or cfg).get("model_type", "")
    unshift = (tm.startswith("qwen3_5") or tm.startswith("qwen3_next")) and probe is not None and float(out[probe].float().mean()) > 0.5
    if unshift:
        for n in list(out):
            if shifted_norm(n):
                out[n] = (out[n].float() - 1.0).to(torch.bfloat16)
    os.makedirs(a.out, exist_ok=True)
    save_file({k: v.contiguous() for k, v in out.items()}, os.path.join(a.out, "model.safetensors"), metadata={"format": "pt"})
    for k in ("quantization", "quantization_config"):
        cfg.pop(k, None)
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    for f in os.listdir(a.src):
        if f.endswith((".json", ".jinja", ".txt")) and f not in ("config.json", "model.safetensors.index.json"):
            shutil.copy(os.path.join(a.src, f), os.path.join(a.out, f))
    print(f"{nq} quantised modules dequantised, {len(out)} tensors, norms unshifted: {unshift} -> {a.out}")


main()
