#!/usr/bin/env python3
"""Lane dfh: a tiny DFlash2 parity fixture from z-lab's own reference (dflash/model_mlx.py, MIT, vendored unmodified in
TensorFold at src/tensorfold/drafters/vendor/z_lab_dflash/). Run with TensorFold's venv (mlx 0.32):

    /opt/code/TensorFold/.venv/bin/python -I tools/dfh/dflash2_mlx_fixture.py <out dir>

Writes <out dir>/config.json + model.safetensors (a random 2-layer DFlash2 draft, bf16, in the checkpoint layout) and
<out dir>/golden.safetensors: context features `feat` [C, taps*H] f32 for C = 3 and 13, the block ids, and the reference's
final normed hidden rows of every block row (`hidden_states(..., logits_start=0)`), computed in f32 from the bf16 weights.
The draft binds the target embedding of tests/fixtures/tiny_qwen3_5 (`model.embed_tokens.weight`). Nothing is copied from
the reference: it is imported and run."""
import json, sys
from pathlib import Path

REF = "/opt/code/TensorFold/src/tensorfold/drafters/vendor"
sys.path.insert(0, REF)
import numpy as np
import mlx.core as mx
# the CPU device: MLX on the Apple GPU differs from its own CPU result by ~2e-3 rel L2 in f32 (lane dfh), the CPU matches an f64
# host implementation and the engine to ~1e-6
mx.set_default_device(mx.cpu)
from mlx import nn
from z_lab_dflash import model_mlx as zl

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
fixture = Path(__file__).resolve().parents[2] / "crates/intarsia-models/tests/fixtures/tiny_qwen3_5"
H, V, NH, NKV, HD, DI, L, K, GS, BLOCK, W, MASK = 64, 96, 4, 2, 16, 48, 2, 2, 16, 4, 6, 95
TAPS = [1, 3]
THETA = 1_000_000.0
cfg = {
    "architectures": ["DFlash2DraftModel"], "is_causal": False, "hidden_size": H, "num_hidden_layers": L,
    "num_attention_heads": NH, "num_key_value_heads": NKV, "head_dim": HD, "intermediate_size": DI, "vocab_size": V,
    "rms_norm_eps": 1e-6, "rope_parameters": {"rope_theta": THETA, "rope_type": "default"}, "max_position_embeddings": 4096,
    "layer_types": ["sliding_attention"] * L, "sliding_window": W, "use_sliding_window": True, "num_target_layers": 6,
    "dflash_config": {"block_size": BLOCK, "conv_group_size": GS, "conv_kernel_size": K, "mask_token_id": MASK,
                      "selector_rank": 8, "selector_top_k": 4, "target_layer_ids": TAPS},
    "tie_word_embeddings": False, "hidden_act": "silu",
}
(out / "config.json").write_text(json.dumps(cfg, indent=1))
rng = np.random.default_rng(7)
G = H // GS
w = {}
def put(name, shape, kind):
    if kind == "norm":
        a = 1.0 + 0.1 * rng.standard_normal(shape)
    elif kind == "base":
        a = 0.3 * rng.standard_normal(shape); a[:, 0, :] += 1.0
    else:
        a = kind * rng.standard_normal(shape)
    w[name] = mx.array(a.astype(np.float32)).astype(mx.bfloat16)
for l in range(L):
    p = f"layers.{l}"
    put(f"{p}.input_layernorm.weight", (H,), "norm"); put(f"{p}.post_attention_layernorm.weight", (H,), "norm")
    put(f"{p}.self_attn.q_proj.weight", (NH * HD, H), 0.15); put(f"{p}.self_attn.k_proj.weight", (NKV * HD, H), 0.15)
    put(f"{p}.self_attn.v_proj.weight", (NKV * HD, H), 0.15); put(f"{p}.self_attn.o_proj.weight", (H, NH * HD), 0.15)
    put(f"{p}.self_attn.q_norm.weight", (HD,), "norm"); put(f"{p}.self_attn.k_norm.weight", (HD,), "norm")
    put(f"{p}.mlp.gate_proj.weight", (DI, H), 0.15); put(f"{p}.mlp.up_proj.weight", (DI, H), 0.15)
    put(f"{p}.mlp.down_proj.weight", (H, DI), 0.15)
    for c in ("attention_conv", "mlp_conv"):
        put(f"{p}.{c}.base_kernel", (2, K, H), "base"); put(f"{p}.{c}.kernel_projection.weight", (2 * K * G, H), 0.1)
put("fc.weight", (H, len(TAPS) * H), 0.1); put("hidden_norm.weight", (H,), "norm"); put("norm.weight", (H,), "norm")
put("candidate_selector.hidden_projection.weight", (8, H), 0.1)
put("candidate_selector.predecessor_codebook", (V, 8), 0.1); put("candidate_selector.successor_codebook", (V, 8), 0.1)
mx.save_safetensors(str(out / "model.safetensors"), w)

# the reference model, its parameters cast to f32 (the engine computes in f32 from the same bf16-rounded weights)
c = zl.load_draft.__globals__["DFlashConfig"](
    hidden_size=H, num_hidden_layers=L, num_attention_heads=NH, num_key_value_heads=NKV, head_dim=HD, intermediate_size=DI,
    vocab_size=V, rms_norm_eps=1e-6, rope_theta=THETA, max_position_embeddings=4096, block_size=BLOCK,
    target_layer_ids=tuple(TAPS), num_target_layers=6, mask_token_id=MASK, rope_scaling=cfg["rope_parameters"],
    layer_types=tuple(cfg["layer_types"]), sliding_window=W, conv_kernel_size=K, conv_group_size=GS, selector_rank=8,
    selector_top_k=4, is_causal=False)
model = zl.DFlash2DraftModel(c)
ww = {k: v.astype(mx.float32) for k, v in w.items()}
for name in ("predecessor_codebook", "successor_codebook"):
    ww[f"candidate_selector.{name}.weight"] = ww.pop(f"candidate_selector.{name}")
model.load_weights(list(ww.items()))
emb = mx.load(str(fixture / "model.safetensors"))["model.embed_tokens.weight"].astype(mx.float32)
model.embed_tokens = nn.Embedding(V, H); model.embed_tokens.weight = emb
model.embed_scale = 1.0
gold = {}
for C in (3, 13):
    feat = mx.array(np.random.default_rng(C).standard_normal((1, C, len(TAPS) * H)).astype(np.float32))
    ids = mx.array([[17] + [MASK] * (BLOCK - 1)])
    cache = model.make_cache()
    hid = model.hidden_states(ids, feat, cache, 0)
    mx.eval(hid)
    gold[f"feat_{C}"] = feat[0]; gold[f"hidden_{C}"] = hid[0]
gold["ids"] = mx.array([17] + [MASK] * (BLOCK - 1), dtype=mx.int32)
mx.save_safetensors(str(out / "golden.safetensors"), gold)
print("wrote", out, {k: v.shape for k, v in gold.items()})
