"""Expected values for the quantised-weight / MoE unit tests, computed independently of the Rust code.

Input : crates/intarsia-backend-cpu/tests/fixtures/qwen36_quant_slices.safetensors (real tensor slices of
        nvidia/Qwen3.6-35B-A3B-NVFP4, made by tools/extract_quant_slices.py)
Output: crates/intarsia-backend-cpu/tests/fixtures/qwen36_quant_expected.safetensors
Uses torch's float8_e4m3fn for the e4m3 scales, the e2m1 value list for the packed weights and the HF
transformers Qwen3_5MoeSparseMoeBlock (the spec) for the MoE block. Run with tools/.venv/bin/python.
"""
import numpy as np, torch
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as m
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

fx = Path(__file__).resolve().parent.parent / "crates/intarsia-backend-cpu/tests/fixtures"
f = safe_open(str(fx / "qwen36_quant_slices.safetensors"), "pt")
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float64)
out = {}
g = torch.Generator().manual_seed(7)


def nv_dequant(prefix):
    w = f.get_tensor(prefix + ".weight")
    s = f.get_tensor(prefix + ".weight_scale").to(torch.float64)  # float8_e4m3fn -> exact
    gs = float(f.get_tensor(prefix + ".weight_scale_2"))
    lo, hi = (w & 0xF).long(), (w >> 4).long()
    codes = torch.stack([lo, hi], -1).reshape(w.shape[0], -1)  # element 2i = low nibble
    v = E2M1[codes & 7] * torch.where(codes & 8 > 0, -1.0, 1.0)
    return v * s.repeat_interleave(16, dim=1) * gs


def bf16r(x):
    return x.to(torch.bfloat16).to(torch.float32)


for key in ("nv_gate", "nv_down", "nv_head"):
    W = nv_dequant(key)
    k = W.shape[1]
    x = bf16r(torch.randn(3, k, generator=g))
    out[key + ".x"] = x
    out[key + ".y"] = (x.double() @ W.T).float()
    out[key + ".W_rows0"] = W[:4].float().contiguous()  # first rows of the dequantised weight, for dequant tests

for key in ("fp8_z", "fp8_o"):
    w = f.get_tensor(key + ".weight").to(torch.float64)
    ws = float(f.get_tensor(key + ".weight_scale"))
    is_ = float(f.get_tensor(key + ".input_scale"))
    k = w.shape[1]
    x = bf16r(torch.randn(3, k, generator=g))
    out[key + ".x"] = x
    out[key + ".y_w8a16"] = (x.double() @ (w * ws).T).float()
    xq = (x / is_).clamp(-448, 448).to(torch.float8_e4m3fn).to(torch.float64) * is_
    out[key + ".y_w8a8"] = (xq @ (w * ws).T).float()

# --- the MoE block, HF module
E, H, I, K = 8, 256, 64, 2
cfg = Qwen3_5MoeTextConfig(hidden_size=H, moe_intermediate_size=I, shared_expert_intermediate_size=I, num_experts=E,
                           num_experts_per_tok=K, hidden_act="silu", num_hidden_layers=1, num_attention_heads=4,
                           num_key_value_heads=2, head_dim=64, vocab_size=100)
blk = m.Qwen3_5MoeSparseMoeBlock(cfg).eval()
with torch.no_grad():
    blk.gate.weight.copy_(f.get_tensor("moe.router").float())
    blk.shared_expert_gate.weight.copy_(f.get_tensor("moe.shared_gate").float())
    for proj in ("gate_proj", "up_proj", "down_proj"):
        getattr(blk.shared_expert, proj).weight.copy_(nv_dequant(f"moe.shared.{proj}").float())
    for e in range(E):
        gate, up, down = (nv_dequant(f"moe.e{e}.{p}").float() for p in ("gate_proj", "up_proj", "down_proj"))
        blk.experts.gate_up_proj[e].copy_(torch.cat([gate, up], 0))
        blk.experts.down_proj[e].copy_(down)
T = 5
x = bf16r(torch.randn(T, H, generator=g) * 2.0)
with torch.no_grad():
    logits, scores, idx = blk.gate(x)
    out["moe.x"] = x
    out["moe.logits"] = logits
    out["moe.topk_w"] = scores.float()
    out["moe.topk_id"] = idx.to(torch.int32)
    out["moe.y_f32"] = blk(x[None])[0]
    bb = blk.to(torch.bfloat16)
    xb = x.to(torch.bfloat16)
    l2, s2, i2 = bb.gate(xb)
    out["moe.logits_bf16"] = l2.float()
    out["moe.topk_w_bf16"] = s2.float()
    out["moe.topk_id_bf16"] = i2.to(torch.int32)
    out["moe.y_bf16"] = bb(xb[None])[0].float()
save_file({k: v.contiguous() for k, v in out.items()}, str(fx / "qwen36_quant_expected.safetensors"))
print({k: tuple(v.shape) for k, v in out.items()})
