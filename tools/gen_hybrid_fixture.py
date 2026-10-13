"""Generate the GatedDeltaNet / gated-attention test fixtures from the HF reference implementation.

Run with the tools venv (transformers >= 5.19, torch fallbacks, no fla / causal_conv1d):

    tools/.venv/bin/python tools/gen_hybrid_fixture.py

Writes (all fp32, < 5 MB together):

  crates/intarsia-core/tests/fixtures/gdn_hf.json
      op-level cases: causal conv + SiLU, and the gated delta rule (HF `torch_chunk_gated_delta_rule` and
      `torch_recurrent_gated_delta_rule`, which must agree), with a non-zero initial state and several chunkings.
  crates/intarsia-models/tests/fixtures/tiny_qwen3_5/{config.json,model.safetensors,golden.safetensors}
      a random tiny `Qwen3_5ForCausalLM` (3 GDN layers + full-attention layers interleaved, partial rotary 0.25,
      gated attention, zero-centred norms, dense FFN) and its logits / per-layer hidden states for a fixed prompt.

Every norm weight, A_log and dt_bias is re-randomised: HF initialises the zero-centred norms to 0 and dt_bias to 1,
which would hide bugs in the (1 + w) handling and the gate math.
"""
import json
import pathlib

import torch
from safetensors.torch import save_file
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as M

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORE_FIX = ROOT / "crates/intarsia-core/tests/fixtures"
MODEL_FIX = ROOT / "crates/intarsia-models/tests/fixtures/tiny_qwen3_5"

torch.manual_seed(1234)


def flat(t):
    return {"shape": list(t.shape), "data": [float(x) for x in t.detach().double().flatten()]}


def op_fixture():
    """Conv + delta-rule cases straight from the HF functions."""
    cases = []
    for (hk, hv, dk, dv, ks, t) in [(2, 4, 8, 8, 4, 11), (1, 3, 16, 8, 3, 9), (2, 2, 8, 16, 4, 70)]:
        c = 2 * hk * dk + hv * dv
        x = torch.randn(1, t, c)
        w = torch.randn(c, ks) * 0.5
        # --- conv: HF causal_conv1d_fn works on [B, C, T] with zero left padding
        prev = torch.randn(1, c, ks - 1)  # a non-zero carried state: prepend it as real history
        y_zero = M.causal_conv1d_fn(x.transpose(1, 2), w, None, activation="silu").transpose(1, 2)
        full = torch.cat([prev, x.transpose(1, 2)], dim=-1)
        y_carry = M.causal_conv1d_fn(full, w, None, activation="silu")[:, :, ks - 1:].transpose(1, 2)
        new_state = full[:, :, -(ks - 1):]
        # --- delta rule inputs as the block builds them
        a = torch.randn(1, t, hv)
        b = torch.randn(1, t, hv)
        a_log = torch.randn(hv) * 0.5 - 1.0
        dt_bias = torch.randn(hv) * 0.5
        conv_out = torch.nn.functional.silu(torch.randn(1, t, c))  # stands in for the conv output
        q, k, v = torch.split(conv_out, [hk * dk, hk * dk, hv * dv], dim=-1)
        q = q.reshape(1, t, hk, dk)
        k = k.reshape(1, t, hk, dk)
        v = v.reshape(1, t, hv, dv)
        beta = b.sigmoid()
        g = -a_log.float().exp() * torch.nn.functional.softplus(a.float() + dt_bias)
        rep = hv // hk
        qr, kr = q.repeat_interleave(rep, dim=2), k.repeat_interleave(rep, dim=2)
        s0 = torch.randn(1, hv, dk, dv) * 0.1
        o_chunk, s_chunk = M.torch_chunk_gated_delta_rule(
            qr, kr, v, g=g, beta=beta, initial_state=s0, output_final_state=True, use_qk_l2norm_in_kernel=True
        )
        o_rec, s_rec = M.torch_recurrent_gated_delta_rule(
            qr, kr, v, g=g, beta=beta, initial_state=s0, output_final_state=True, use_qk_l2norm_in_kernel=True
        )
        assert torch.allclose(o_chunk, o_rec, atol=1e-4, rtol=1e-4), (o_chunk - o_rec).abs().max()
        assert torch.allclose(s_chunk, s_rec, atol=1e-4, rtol=1e-4)
        cases.append(
            {
                "hk": hk, "hv": hv, "dk": dk, "dv": dv, "ks": ks, "t": t,
                "conv_x": flat(x[0]), "conv_w": flat(w), "conv_prev": flat(prev[0]),  # prev: [C, K-1]
                "conv_y_zero_state": flat(y_zero[0]), "conv_y_carry": flat(y_carry[0]), "conv_new_state": flat(new_state[0]),
                "qkv": flat(conv_out[0]), "a": flat(a[0]), "b": flat(b[0]), "a_log": flat(a_log), "dt_bias": flat(dt_bias),
                # HF state is [Hv, dk, dv]; the engine stores it transposed, [Hv, dv, dk]
                "state0": flat(s0[0].transpose(1, 2)),
                "out": flat(o_rec[0]),  # [T, Hv, dv]
                "state1": flat(s_rec[0].transpose(1, 2)),
            }
        )
    CORE_FIX.mkdir(parents=True, exist_ok=True)
    (CORE_FIX / "gdn_hf.json").write_text(json.dumps({"cases": cases}))
    print("gdn_hf.json", (CORE_FIX / "gdn_hf.json").stat().st_size, "bytes")


def model_fixture():
    cfg = Qwen3_5TextConfig(
        vocab_size=96, hidden_size=64, intermediate_size=96, num_hidden_layers=6,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_conv_kernel_dim=4, linear_key_head_dim=32, linear_value_head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4, max_position_embeddings=256,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention",
                     "linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25},
        tie_word_embeddings=False, rms_norm_eps=1e-6, dtype="float32",
    )
    model = Qwen3_5ForCausalLM(cfg).float().eval()
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("layernorm.weight") or name.endswith("q_norm.weight") or name.endswith("k_norm.weight") \
                    or name == "model.norm.weight":
                p.copy_(torch.randn(p.shape, generator=g) * 0.2)  # zero-centred: scale = 1 + w
            elif name.endswith("linear_attn.norm.weight"):
                p.copy_(1.0 + torch.randn(p.shape, generator=g) * 0.2)  # plain gated-norm weight
            elif name.endswith("A_log"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.7 - 1.5)
            elif name.endswith("dt_bias"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.5 - 0.3)
            elif "conv1d" in name:
                p.copy_(torch.randn(p.shape, generator=g) * 0.4)
            elif name.endswith("embed_tokens.weight"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.8)
            else:
                p.copy_(torch.randn(p.shape, generator=g) * 0.18)
    ids = torch.randint(0, 96, (1, 26), generator=g)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, use_cache=False)
        # cached decode must agree with the full pass (sanity check of the fixture itself)
        part = model(ids[:, :17], use_cache=True)
        pkv = part.past_key_values
        steps = []
        for i in range(17, 26):
            o = model(ids[:, i:i + 1], past_key_values=pkv, use_cache=True)
            pkv = o.past_key_values
            steps.append(o.logits[0, 0])
        dec = torch.stack(steps)
        err = (dec - out.logits[0, 17:]).abs().max().item()
        print("cached decode vs full pass max abs err:", err)
        assert err < 1e-3, err
    MODEL_FIX.mkdir(parents=True, exist_ok=True)
    sd = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(sd, str(MODEL_FIX / "model.safetensors"))
    d = cfg.to_dict()
    d["architectures"] = ["Qwen3_5ForCausalLM"]
    d["model_type"] = "qwen3_5_text"
    # keep the config small and explicit; the loader must not depend on HF defaults it cannot see
    d = {k: v for k, v in d.items() if k not in ("id2label", "label2id", "output_attentions", "return_dict")}
    (MODEL_FIX / "config.json").write_text(json.dumps(d, indent=1, default=str))
    gold = {"ids": ids[0].to(torch.int32).contiguous(), "logits": out.logits[0].contiguous()}
    for i, h in enumerate(out.hidden_states):
        gold[f"hidden_{i}"] = h[0].contiguous()
    save_file(gold, str(MODEL_FIX / "golden.safetensors"))
    sz = sum(p.stat().st_size for p in MODEL_FIX.iterdir())
    print("tiny model fixture", sz, "bytes; logits absmax", out.logits.abs().max().item())


if __name__ == "__main__":
    op_fixture()
    model_fixture()
