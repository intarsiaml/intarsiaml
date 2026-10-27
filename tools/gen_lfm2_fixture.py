#!/usr/bin/env python3
"""Tiny LFM2 (gated short conv + GQA with qk-norm) fixture from the Hugging Face reference (lane nab2).

tools venv (transformers >= 5.19, torch fallbacks, no causal_conv1d):  tools/.venv/bin/python tools/gen_lfm2_fixture.py

Writes crates/intarsia-models/tests/fixtures/tiny_lfm2/{config.json,model.safetensors,golden.safetensors}: a random tiny
`Lfm2ForCausalLM` (6 layers: conv, conv, full, conv, full, conv; fp32; tied embeddings; FFN width through
`block_auto_adjust_ff_dim`: int(2 * 96 / 3) = 64) with the logits HF produced for a fixed 26-token prompt. Weights are
re-randomised so that the norms, the conv taps and the three chunks of `in_proj` all matter (HF initialises norms to 1).
"""
import json
import pathlib

import torch
from safetensors.torch import save_file
from transformers import Lfm2Config, Lfm2ForCausalLM

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIX = ROOT / "crates/intarsia-models/tests/fixtures/tiny_lfm2"
torch.manual_seed(1234)


def main():
    cfg = Lfm2Config(
        vocab_size=96, hidden_size=32, intermediate_size=96, num_hidden_layers=6, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=256, norm_eps=1e-5, conv_bias=False, conv_L_cache=3,
        block_multiple_of=16, block_ffn_dim_multiplier=1.0, block_auto_adjust_ff_dim=True,
        layer_types=["conv", "conv", "full_attention", "conv", "full_attention", "conv"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0}, tie_word_embeddings=True, dtype="float32",
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    model = Lfm2ForCausalLM(cfg).float().eval()
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
                p.copy_(1.0 + torch.randn(p.shape, generator=g) * 0.2)  # plain RMSNorm weight
            elif "conv.conv" in name:
                p.copy_(torch.randn(p.shape, generator=g) * 0.6)
            elif name.endswith("embed_tokens.weight"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.8)
            else:
                p.copy_(torch.randn(p.shape, generator=g) * 0.2)
    ids = torch.randint(0, 96, (1, 26), generator=g)
    with torch.no_grad():
        out = model(ids, use_cache=False)
        # cached decode must agree with the full pass (sanity check of the fixture itself)
        part = model(ids[:, :17], use_cache=True)
        pkv = part.past_key_values
        steps = []
        for i in range(17, 26):
            o = model(ids[:, i:i + 1], past_key_values=pkv, use_cache=True)
            pkv = o.past_key_values
            steps.append(o.logits[0, 0])
        err = (torch.stack(steps) - out.logits[0, 17:]).abs().max().item()
        print("cached decode vs full pass max abs err:", err)
        assert err < 1e-3, err
    FIX.mkdir(parents=True, exist_ok=True)
    sd = {k: v.contiguous() for k, v in model.state_dict().items() if k != "lm_head.weight"}  # tied: no lm_head tensor
    save_file(sd, str(FIX / "model.safetensors"))
    d = cfg.to_dict()
    d["architectures"] = ["Lfm2ForCausalLM"]
    d["model_type"] = "lfm2"
    # the real LFM2.5 config spells these its own way; mimic it so the recipe's [config] renames are exercised
    d["tie_embedding"] = d.pop("tie_word_embeddings")
    d["rope_theta"] = d["rope_parameters"]["rope_theta"]
    d.pop("rope_parameters", None)
    d = {k: v for k, v in d.items() if k not in ("id2label", "label2id", "output_attentions", "return_dict")}
    (FIX / "config.json").write_text(json.dumps(d, indent=1, default=str))
    save_file({"ids": ids[0].to(torch.int32).contiguous(), "logits": out.logits[0].contiguous()}, str(FIX / "golden.safetensors"))
    print("tiny LFM2 fixture", sum(p.stat().st_size for p in FIX.iterdir()), "bytes; logits absmax", out.logits.abs().max().item())
    print(sorted(sd)[:12])


if __name__ == "__main__":
    main()
