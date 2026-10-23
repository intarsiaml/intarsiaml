#!/usr/bin/env python3
"""Tiny random-weight Gemma4 (text-only) checkpoint + HF reference logits.

Run:
  uv run --no-project --with torch --with transformers==5.6.2 --with safetensors \
      --with numpy python tools/gemma4_tiny.py --out DIR

Writes into DIR:
  config.json                    Gemma4ForConditionalGeneration layout (text_config sub-dict)
  model.safetensors              bf16, real-checkpoint tensor names (per-expert, un-quantised)
  prompts.json                   the 3 fixed token-id prompts (lengths 5, 21, 40)
  ref_logits.safetensors         p0..p2 f32 [T, vocab], HF forward, model in bf16 (post softcap)
  ref_logits_f32.safetensors     same with the model in float32
  ref_greedy.json                per-prompt argmax of every position + greedy next token

Numerics reference = transformers 5.6.2, CPU, eager attention, bf16 weights.
"""
import argparse
import json
import os
import re

import numpy as np
import torch
from safetensors.torch import load_file, save_file

REAL_NAMES = (
    "/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/"
    "scratchpad/g4/model.safetensors.index.json"
)
REAL_CONFIG = (
    "/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/"
    "scratchpad/g4/config.json"
)

NL = 6
TEXT_CFG = dict(
    vocab_size=512,
    vocab_size_per_layer_input=512,
    hidden_size=128,
    intermediate_size=96,
    moe_intermediate_size=48,
    num_hidden_layers=NL,
    num_attention_heads=4,
    num_key_value_heads=2,
    num_global_key_value_heads=1,
    head_dim=32,
    global_head_dim=64,
    attention_k_eq_v=True,
    sliding_window=8,
    layer_types=["sliding_attention", "sliding_attention", "full_attention"] * 2,
    num_experts=8,
    top_k_experts=2,
    enable_moe_block=True,
    hidden_size_per_layer_input=0,
    num_kv_shared_layers=0,
    final_logit_softcapping=30.0,
    hidden_activation="gelu_pytorch_tanh",
    rms_norm_eps=1e-6,
    max_position_embeddings=4096,
    tie_word_embeddings=True,
    rope_parameters={
        "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
    },
    attention_bias=False,
    attention_dropout=0.0,
    use_bidirectional_attention="vision",
    use_double_wide_mlp=False,
    pad_token_id=0,
    bos_token_id=2,
    eos_token_id=1,
)

ROUTER_STD = float(os.environ.get("ROUTER_STD", "2.0"))
QK_NORM = float(os.environ.get("QK_NORM", "0.2"))
EMBED_STD = float(os.environ.get("EMBED_STD", "0.4"))
PROMPT_LENS = [5, 21, 40]


def make_prompts():
    rng = np.random.RandomState(1234)
    return [rng.randint(3, 512, size=n).tolist() for n in PROMPT_LENS]


def randomize(model, seed=0):
    """Random weights with sensible scales (bf16-representable values)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in list(model.named_parameters()) + list(model.named_buffers()):
            if name.endswith("embed_scale") or "inv_freq" in name:
                continue  # non-persistent rotary/embed buffers: keep as built
            if name.endswith("layer_scalar"):
                p.copy_(0.3 + 0.6 * torch.rand(p.shape, generator=g))
            elif name.endswith("per_expert_scale"):
                p.copy_(1.0 + 0.2 * (2 * torch.rand(p.shape, generator=g) - 1))
            elif name.endswith("router.scale"):
                p.copy_(1.0 + 0.3 * (2 * torch.rand(p.shape, generator=g) - 1))
            elif name.endswith(("q_norm.weight", "k_norm.weight")):
                # attention scaling is 1.0 with q/k-norm, so unit norm weights give peaky scores
                # (std ~ sqrt(head_dim)) and bf16 noise gets amplified chaotically; shrink them.
                p.copy_(QK_NORM * (1.0 + 0.1 * torch.randn(p.shape, generator=g)))
            elif name.endswith("norm.weight") or "layernorm" in name or name.endswith("_norm.weight"):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            elif name.endswith("router.proj.weight"):
                # decisive router: router input has rms ~0.09 so score std ~= weight std * 1;
                # post_feedforward_layernorm_2 re-normalises expert output, so a top-k flip from
                # bf16 rounding is an O(1) change -- keep gaps wide so bf16/f32 references agree.
                p.copy_(ROUTER_STD * torch.randn(p.shape, generator=g))
            elif name.endswith("embed_tokens.weight"):
                p.copy_(EMBED_STD * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.05 * torch.randn(p.shape, generator=g))
        # round to bf16 so f32 and bf16 models share identical weights
        for name, p in list(model.named_parameters()) + list(model.named_buffers()):
            if "inv_freq" in name or name.endswith("embed_scale"):
                continue
            p.copy_(p.to(torch.bfloat16).to(p.dtype))


E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
LUT = torch.tensor(E2M1 + [-v for v in E2M1], dtype=torch.float32)  # bit 3 = sign
_MIDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
_TIE_UP = torch.tensor([False, True, False, True, False, True, False])  # ties to even code


def quant_nvfp4(w):
    """ModelOpt-style RTN NVFP4 of f32 [N,K] (see quantize_rows in nvfp4.rs):
    gs = amax/(6*448); per-16 scale = e4m3(block_amax/6/gs); e2m1 nearest, ties to even code.
    Returns (u8 [N,K/2] low nibble = even element, f8e4m3 [N,K/16], gs f32 scalar)."""
    w = w.float()
    n, k = w.shape
    gs = w.abs().max().item() / (6.0 * 448.0)
    blocks = w.reshape(n, k // 16, 16)
    sc = (blocks.abs().amax(-1) / 6.0 / gs).to(torch.float8_e4m3fn)
    eff = (sc.float() * gs).clamp_min(1e-30).unsqueeze(-1)
    x = blocks / eff
    a = x.abs()
    idx = (a.unsqueeze(-1) > _MIDS).sum(-1)  # strictly above midpoint -> next code
    code = idx.clone()
    # exact ties go to the even code (see quantize_rows in nvfp4.rs)
    for i in range(7):
        m = (a == _MIDS[i])
        code = torch.where(m, torch.full_like(code, i + (1 if bool(_TIE_UP[i]) else 0)), code)
    code = code | ((x < 0).long() << 3)
    code = code.reshape(n, k)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8)
    return packed, sc, torch.tensor(gs, dtype=torch.float32)


def dequant_nvfp4(packed, sc, gs):
    n, kh = packed.shape
    lo, hi = (packed & 0xF).long(), (packed >> 4).long()
    vals = LUT[torch.stack([lo, hi], -1).reshape(n, kh * 2)]
    return (vals * sc.float().repeat_interleave(16, 1) * float(gs)).to(torch.bfloat16)


def quantize_experts(model):
    """Replace expert weights in-place by their NVFP4 dequantisation (bf16-representable);
    return {(layer, expert, proj): (packed, scale, gs)} for export."""
    q = {}
    with torch.no_grad():
        for L, layer in enumerate(model.model.layers):
            ex = layer.experts
            I = ex.down_proj.shape[2]
            for e in range(ex.down_proj.shape[0]):
                for proj, w in (("gate_proj", ex.gate_up_proj[e, :I]), ("up_proj", ex.gate_up_proj[e, I:]),
                                ("down_proj", ex.down_proj[e])):
                    pk, sc, gs = quant_nvfp4(w)
                    q[(L, e, proj)] = (pk, sc, gs)
                    dq = dequant_nvfp4(pk, sc, gs).to(w.dtype)
                    w.copy_(dq)
    return q


def to_bf16(model):
    """Cast params/buffers to bf16 but keep rotary inv_freq (and embed_scale) in f32.

    A plain model.to(bfloat16) would round inv_freq to bf16, which is not what
    from_pretrained(dtype=bf16) does (inv_freq is built in float32)."""
    for n, p in model.named_parameters():
        p.data = p.data.to(torch.bfloat16)
    for mod in model.modules():
        for n, b in list(mod._buffers.items()):
            if b is None or "inv_freq" in n or n == "embed_scale":
                continue
            mod._buffers[n] = b.to(torch.bfloat16)
    return model


def to_ckpt(sd, cfg, q):
    """HF state dict -> real-checkpoint tensor names (bf16, per-expert split)."""
    out = {}
    inter = cfg.moe_intermediate_size
    for k, v in sd.items():
        if k == "lm_head.weight":
            continue  # tied
        assert k.startswith("model."), k
        nk = "model.language_model." + k[len("model."):]
        v = v.detach().to(torch.bfloat16).contiguous() if not re.search(r'experts\.(gate_up_proj|down_proj)$', nk) else v.detach()
        m = re.match(r"(.*\.layers\.\d+)\.experts\.(gate_up_proj|down_proj)$", nk)
        if m:
            pre, which = m.groups()
            L = int(re.search(r"layers\.(\d+)", pre).group(1))
            for e in range(v.shape[0]):
                for proj in (("gate_proj", "up_proj") if which == "gate_up_proj" else ("down_proj",)):
                    pk, sc, gs = q[(L, e, proj)]
                    b = f"{pre}.experts.{e}.{proj}"
                    out[b + ".weight"] = pk.contiguous()
                    out[b + ".weight_scale"] = sc.contiguous()
                    out[b + ".weight_scale_2"] = gs.reshape(()).contiguous()
                    out[b + ".input_scale"] = torch.tensor(1.0, dtype=torch.float32)
        else:
            out[nk] = v
    return out


def from_ckpt(ck, cfg):
    """Inverse of to_ckpt (for the round-trip check)."""
    sd = {}
    buckets = {}
    for k, v in ck.items():
        assert k.startswith("model.language_model."), k
        nk = "model." + k[len("model.language_model."):]
        m = re.match(r"(.*\.layers\.\d+)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$", nk)
        if m:
            sc = ck[k.replace(".weight", ".weight_scale")]
            gs = ck[k.replace(".weight", ".weight_scale_2")]
            buckets.setdefault(m.group(1), {})[(int(m.group(2)), m.group(3))] = dequant_nvfp4(v, sc, gs)
        elif re.search(r"\.experts\.\d+\..*(weight_scale|weight_scale_2|input_scale)$", nk):
            continue
        else:
            sd[nk] = v
    for pre, d in buckets.items():
        E = cfg.num_experts
        sd[f"{pre}.experts.gate_up_proj"] = torch.stack(
            [torch.cat([d[(e, "gate_proj")], d[(e, "up_proj")]], 0) for e in range(E)])
        sd[f"{pre}.experts.down_proj"] = torch.stack([d[(e, "down_proj")] for e in range(E)])
    return sd


def real_patterns(names):
    """Layer-0 / layer-5 names of the real checkpoint, normalised to patterns."""
    pats = set()
    for n in names:
        if n.startswith("model.vision_tower") or n.startswith("model.embed_vision") or n.startswith("model.audio"):
            continue
        n = re.sub(r"\.layers\.\d+\.", ".layers.N.", n)
        n = re.sub(r"\.experts\.\d+\.", ".experts.E.", n)
        pats.add(n)
    return pats


def forward_logits(model, prompts):
    outs = []
    with torch.no_grad():
        for p in prompts:
            ids = torch.tensor([p])
            lg = model(input_ids=ids, use_cache=False).logits[0]
            outs.append(lg.float().contiguous())
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import transformers
    from transformers import Gemma4TextConfig, Gemma4ForCausalLM
    print("transformers", transformers.__version__, "torch", torch.__version__)

    cfg = Gemma4TextConfig(**TEXT_CFG)
    cfg._attn_implementation = "eager"
    print("layer_types:", cfg.layer_types)
    assert cfg.layer_types == TEXT_CFG["layer_types"]

    torch.manual_seed(0)
    model = Gemma4ForCausalLM(cfg)
    randomize(model)
    qexp = quantize_experts(model)  # experts = NVFP4 dequant (W4A16 semantics) for HF and engine alike
    model = to_bf16(model).eval()

    for i, l in enumerate(model.model.layers):
        a = l.self_attn
        print(f"layer {i}: {'sliding' if a.is_sliding else 'FULL   '} head_dim={a.head_dim} "
              f"kv_heads={a.k_proj.out_features // a.head_dim} v_proj={'yes' if a.v_proj is not None else 'NONE (k==v)'} "
              f"window={a.sliding_window}")

    # ---- checkpoint
    ck = to_ckpt(model.state_dict(), cfg, qexp)
    save_file(ck, os.path.join(args.out, "model.safetensors"), metadata={"format": "pt"})

    # name-pattern check against the real checkpoint
    real = json.load(open(REAL_NAMES))["weight_map"]
    rp = real_patterns(real.keys())
    tp = {re.sub(r"\.experts\.\d+\.", ".experts.E.", re.sub(r"\.layers\.\d+\.", ".layers.N.", n)) for n in ck}
    # real layer-N patterns include v_proj (sliding layers); global layers lack it. Compare as sets.
    assert tp == rp, f"name mismatch\n only tiny: {sorted(tp - rp)}\n only real: {sorted(rp - tp)}"
    assert not any("vision" in n or "audio" in n for n in ck)
    # global layers have no v_proj; sliding layers have
    for i, t in enumerate(cfg.layer_types):
        has_v = f"model.language_model.layers.{i}.self_attn.v_proj.weight" in ck
        assert has_v == (t == "sliding_attention"), (i, t, has_v)
    print(f"NAME CHECK OK: {len(ck)} tensors, {len(tp)} patterns == real (vision excluded)")

    # config.json: real top-level layout, tiny text_config
    real_cfg = json.load(open(REAL_CONFIG))
    tc = dict(real_cfg["text_config"])
    tc.update({k: v for k, v in TEXT_CFG.items()})
    tc["model_type"] = "gemma4_text"
    tc["dtype"] = "bfloat16"
    real_cfg["text_config"] = tc
    real_cfg.pop("quantization_config", None)
    real_cfg["architectures"] = ["Gemma4ForConditionalGeneration"]
    json.dump(real_cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)

    # ---- round trip: reload checkpoint into a fresh model
    ck2 = load_file(os.path.join(args.out, "model.safetensors"))
    m2 = Gemma4ForCausalLM(cfg)
    sd2 = from_ckpt(ck2, cfg)
    sd2["lm_head.weight"] = sd2["model.embed_tokens.weight"]
    missing, unexpected = m2.load_state_dict(sd2, strict=False)
    assert not unexpected, unexpected
    assert all(k.endswith("embed_scale") for k in missing), missing
    m2 = to_bf16(m2).eval()

    # ---- mask check: capture the sliding mask given to layer 0 and the full mask on layer 2
    prompts = make_prompts()
    json.dump(prompts, open(os.path.join(args.out, "prompts.json"), "w"))
    caps = {}

    def mk(i):
        def hook(mod, a, kw):
            caps[i] = kw["attention_mask"]
        return hook
    h0 = model.model.layers[0].self_attn.register_forward_pre_hook(mk(0), with_kwargs=True)
    h2 = model.model.layers[2].self_attn.register_forward_pre_hook(mk(2), with_kwargs=True)
    with torch.no_grad():
        model(input_ids=torch.tensor([prompts[1]]), use_cache=False)
    h0.remove(); h2.remove()
    m0 = (caps[0][0, 0] == 0)  # allowed
    q = 15
    allowed = [j for j in range(21) if bool(m0[q, j])]
    print(f"sliding mask (window=8) q={q}: attends kv {allowed}  (inclusive of self: {q in allowed}; "
          f"count={len(allowed)})")
    assert allowed == list(range(q - 7, q + 1)), allowed
    m2f = (caps[2][0, 0] == 0) if caps[2] is not None else None
    if m2f is not None:
        print("full mask q=15 attends kv 0..", max(j for j in range(21) if bool(m2f[q, j])))

    # ---- reference logits
    ref_bf = forward_logits(model, prompts)
    ref_bf_rt = forward_logits(m2, prompts)
    sa, sb = model.state_dict(), m2.state_dict()
    for k in sa:
        if k in sb and not torch.equal(sa[k].float(), sb[k].float()):
            print("STATE DIFF", k, sa[k].dtype, sb[k].dtype)
    for k in set(sb) - set(sa):
        print("only m2", k)
    for a, b in zip(ref_bf, ref_bf_rt):
        assert torch.equal(a, b), "checkpoint round-trip changed logits"
    print("ROUND-TRIP OK (reload from exported safetensors gives bit-identical bf16 logits)")
    save_file({f"p{i}": t for i, t in enumerate(ref_bf)}, os.path.join(args.out, "ref_logits.safetensors"))

    model32 = model.float().eval()
    for mod in model32.modules():
        for n, b in list(mod._buffers.items()):
            if b is not None and b.is_floating_point():
                mod._buffers[n] = b.float()
    ref_32 = forward_logits(model32, prompts)
    save_file({f"p{i}": t for i, t in enumerate(ref_32)}, os.path.join(args.out, "ref_logits_f32.safetensors"))

    greedy = {}
    for i, (p, lb, lf) in enumerate(zip(prompts, ref_bf, ref_32)):
        am = lb.argmax(-1).tolist()
        amf = lf.argmax(-1).tolist()
        greedy[f"p{i}"] = {
            "prompt_len": len(p),
            "next_token_bf16": am[-1],
            "next_token_f32": amf[-1],
            "argmax_all_bf16": am,
            "argmax_all_f32": amf,
        }
        agree = float(np.mean(np.array(am) == np.array(amf)))
        d = (lb - lf).abs().max().item()
        print(f"p{i}: T={len(p)} logits {tuple(lb.shape)} next(bf16)={am[-1]} next(f32)={amf[-1]} "
              f"argmax-agree={agree:.2f} max|bf16-f32|={d:.3f} logit-std={lb.std():.2f} max|logit|={lb.abs().max():.2f}")
    # greedy continuation: HF generate (do_sample=False, KV cache) on the bf16 model
    model_bf = to_bf16(Gemma4ForCausalLM(cfg)).eval()
    model_bf.load_state_dict(m2.state_dict())
    cont = []
    for p in prompts:
        with torch.no_grad():
            gen = model_bf.generate(torch.tensor([p]), attention_mask=torch.ones(1, len(p), dtype=torch.long),
                                    max_new_tokens=8, min_new_tokens=8, do_sample=False, pad_token_id=0)
        cont.append(gen[0, len(p):].tolist())
    out = {
        "prompts": prompts,
        "greedy_next": [g["next_token_bf16"] for g in greedy.values()],
        "greedy_continuation": cont,
    }
    for i, c in enumerate(cont):
        assert c[0] == out["greedy_next"][i], (i, c, out["greedy_next"][i])  # cache path == full forward
    out.update(greedy)  # extra diagnostics (per-position argmax, f32 next token)
    json.dump(out, open(os.path.join(args.out, "ref_greedy.json"), "w"), indent=1)
    print("greedy_continuation:", cont)

    # does the window actually matter? compare against a model with a huge window
    cfg_big = Gemma4TextConfig(**{**TEXT_CFG, "sliding_window": 4096})
    cfg_big._attn_implementation = "eager"
    mb = Gemma4ForCausalLM(cfg_big)
    mb.load_state_dict(model32.state_dict(), strict=False)
    mb = mb.float().eval()
    lbig = forward_logits(mb, prompts)
    for i in range(3):
        print(f"p{i}: window matters: max|win8 - win4096| = {(ref_32[i] - lbig[i]).abs().max().item():.4f}")

    sz = sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out))
    print(f"wrote {args.out} ({sz/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
