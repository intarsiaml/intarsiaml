#!/usr/bin/env python3
"""Golden for the Gemma-4 NVFP4 checkpoint (W4A16 reference): HF transformers, CPU, bf16.

Loads the modelopt NVFP4 checkpoint into an in-memory Gemma4ForCausalLM (text only):
  * per-expert gate/up/down `weight` (U8 packed e2m1, low nibble = even element),
    `weight_scale` (F8_E4M3 per 16), `weight_scale_2` (F32 scalar) are dequantised to bf16:
        W[n,k] = e2m1(nibble) * e4m3(weight_scale[n,k/16]) * weight_scale_2
    (same as crates/intarsia-core/src/quant/nvfp4.rs `dequant_rows`); `input_scale` is ignored.
  * everything else (attention, dense MLP, router, norms, embeddings) is bf16 as stored.
  * experts are fused to HF's [E, 2*I, H] gate_up_proj (gate rows first) and [E, H, I] down_proj.
Then runs the existing tools/golden.py unchanged (its AutoModelForCausalLM is swapped for this loader),
so the output format is exactly golden.py's. A checkpoint without weight_scale tensors (e.g. the
tiny one from gemma4_tiny.py) loads as plain bf16.

  python tools/gemma4_nvfp4_golden.py --model SNAPSHOT_DIR --out ../golden/NAME.safetensors \
      --topk 20 --template hf [--prompts tools/prompts/golden.jsonl]
(all golden.py flags pass through). `--selftest` runs the dequant unit check only.
"""
import json
import pathlib
import sys
import time

import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
LUT = torch.tensor(E2M1 + [-v for v in E2M1], dtype=torch.float32)  # bit 3 = sign
GROUP = 16


def dequant_nvfp4(w_u8, w_scale, gs):
    """w_u8 [N,K/2] uint8, w_scale [N,K/16] float8_e4m3fn, gs f32 scalar -> bf16 [N,K]."""
    n, kh = w_u8.shape
    lo = (w_u8 & 0xF).long()
    hi = (w_u8 >> 4).long()
    codes = torch.stack([lo, hi], dim=-1).reshape(n, kh * 2)  # low nibble = even element
    vals = LUT.to(codes.device)[codes]
    sc = w_scale.to(torch.float32).repeat_interleave(GROUP, dim=1)
    assert sc.shape == vals.shape, (sc.shape, vals.shape)
    return (vals * sc * float(gs)).to(torch.bfloat16)


def selftest():
    w = torch.tensor([[0x21, 0xF9]], dtype=torch.uint8)  # elems: 0.5, 1.0, -6.0, -0.5
    sc = torch.tensor([[2.0]], dtype=torch.float32).to(torch.float8_e4m3fn)
    out = dequant_nvfp4(torch.cat([w, torch.zeros(1, 6, dtype=torch.uint8)], 1), sc, 0.5).float()
    assert out[0, :4].tolist() == [0.5, 1.0, -0.5, -6.0], out[0, :4]  # low nibble first
    print("selftest ok:", out[0, :4].tolist())


DEV = "cpu"  # set from --device: tensors are placed (and the NVFP4 experts dequantised) straight on it,
# so a 52 GB bf16 model never exists twice (golden.py's later `.to(device)` is then a no-op)


def build_model(path, dtype=torch.bfloat16):
    from safetensors import safe_open
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    path = pathlib.Path(path)
    t0 = time.time()
    cfg = Gemma4TextConfig(**json.load(open(path / "config.json"))["text_config"])
    cfg._attn_implementation = "eager"
    with torch.device("meta"):
        model = Gemma4ForCausalLM(cfg)

    files = sorted(path.glob("*.safetensors"))
    handles = [safe_open(str(f), framework="pt") for f in files]
    where = {}
    for h in handles:
        for k in h.keys():
            where[k] = h
    P = "model.language_model."

    def get(k):
        return where[k].get_tensor(k).to(DEV)

    sd = {}
    n_deq = 0
    for k in where:
        if not k.startswith(P) or ".experts." in k:
            continue
        sd["model." + k[len(P):]] = get(k).to(dtype)
    E, I, H = cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size
    for L in range(cfg.num_hidden_layers):
        pre = f"{P}layers.{L}.experts"
        gu = torch.empty(E, 2 * I, H, dtype=dtype, device=DEV)
        dn = torch.empty(E, H, I, dtype=dtype, device=DEV)
        for e in range(E):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                b = f"{pre}.{e}.{proj}"
                if b + ".weight_scale" in where:
                    w = dequant_nvfp4(get(b + ".weight"), get(b + ".weight_scale"), get(b + ".weight_scale_2"))
                    n_deq += 1
                else:
                    w = get(b + ".weight").to(dtype)
                if proj == "gate_proj":
                    gu[e, :I] = w
                elif proj == "up_proj":
                    gu[e, I:] = w
                else:
                    dn[e] = w
        sd[f"model.layers.{L}.experts.gate_up_proj"] = gu
        sd[f"model.layers.{L}.experts.down_proj"] = dn
        print(f"[load] layer {L} experts done {time.time() - t0:.0f}s", flush=True)
    sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    assert not unexpected, unexpected
    # non-persistent buffers were created on meta: rebuild them (inv_freq stays f32 like from_pretrained)
    model.model.rotary_emb = Gemma4TextRotaryEmbedding(cfg).to(DEV)
    model.model.embed_tokens.embed_scale = torch.tensor(cfg.hidden_size ** 0.5, device=DEV)
    model.tie_weights()
    bad = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.device.type == "meta"]
    assert not bad, bad[:10]
    print(f"[load] {path} dequantised {n_deq} NVFP4 tensors in {time.time() - t0:.0f}s; missing={missing}", flush=True)
    return model.eval()


def fp32_compute(model):
    """`--dtype float32`: everything except the routed experts is held and computed in f32, and the experts compute in f32 too
    (their weights stay the bf16 dequantised values, cast per expert on the fly), so the 26B model fits in ~60 GB instead of 104 GB.
    bf16 forwards of this model flip the router's top-8 on near ties (bf16 noise is the reference noise of the bf16 golden)."""
    import types
    F = torch.nn.functional
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts
    for n, p in model.named_parameters():
        if not n.endswith((".experts.gate_up_proj", ".experts.down_proj")):
            p.data = p.data.float()

    def fwd(self, hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states)
        mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        for e in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()[:, 0].tolist():
            pos, tok = torch.where(mask[e])
            gate, up = F.linear(hidden_states[tok], self.gate_up_proj[e].float()).chunk(2, dim=-1)
            h = F.linear(self.act_fn(gate) * up, self.down_proj[e].float()) * top_k_weights[tok, pos, None]
            out.index_add_(0, tok, h.to(out.dtype))
        return out

    for m in model.modules():
        if isinstance(m, Gemma4TextExperts):
            m.forward = types.MethodType(fwd, m)
    # the bf16 forward rounds the embedding scale to bf16 (53.0); keep that value, it is what the weights were trained with
    model.model.embed_tokens.embed_scale = torch.tensor(float(torch.tensor(model.config.hidden_size ** 0.5).bfloat16()), device=DEV)
    return model


class _Shim:
    @staticmethod
    def from_pretrained(path, torch_dtype=None, dtype=None, **kw):
        dt = torch_dtype or dtype or torch.bfloat16
        if dt == torch.float32:
            return fp32_compute(build_model(path, torch.bfloat16))
        return build_model(path, dt)


def main():
    if "--selftest" in sys.argv:
        selftest()
        return
    global DEV
    if "--device" in sys.argv:
        DEV = sys.argv[sys.argv.index("--device") + 1]
    import golden
    golden.AutoModelForCausalLM = _Shim
    golden.main()


if __name__ == "__main__":
    main()
