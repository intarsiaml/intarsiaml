#!/usr/bin/env python3
"""Build 2-layer slice (real layers 0 sliding + 5 global) of Gemma-4-26B-A4B-NVFP4.
Outputs: engine-format NVFP4 dir and HF-loadable bf16 dir (experts dequantised, fused)."""
import json, os, re, shutil, sys, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = os.path.expanduser("~/models/Gemma-4-26B-A4B-NVFP4")
OUT_Q = os.path.expanduser("~/g4-trunc/gemma4-2L-nvfp4")
OUT_B = os.path.expanduser("~/g4-trunc/gemma4-2L-bf16")
MAP = {0: 0, 5: 1}  # real layer -> slice layer
E = 128
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
LUT = torch.tensor(E2M1 + [-v for v in E2M1], dtype=torch.float32)

def dequant_nvfp4(packed, sc, gs):
    n, kh = packed.shape
    lo, hi = (packed & 0xF).long(), (packed >> 4).long()
    vals = LUT[torch.stack([lo, hi], -1).reshape(n, kh * 2)]
    return (vals * sc.float().repeat_interleave(16, 1) * float(gs)).to(torch.bfloat16)

t0 = time.time()
idx = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
handles = {}
def get(name):
    f = idx[name]
    if f not in handles:
        handles[f] = safe_open(f"{SRC}/{f}", "pt")
    return handles[f].get_tensor(name)

wanted = {}  # src name -> dst name
lp = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.*)$")
for k in idx:
    m = lp.match(k)
    if m:
        L = int(m.group(1))
        if L in MAP:
            wanted[k] = f"model.language_model.layers.{MAP[L]}.{m.group(2)}"
    elif k.startswith("model.language_model."):
        wanted[k] = k  # embed_tokens, norm
    # vision/audio: dropped
print(len(wanted), "tensors; non-layer:", [k for k in wanted if "layers" not in k], flush=True)

os.makedirs(OUT_Q, exist_ok=True); os.makedirs(OUT_B, exist_ok=True)
q, b = {}, {}
for src, dst in wanted.items():
    t = get(src)
    q[dst] = t
    if re.search(r"\.experts\.\d+\.", dst):
        continue
    b[dst] = t
print("loaded", time.time() - t0, flush=True)
# configs first need cfg; write nvfp4 now
save_file({k: v.contiguous() for k, v in q.items()}, f"{OUT_Q}/model.safetensors", metadata={"format": "pt"})
print("wrote nvfp4", time.time() - t0, flush=True)
# fused dequantised experts for bf16 version
for L in MAP.values():
    pre = f"model.language_model.layers.{L}.experts"
    gu, dn = [], []
    for e in range(E):
        d = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            base = f"{pre}.{e}.{proj}"
            d[proj] = dequant_nvfp4(q[base + ".weight"], q[base + ".weight_scale"], q[base + ".weight_scale_2"].item())
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for suf in (".weight", ".weight_scale", ".weight_scale_2", ".input_scale"):
                q.pop(f"{pre}.{e}.{proj}{suf}", None)
        gu.append(torch.cat([d["gate_proj"], d["up_proj"]], 0))
        dn.append(d["down_proj"])
    b[f"{pre}.gate_up_proj"] = torch.stack(gu).contiguous()
    b[f"{pre}.down_proj"] = torch.stack(dn).contiguous()
    print("dequant layer", L, gu[0].shape, dn[0].shape, time.time() - t0, flush=True)

# configs
cfg = json.load(open(f"{SRC}/config.json"))
tc = cfg["text_config"]
assert tc["num_kv_shared_layers"] == 0 and tc["layer_types"][0] == "sliding_attention" and tc["layer_types"][5] == "full_attention"
tc["num_hidden_layers"] = 2
tc["layer_types"] = ["sliding_attention", "full_attention"]
json.dump(cfg, open(f"{OUT_Q}/config.json", "w"), indent=2)
cfgb = json.loads(json.dumps(cfg)); cfgb.pop("quantization_config", None)
json.dump(cfgb, open(f"{OUT_B}/config.json", "w"), indent=2)
for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "processor_config.json"):
    shutil.copy(f"{SRC}/{f}", f"{OUT_B}/{f}")
    shutil.copy(f"{SRC}/{f}", f"{OUT_Q}/{f}")
shutil.copy(f"{SRC}/hf_quant_config.json", f"{OUT_Q}/hf_quant_config.json")

save_file({k: v.contiguous() for k, v in b.items()}, f"{OUT_B}/model.safetensors", metadata={"format": "pt"})
print("wrote bf16", time.time() - t0, flush=True)
del b
