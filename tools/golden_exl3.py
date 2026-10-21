"""A golden made by ExLlamaV3 itself on an EXL3 checkpoint, with the token ids of an existing golden.

Usage (inside an image with exllamav3 + torch + safetensors, e.g. dsv41-serve:v4 on spark1, under fleetq):
    python3 -I tools/golden_exl3.py --model <exl3 model dir> --ids-from <existing golden.safetensors> --out <out.safetensors>

Why: comparing intarsiaml's EXL3 run against the bf16 model's golden mixes quantisation loss with engine error. This
golden is the same EXL3 weights run by the reference implementation, so KL(intarsia || this) measures the engine
alone (expected near fp16 noise: exllamav3 runs fp16 activations), while KL(this || bf16 golden) is the quant loss.

Same format as tools/golden.py (`intarsia-golden-v1`, docs/accuracy.md): `ids_<i>` I32 copied from --ids-from,
`logprobs_<i>` F16 `[n_i, vocab]` = log_softmax of exllamav3's fp32-upcast logits. One no-cache forward per prompt
(`attn_mode = flash_attn_nc`, the mode exllamav3's own perplexity eval uses). The metadata of --ids-from is kept
(prompts, template) with `format`, `model`, `device`, `dtype` and the exllamav3 / torch versions replaced.
"""
import argparse
import json

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--ids-from", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

import exllamav3
from exllamav3 import Config, Model

try:
    from exllamav3.version import __version__ as exl_version
except Exception:
    exl_version = "unknown"

with safe_open(args.ids_from, "pt") as f:
    meta = dict(f.metadata() or {})
    ids = {k: f.get_tensor(k) for k in f.keys() if k.startswith("ids_")}

config = Config.from_directory(args.model)
model = Model.from_config(config)
model.load(args.device)

out = {}
vocab = None
with torch.inference_mode():
    for k in sorted(ids, key=lambda s: int(s.split("_")[1])):
        i = k.split("_")[1]
        x = ids[k].to(torch.long).unsqueeze(0).to(args.device)
        logits = model.forward(x, {"attn_mode": "flash_attn_nc"})
        lp = torch.log_softmax(logits[0].float(), dim=-1)
        vocab = lp.shape[-1]
        out[k] = ids[k].to(torch.int32).contiguous()
        out[f"logprobs_{i}"] = lp.to(torch.float16).cpu().contiguous()
        print(f"prompt {i}: {x.shape[1]} tokens, vocab {vocab}", flush=True)

meta.update(
    {
        "format": "intarsia-golden-v1",
        "model": args.model,
        "device": torch.cuda.get_device_name(0),
        "dtype": "exllamav3 (fp16 activations)",
        "torch": torch.__version__,
        "transformers": "n/a (exllamav3 " + exl_version + ", " + exllamav3.__file__ + ")",
        "vocab": str(vocab),
        "ids_from": args.ids_from,
    }
)
save_file(out, args.out, metadata=meta)
print(json.dumps({"out": args.out, "prompts": len(ids), "vocab": vocab}))
