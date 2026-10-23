import os, shutil, sys, torch
from safetensors.torch import load_file, save_file
K = 5
for kind in ("nvfp4", "bf16"):
    src = os.path.expanduser(f"~/g4-trunc/gemma4-2L-{kind}"); dst = src + "-fn"
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        if f != "model.safetensors": shutil.copy(f"{src}/{f}", f"{dst}/{f}")
    sd = load_file(f"{src}/model.safetensors")
    n = "model.language_model.norm.weight"
    w = sd[n]; w2 = (w.float() / 2**K).to(torch.bfloat16)
    assert torch.equal(w2.float() * 2**K, w.float()), "scaling not exact"
    sd[n] = w2
    save_file(sd, f"{dst}/model.safetensors", metadata={"format": "pt"})
    print(kind, "->", dst, "norm weight scaled by 2^-%d" % K, flush=True)
