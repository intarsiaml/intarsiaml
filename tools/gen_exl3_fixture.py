#!/usr/bin/env python3
"""Generate crates/intarsia-backend-cpu/tests/fixtures/exl3_cases.safetensors.

Runs inside a container that has exllamav3 + torch + CUDA (spark1, see
docs/results/2026-10-21-m2-exl3/raw/fixture-gen.log for the exact invocation). All reference outputs come
from exllamav3's own LinearEXL3 code, not from a reimplementation.

Cases
  A. real EXL3 tensors sliced out of the Mia-AiLab / neko-legends GLM-5.3-Flash EXL3 checkpoints
     (mcg codebook, 4 bpw). A slice trellis[a0:a1, b0:b1] + suh[16*a0:16*a1] + svh[16*b0:16*b1] is a valid
     smaller EXL3 tensor as long as the offsets/extents are multiples of 128 elements (the Hadamards are
     block-diagonal over 128).
  B. a real bf16 Qwen3-8B weight slice quantized fresh with exllamav3's quantize_exl3 (identity Hessian,
     no calibration) for a spread of (bits, codebook).

Per case cN (see __metadata__["cases"]):
  cN.trellis I16 [K/16, N/16, 16*bits]   cN.suh F16 [K]   cN.svh F16 [N]
  cN.mcg / cN.mul1 I32 marker, exactly as exllamav3 stores it (absent for 3inst)
  cN.wq F16 [K, N]  get_inner_weight_tensor()  (pure decode, rotated basis)
  cN.w  F16 [K, N]  get_weight_tensor()        (both Hadamards and sign/scale vectors applied)
  cN.x  F16 [4, K]  N(0,1) input, fixed seed
  cN.y  F16 [4, N]  LinearEXL3.forward(x, {})                        (GEMV kernel path, rows <= 144)
  cN.y1 F16 [1, N]  LinearEXL3.forward(x[0:1], {})
  cN.yr F16 [4, N]  LinearEXL3.forward(x, {"reconstruct": True})     (reconstruct + hgemm path)
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import exllamav3
from exllamav3.version import __version__ as EXL3_VERSION
from exllamav3.modules.quant.exl3 import LinearEXL3
from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
from exllamav3.util.hadamard import get_hadamard_dt

HF = "/hf"  # ~/.cache/huggingface/hub, mounted read-only


def snap(model_dir):
    return sorted(glob.glob(f"{HF}/{model_dir}/snapshots/*"))[0]


def shard_of(snapshot, key):
    idx = glob.glob(f"{snapshot}/model.safetensors.index.json")
    if idx:
        return f"{snapshot}/" + json.load(open(idx[0]))["weight_map"][key]
    return glob.glob(f"{snapshot}/*.safetensors")[0]


# (source model dir, layer, expert, proj, (a0, a1, b0, b1) trellis tile slice)
# tile slice: rows a0:a1 of the K/16 axis, cols b0:b1 of the N/16 axis. 8 tiles = 128 elements.
REAL = [
    ("models--Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw", 5, 1, "gate_proj", (0, 16, 0, 8)),     # K=256 N=128
    ("models--Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw", 5, 1, "down_proj", (0, 8, 0, 16)),     # K=128 N=256
    ("models--Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw", 17, 7, "up_proj", (8, 24, 16, 24)),    # K=256 N=128, offset slice
    ("models--neko-legends--GLM-5.3-Flash-Uncensored-EXL3", 4, 2, "gate_proj", (0, 16, 0, 8)),  # K=256 N=128
]

# (bits, codebook)
FRESH = [
    (1, "mul1"), (2, "3inst"), (3, "mcg"), (4, "3inst"), (5, "mcg"),
    (6, "mul1"), (8, "3inst"), (1.5, "mul1"), (2.5, "mul1"), (3.5, "mul1"),
]
QWEN_K, QWEN_N = 256, 128


def log(*a):
    print(*a, flush=True)


def build_linear(K, N, trellis, suh, svh, mcg, mul1, dev):
    return LinearEXL3(
        None, K, N,
        suh=suh.to(dev), svh=svh.to(dev), trellis=trellis.to(dev),
        mcg=None if mcg is None else mcg.to(dev),
        mul1=None if mul1 is None else mul1.to(dev),
    )


def w32_from_wq(wq, suh, svh):
    """float64 re-derivation of get_weight_tensor() from wq (library's own 128x128 Hadamard matrix)."""
    K, N = wq.shape
    had = get_hadamard_dt(128, "cpu", torch.float64, 1 / math.sqrt(128))
    w = wq.double().cpu()
    w = (had @ w.view(-1, 128, N)).view(K, N)
    w = w * suh.double().cpu().unsqueeze(1)
    w = (w.view(K, -1, 128) @ had).view(K, N)
    w = w * svh.double().cpu().unsqueeze(0)
    return w


def reference(idx, K, N, trellis, suh, svh, mcg, mul1, dev, out, meta, extra):
    p = f"c{idx}"
    lin = build_linear(K, N, trellis, suh, svh, mcg, mul1, dev)
    t0 = time.time()
    wq = lin.get_inner_weight_tensor()
    w = lin.get_weight_tensor()
    g = torch.Generator().manual_seed(1000 + idx)
    x = torch.randn(4, K, generator=g).half().contiguous().to(dev)
    y = lin.forward(x, {})
    y1 = lin.forward(x[0:1].contiguous(), {})
    yr = lin.forward(x, {"reconstruct": True})
    torch.cuda.synchronize()
    ref = x.float() @ w.float()
    w32 = w32_from_wq(wq, suh, svh)
    stats = {
        "max_abs_ref": ref.abs().max().item(),
        "y_vs_xw": (y.float() - ref).abs().max().item(),
        "yr_vs_xw": (yr.float() - ref).abs().max().item(),
        "y_vs_yr": (y.float() - yr.float()).abs().max().item(),
        "y1_vs_y0": (y1.float() - y[0:1].float()).abs().max().item(),
        "w_vs_w32": (w.double().cpu() - w32).abs().max().item(),
        "max_abs_w": w.abs().max().item(),
        "wq_nonfinite": int((~torch.isfinite(wq)).sum().item()),
        "w_nonfinite": int((~torch.isfinite(w)).sum().item()),
    }
    log(f"[{p}] {extra['source']} K={K} N={N} bits={extra['bits']} cb={extra['codebook']} "
        f"({time.time() - t0:.2f}s) " + " ".join(f"{k}={v:.4g}" for k, v in stats.items()))
    cpu = lambda t: t.detach().cpu().contiguous()
    out[f"{p}.trellis"] = cpu(trellis)
    out[f"{p}.suh"] = cpu(suh)
    out[f"{p}.svh"] = cpu(svh)
    if mcg is not None:
        out[f"{p}.mcg"] = cpu(mcg)
    if mul1 is not None:
        out[f"{p}.mul1"] = cpu(mul1)
    out[f"{p}.wq"] = cpu(wq)
    out[f"{p}.w"] = cpu(w)
    out[f"{p}.x"] = cpu(x)
    out[f"{p}.y"] = cpu(y)
    out[f"{p}.y1"] = cpu(y1)
    out[f"{p}.yr"] = cpu(yr)
    meta.append({"name": p, "K": K, "N": N, "x_seed": 1000 + idx, **extra})
    return lin, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/out/exl3_cases.safetensors")
    ap.add_argument("--qwen", default=None, help="Qwen3-8B snapshot dir (default: auto in /hf)")
    args = ap.parse_args()
    dev = torch.device("cuda:0")
    out, meta = {}, []
    log(f"exllamav3 {EXL3_VERSION} from {exllamav3.__file__}; torch {torch.__version__} "
        f"cuda {torch.version.cuda}; device {torch.cuda.get_device_name(0)}")
    idx = 0

    # ---- A. real sliced tensors
    for model_dir, layer, expert, proj, (a0, a1, b0, b1) in REAL:
        sn = snap(model_dir)
        base = f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}"
        with safe_open(shard_of(sn, base + ".trellis"), framework="pt") as f:
            trellis = f.get_slice(base + ".trellis")[a0:a1, b0:b1, :].contiguous()
            suh = f.get_slice(base + ".suh")[16 * a0:16 * a1].contiguous()
            svh = f.get_slice(base + ".svh")[16 * b0:16 * b1].contiguous()
            mcg = f.get_tensor(base + ".mcg")
            full = f.get_slice(base + ".trellis").get_shape()
        K, N = 16 * (a1 - a0), 16 * (b1 - b0)
        bits = trellis.shape[-1] // 16
        assert trellis.dtype == torch.int16 and suh.dtype == torch.float16 and mcg.dtype == torch.int32
        extra = {
            "source": f"{model_dir.split('--', 1)[1]}:{base}",
            "codebook": "mcg", "bits": bits,
            "origin": "real-slice",
            "slice_tiles": [a0, a1, b0, b1], "full_trellis_shape": list(full),
            "mcg_shape": list(mcg.shape), "mcg_value_u32": int(mcg.view(torch.int32).item() & 0xFFFFFFFF),
        }
        reference(idx, K, N, trellis, suh, svh, mcg, None, dev, out, meta, extra)
        idx += 1

    # ---- B. fresh quantization of a Qwen3-8B slice
    qsn = args.qwen or snap("models--Qwen--Qwen3-8B")
    wname = "model.layers.0.mlp.up_proj.weight"  # HF layout [out=12288, in=4096]
    with safe_open(shard_of(qsn, wname), framework="pt") as f:
        sl = f.get_slice(wname)
        log(f"qwen {wname} shape={sl.get_shape()} dtype={sl.get_dtype()}")
        blocks = []
        for i in range(len(FRESH)):
            # N rows i*128..(i+1)*128, K cols 0..256; exllamav3 weights are [in_features, out_features]
            blocks.append(sl[i * QWEN_N:(i + 1) * QWEN_N, 0:QWEN_K].float().t().contiguous())
    for i, (bits, cb) in enumerate(FRESH):
        W = blocks[i].to(dev)
        W0 = W.clone()  # quantize_exl3 regularizes its input tensor in place
        H = torch.eye(QWEN_K, dtype=torch.float32, device=dev)
        H_data = {
            "H": H, "first_key": f"qwen.up_proj.{i}", "count": 1, "finalized": False, "num_total": 0,
            "inf_nan": torch.zeros(2, dtype=torch.long, device=dev), "device": dev,
        }
        qa = {"seed": idx, "K": bits, "devices": [0], "device_ratios": None, "apply_out_scales": None}
        if cb == "mcg":
            qa["mcg"] = True
        elif cb == "mul1":
            qa["mul1"] = True
        t0 = time.time()
        weight_q, proxy_err, tens = quantize_exl3(W, H_data, qa, True, None, False)
        torch.cuda.synchronize()
        log(f"quantize bits={bits} cb={cb}: {time.time() - t0:.1f}s proxy_err={proxy_err:.5f} "
            f"tensors={ {k: (list(v.shape), str(v.dtype)) for k, v in tens.items()} }")
        extra = {
            "source": f"Qwen/Qwen3-8B:{wname}[{i * QWEN_N}:{(i + 1) * QWEN_N},0:{QWEN_K}]^T",
            "codebook": cb, "bits": bits, "origin": "fresh-quantize",
            "quant_args": {"seed": idx, "K": bits, "apply_out_scales": None, "hessian": "identity", "count": 1},
            "proxy_err": float(proxy_err),
            "mcg_shape": list(tens["mcg"].shape) if "mcg" in tens else None,
            "mul1_shape": list(tens["mul1"].shape) if "mul1" in tens else None,
        }
        lin, w = reference(idx, QWEN_K, QWEN_N, tens["trellis"], tens["suh"], tens["svh"],
                           tens.get("mcg"), tens.get("mul1"), dev, out, meta, extra)
        d_wq = (w.float() - weight_q.float()).abs().max().item()
        nmse = ((w.float() - W0) ** 2).sum().item() / (W0 ** 2).sum().item()
        log(f"  max|w - weight_q(returned)|={d_wq:.4g}  nmse(w vs original W)={nmse:.5f}")
        idx += 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_file(out, args.out, metadata={
        "cases": json.dumps(meta, separators=(",", ":")),
        "exllamav3_version": EXL3_VERSION,
        "generator": "tools/gen_exl3_fixture.py",
    })
    os.chmod(args.out, 0o644)  # container runs as root; keep the output readable by the host user
    log(f"wrote {args.out}: {os.path.getsize(args.out)} bytes, {len(meta)} cases, {len(out)} tensors")


if __name__ == "__main__":
    main()
