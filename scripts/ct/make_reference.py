#!/usr/bin/env python3
"""Fixtures for the compressed-tensors NVFP4A16 loader test: the reference dequantisation of a slice of packed weights.

    uv run --with compressed-tensors==0.17.1 --with torch --with psutil scripts/ct/make_reference.py crates/intarsia-models/tests/fixtures/ct

For every `*.json` in the directory (the files `extract_slice.py` writes; real slices stay out of the repository, point this at a
local directory) this adds `expected_f32_hex`: the little-endian f32
`[n, k]` dequantisation computed by compressed-tensors' own functions (`unpack_fp4_from_uint8` + `dequantize`, the two calls
`NVFP4PackedCompressor.decompress` makes, but in float32: `decompress` itself rounds through bfloat16 because
`unpack_fp4_from_uint8` defaults to it). It also writes two synthetic fixtures:

* `synthetic_every_byte.json`: every packed byte value 0..255 and a spread of e4m3 scales (subnormals up to 448), so each
  nibble code in each nibble position meets each kind of scale;
* `synthetic_roundtrip.json`: random weights compressed by compressed-tensors (`quantize` + `pack_fp4_to_uint8`, scale =
  fp8(global_scale * block_amax / 6), global_scale = 448 * 6 / amax) and then decompressed.

The bf16 `decompress` result is also checked against the float32 one (within bf16 rounding) so a change in the library's own
path shows up here.
"""
import json
import struct
import sys
from pathlib import Path

import torch
from compressed_tensors.compressors.nvfp4.base import NVFP4PackedCompressor
from compressed_tensors.compressors.nvfp4.helpers import pack_fp4_to_uint8, unpack_fp4_from_uint8
from compressed_tensors.quantization.lifecycle.forward import dequantize

GROUP = 16


def reference(packed: torch.Tensor, scale: torch.Tensor, gs: torch.Tensor) -> torch.Tensor:
    n, kb = packed.shape
    q = unpack_fp4_from_uint8(packed, n, kb * 2, dtype=torch.float32)
    return dequantize(x_q=q, scale=scale.to(torch.float32), global_scale=gs, dtype=torch.float32)


def from_json(d):
    n, k = d["n"], d["k"]
    packed = torch.frombuffer(bytearray(bytes.fromhex(d["weight_packed_hex"])), dtype=torch.uint8).reshape(n, k // 2)
    scale = torch.frombuffer(bytearray(bytes.fromhex(d["weight_scale_hex"])), dtype=torch.float8_e4m3fn).reshape(n, k // GROUP)
    gs = torch.tensor([d["weight_global_scale"]], dtype=torch.float32)
    return packed, scale, gs


def add_expected(d):
    packed, scale, gs = from_json(d)
    ref = reference(packed, scale, gs)
    # the library's own decompress (bf16) must agree with the float32 reference up to bf16 rounding
    sd = NVFP4PackedCompressor.decompress(
        {"weight_packed": packed, "weight_scale": scale, "weight_global_scale": gs}, scheme=None
    )
    bf = sd["weight"].to(torch.float32)
    err = ((bf - ref).abs() / ref.abs().clamp_min(1e-30)).max().item()
    assert err <= 2**-7, f"bf16 decompress differs from the float32 reference by {err}"
    d["expected_f32_hex"] = ref.contiguous().numpy().astype("<f4").tobytes().hex()
    return ref


def finish(path: Path, d):
    ref = add_expected(d)
    path.write_text(json.dumps(d))
    print(f"{path.name}: n={d['n']} k={d['k']} gs={d['weight_global_scale']!r} |ref| max {ref.abs().max().item():.6g} std {ref.std().item():.6g}")


def synthetic_every_byte(out: Path):
    n, k = 16, 256
    packed = torch.arange(n * k // 2, dtype=torch.int64).remainder(256).to(torch.uint8).reshape(n, k // 2)
    # finite e4m3 codes (0x7f / 0xff are NaN): subnormals 0x01..0x07, a ladder through the exponents, max 0x7e = 448
    codes = [0x01, 0x03, 0x07, 0x08, 0x20, 0x30, 0x38, 0x40, 0x55, 0x66, 0x70, 0x7E, 0x81, 0x88, 0xB8, 0xFE]
    scale = torch.tensor([codes[(i * 5 + 3) % len(codes)] for i in range(n * k // GROUP)], dtype=torch.uint8)
    scale = scale.reshape(n, k // GROUP).view(torch.float8_e4m3fn)
    d = {
        "source": "synthetic: every packed byte, spread of e4m3 scales",
        "n": n,
        "k": k,
        "weight_packed_hex": packed.numpy().tobytes().hex(),
        "weight_scale_hex": scale.view(torch.uint8).numpy().tobytes().hex(),
        "weight_global_scale": float(torch.tensor(3.7e3, dtype=torch.float32).item()),
    }
    finish(out / "synthetic_every_byte.json", d)


def synthetic_roundtrip(out: Path):
    from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy, QuantizationType
    from compressed_tensors.quantization.lifecycle.forward import quantize

    torch.manual_seed(7)
    n, k = 16, 128
    w = torch.randn(n, k) * 0.02
    w[3, :16] *= 25  # one hot block
    w[5, 32:48] = 0  # one all-zero block
    amax = w.abs().max()
    gs = (448.0 * 6.0 / amax).to(torch.float32).reshape(1)
    bmax = w.reshape(n, k // GROUP, GROUP).abs().amax(dim=-1)
    scale = (gs * bmax / 6.0).to(torch.float8_e4m3fn)
    args = QuantizationArgs(
        num_bits=4, type=QuantizationType.FLOAT, strategy=QuantizationStrategy.TENSOR_GROUP, group_size=GROUP, symmetric=True,
        scale_dtype=torch.float8_e4m3fn,
    )
    qw = quantize(x=w, scale=scale.to(torch.float32), global_scale=gs, zero_point=None, args=args)
    packed = pack_fp4_to_uint8(qw)
    back = reference(packed, scale, gs)
    rel = ((back - w).norm() / w.norm()).item()
    assert rel < 0.15, f"compress/decompress round trip is off by {rel}"
    d = {
        "source": "synthetic: random N(0, 0.02) weights through compressed-tensors quantize + pack_fp4_to_uint8",
        "n": n,
        "k": k,
        "weight_packed_hex": packed.numpy().tobytes().hex(),
        "weight_scale_hex": scale.view(torch.uint8).numpy().tobytes().hex(),
        "weight_global_scale": float(gs.item()),
    }
    finish(out / "synthetic_roundtrip.json", d)


def main():
    out = Path(sys.argv[1])
    for p in sorted(out.glob("*.json")):
        if p.name.startswith("synthetic_"):
            continue
        finish(p, json.loads(p.read_text()))
    synthetic_every_byte(out)
    synthetic_roundtrip(out)


main()
