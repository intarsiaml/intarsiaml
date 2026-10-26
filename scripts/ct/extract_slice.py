#!/usr/bin/env python3
"""Cut a small `[rows, cols]` slice out of one compressed-tensors NVFP4A16 module of a checkpoint and write it as JSON.

Stdlib only (no torch, no safetensors), so it runs on any box that holds the weights; only the header and the
requested byte ranges of the shard are read.

    extract_slice.py MODEL_DIR MODULE ROWS K OUT.json
    extract_slice.py ~/models/ThinkingCap-Qwen3.8-27B-NVFP4-f0d598e04326 model.language_model.layers.3.self_attn.q_proj 8 128 q_proj_slice.json

ROWS = first output rows to keep, K = first input elements to keep (a multiple of 16). The slice is the top-left block of
`weight_packed [N, K/2]` and `weight_scale [N, K/16]`; `weight_global_scale` is the module's own scalar.
"""
import json
import struct
import sys
from pathlib import Path


def read_tensor(model: Path, shard: str, name: str, index: dict):
    """(dtype, shape, raw bytes of the whole tensor) from a shard; reads only that byte range."""
    with open(model / shard, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(n))
        t = hdr[name]
        lo, hi = t["data_offsets"]
        f.seek(8 + n + lo)
        return t["dtype"], t["shape"], f.read(hi - lo)


def main():
    model, module, rows, k, out = Path(sys.argv[1]).expanduser(), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    assert k % 16 == 0
    wm = json.load(open(model / "model.safetensors.index.json"))["weight_map"]
    got = {}
    for part in ("weight_packed", "weight_scale", "weight_global_scale"):
        name = f"{module}.{part}"
        got[part] = read_tensor(model, wm[name], name, wm)
    (dt_w, sh_w, w), (dt_s, sh_s, s), (dt_g, sh_g, g) = got["weight_packed"], got["weight_scale"], got["weight_global_scale"]
    assert dt_w == "U8" and dt_s == "F8_E4M3" and dt_g == "F32", (dt_w, dt_s, dt_g)
    n, kb = sh_w
    assert sh_s == [n, kb * 2 // 16], (sh_w, sh_s)
    assert rows <= n and k // 2 <= kb
    wslice = b"".join(w[r * kb : r * kb + k // 2] for r in range(rows))
    sb = kb * 2 // 16
    sslice = b"".join(s[r * sb : r * sb + k // 16] for r in range(rows))
    json.dump(
        {
            "source": f"{model.name}/{module}",
            "full_shape": [n, kb * 2],
            "n": rows,
            "k": k,
            "weight_packed_hex": wslice.hex(),
            "weight_scale_hex": sslice.hex(),
            "weight_global_scale": struct.unpack("<f", g[:4])[0],
        },
        open(out, "w"),
    )
    print(f"{module}: full {n}x{kb*2}, slice {rows}x{k}, global_scale {struct.unpack('<f', g[:4])[0]!r}")


main()
