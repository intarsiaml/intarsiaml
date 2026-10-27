"""Fixture for crates/intarsia-core/src/quant/mlx_affine.rs: MLX affine-quantised matrices with the expected bf16
dequantisation from a numpy reference of the MLX formula  w = scale * q + bias  (per group, f32 arithmetic, round to
nearest even to bf16). The packed words come from mlx's own mx.quantize; the script also checks the numpy result against
mx.dequantize and fails on any difference, so the fixture ties the Rust reader to MLX itself.

  /opt/code/kev-serve/.venv/bin/python tools/gen_mlx_affine_fixture.py crates/intarsia-core/tests/fixtures/mlx_affine.json
(needs numpy and mlx; the Rust tests need neither)
"""
import json
import sys

import mlx.core as mx
import numpy as np


def bf16_bits(x32):
    u = x32.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def unpack(words_u32, rows, cols, bits):
    """Little-endian bit stream per row: element i is bits [i*bits, (i+1)*bits)."""
    raw = words_u32.astype("<u4").view(np.uint8).reshape(rows, -1)
    stream = np.unpackbits(raw, axis=1, bitorder="little")
    q = np.zeros((rows, cols), np.uint32)
    for j in range(bits):
        q |= stream[:, j : cols * bits : bits].astype(np.uint32) << j
    return q


def reference(words, scales_f32, biases_f32, rows, cols, group, bits):
    q = unpack(words, rows, cols, bits)
    out = q.astype(np.float32) * np.repeat(scales_f32, group, axis=1) + np.repeat(biases_f32, group, axis=1)
    return bf16_bits(out)


def hexof(a):
    return np.ascontiguousarray(a).tobytes().hex()


def main(path):
    rng = np.random.default_rng(20261027)
    cases = []
    for bits in (2, 3, 4, 5, 6, 8):
        for group in (32, 64, 128):
            rows, cols = 5, 256
            x = rng.standard_normal((rows, cols)).astype(np.float32) * rng.choice([0.02, 1.0, 40.0], size=(rows, 1))
            w = mx.array(x).astype(mx.bfloat16)
            wq, s, b = mx.quantize(w, group_size=group, bits=bits)
            words = np.asarray(wq).astype(np.uint32)
            s_bits = np.asarray(s.view(mx.uint16)).astype(np.uint16)
            b_bits = np.asarray(b.view(mx.uint16)).astype(np.uint16)
            sf = np.asarray(s.astype(mx.float32))
            bf = np.asarray(b.astype(mx.float32))
            ref = reference(words, sf, bf, rows, cols, group, bits)
            mxd = np.asarray(mx.dequantize(wq, s, b, group_size=group, bits=bits).view(mx.uint16)).astype(np.uint16)
            if not np.array_equal(ref, mxd):
                sys.exit(f"numpy reference differs from mx.dequantize at bits={bits} group={group}")
            cases.append(
                dict(
                    bits=bits, group=group, rows=rows, cols=cols,
                    words=hexof(words), scales=hexof(s_bits), biases=hexof(b_bits), expected=hexof(ref),
                )
            )
    json.dump({"formula": "w = scale * q + bias per group, f32, RNE to bf16", "cases": cases}, open(path, "w"))
    print(f"{len(cases)} cases -> {path}")


main(sys.argv[1])
