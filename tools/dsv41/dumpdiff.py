#!/usr/bin/env python3
"""Compare two INTARSIA_DSV41_DUMP directories (same graph run on two backends, e.g. cpu vs cuda) op by op.

    uv run --no-project --with numpy python tools/dsv41/dumpdiff.py <dir_a> <dir_b> [--thresh 2e-2] [--all]

File names are `{seq:05}-{step}-o{k}.{dtype}.{shape}.bin` (dsv41::runtime dump hook). Prints rel_rms (float) or the
mismatch fraction (int) per output, flags the ones above --thresh; the first flagged op is where to look.
"""
import argparse
import os
import sys

import numpy as np


def load(path, dtype):
    raw = np.fromfile(path, dtype=np.uint8)
    if dtype == "f32":
        return raw.view(np.float32).astype(np.float64)
    if dtype == "bf16":
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    if dtype == "f16":
        return raw.view(np.float16).astype(np.float64)
    if dtype == "i32":
        return raw.view(np.int32)
    if dtype == "i16":
        return raw.view(np.int16)
    return raw  # u8 / i8 / fp8: compared bytewise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--thresh", type=float, default=2e-2)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    fa, fb = sorted(os.listdir(a.a)), set(os.listdir(a.b))
    flagged = 0
    for f in fa:
        if not f.endswith(".bin"):
            continue
        if f not in fb:
            print(f"{f}: missing in {a.b}")
            continue
        dt = f.rsplit(".", 3)[1]
        x, y = load(os.path.join(a.a, f), dt), load(os.path.join(a.b, f), dt)
        if x.shape != y.shape:
            print(f"{f}: sizes {x.size} vs {y.size}")
            continue
        if x.dtype.kind == "f":
            fin = np.isfinite(x) & np.isfinite(y)
            nonfin = int((~fin).sum())
            d = x[fin] - y[fin]
            ref = np.sqrt(np.mean(x[fin] ** 2)) if fin.any() else 0.0
            v = float(np.sqrt(np.mean(d**2)) / ref) if ref > 0 else float(np.abs(d).max(initial=0.0))
            msg = f"rel_rms {v:.3e}" + (f" non-finite {nonfin}" if nonfin else "")
        else:
            v = float(np.mean(x != y)) if x.size else 0.0
            msg = f"mismatch {v:.3e}"
        bad = v > a.thresh
        flagged += bad
        if bad or a.all:
            print(f"{'!!' if bad else '  '} {f}: {msg}")
    print(f"{flagged} outputs above {a.thresh}", file=sys.stderr)


if __name__ == "__main__":
    main()
