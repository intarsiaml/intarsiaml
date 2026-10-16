#!/usr/bin/env python3
"""Convert a full-vocabulary golden (intarsia-golden-v1, fp16) into a top-K golden (intarsia-golden-topk-v1).

    python3 golden_topk.py IN.safetensors OUT.safetensors [--topk 20]

Used to exercise `intarsia accuracy` top-k comparison on models that have a full golden (and to keep
disk small on boxes that are nearly full). Needs only numpy and safetensors.
"""
import argparse
import json

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

import goldenlib as gl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--topk", type=int, default=20)
    a = ap.parse_args()
    out, meta = {}, {}
    with safe_open(a.src, "np") as f:
        meta = dict(f.metadata() or {})
        if meta.get("format") != gl.FORMAT_FULL:
            raise SystemExit(f"{a.src}: not a {gl.FORMAT_FULL} file")
        i = 0
        while f"ids_{i}" in f.keys():
            lp = f.get_tensor(f"logprobs_{i}").astype(np.float32)
            lp = lp - np.log(np.exp(lp.astype(np.float64)).sum(-1, keepdims=True)).astype(np.float32)
            ix = np.argsort(-lp, axis=-1, kind="stable")[:, : a.topk]
            out[f"ids_{i}"] = f.get_tensor(f"ids_{i}")
            out[f"topk_ids_{i}"] = ix.astype(np.int32)
            out[f"topk_logprobs_{i}"] = np.take_along_axis(lp, ix, axis=-1)
            i += 1
    meta["format"] = gl.FORMAT_TOPK
    meta["topk"] = str(a.topk)
    meta["note"] = "converted from " + a.src
    save_file(out, a.dst, metadata=meta)
    print(f"wrote {a.dst} ({i} prompts, top-{a.topk})")


if __name__ == "__main__":
    main()
