#!/usr/bin/env python3
"""Read-only: the KV-cache quantisation scales the 35B checkpoint ships (names and values from the safetensors headers)."""
import json
import struct
import sys

D = sys.argv[1] if len(sys.argv) > 1 else ("/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/"
                                           "snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5")
idx = json.load(open(D + "/model.safetensors.index.json"))["weight_map"]
keys = sorted(k for k in idx if any(s in k for s in ("k_scale", "v_scale", "kv_scale")))
print(len(keys), "kv-scale tensors")


def value(k):
    with open(D + "/" + idx[k], "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        e = json.loads(fh.read(n))[k]
        fh.seek(8 + n + e["data_offsets"][0])
        raw = fh.read(e["data_offsets"][1] - e["data_offsets"][0])
    if e["dtype"] == "F32":
        return list(struct.unpack("<%df" % (len(raw) // 4), raw))
    if e["dtype"] == "BF16":
        return [struct.unpack("<f", b"\x00\x00" + raw[i:i + 2])[0] for i in range(0, len(raw), 2)]
    return [e["dtype"], e["shape"]]


for k in keys:
    print(k, value(k)[:4])
