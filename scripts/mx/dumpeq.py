#!/usr/bin/env python3
"""Lane mx: are two `intarsia accuracy --write-self-golden` dumps bitwise equal in every tensor (prefill and decode rows)?
Compares each tensor's bytes (the metadata, which carries the code stamp, is not compared). Exit 0 if all equal.

    dumpeq.py A.safetensors B.safetensors
"""
import json, struct, sys


def load(p):
    b = open(p, "rb").read()
    n = struct.unpack("<Q", b[:8])[0]
    h = json.loads(b[8:8 + n])
    h.pop("__metadata__", None)
    d = b[8 + n:]
    return {k: (v["dtype"], v["shape"], d[v["data_offsets"][0]:v["data_offsets"][1]]) for k, v in h.items()}


a, b = load(sys.argv[1]), load(sys.argv[2])
only = sorted(set(a) ^ set(b))
diff = sorted(k for k in set(a) & set(b) if a[k] != b[k])
print(f"{len(a)} / {len(b)} tensors; only in one: {len(only)} {only[:5]}; differing: {len(diff)} {diff[:8]}")
print("BITWISE EQUAL" if not only and not diff else "DIFFER")
sys.exit(0 if not only and not diff else 1)
