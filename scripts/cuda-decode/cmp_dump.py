#!/usr/bin/env python3
"""Compare the tensors of two safetensors files bit for bit (header metadata ignored): cmp_dump.py a.safetensors b.safetensors.
Used on the `accuracy --write-self-golden` dumps (top-64 log-probs of every prefill and decode row) of two builds: identical
tensors mean the two builds produce the same logits ranking and log-probs at every checked position. Exit 0 when identical."""
import hashlib, json, struct, sys


def load(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            a, b = v["data_offsets"]
            f.seek(base + a)
            out[k] = (v["dtype"], v["shape"], hashlib.sha256(f.read(b - a)).hexdigest())
    return out


a, b = load(sys.argv[1]), load(sys.argv[2])
bad = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
print(f"{len(a)} tensors in {sys.argv[1]}, {len(b)} in {sys.argv[2]}: {len(bad)} differ")
for k in bad[:10]:
    print("  differs:", k, a.get(k, ("missing",))[:2], b.get(k, ("missing",))[:2])
sys.exit(1 if bad else 0)
