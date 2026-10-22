"""Bitwise comparison of two `accuracy --write-self-golden` dumps (every tensor, byte for byte).
usage: uv run --with numpy --with safetensors python scripts/35b-tp2-hip/cmpdump.py A.safetensors B.safetensors"""
import sys
import numpy as np
from safetensors import safe_open

a, b = (safe_open(p, 'np') for p in sys.argv[1:3])
ka, kb = set(a.keys()), set(b.keys())
if ka != kb:
    print(f'DIFFER: tensor names differ ({len(ka ^ kb)} not in both)')
    sys.exit(1)
bad = []
for k in sorted(ka):
    x, y = a.get_tensor(k), b.get_tensor(k)
    if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
        bad.append(k)
if bad:
    print(f'DIFFER: {len(bad)} of {len(ka)} tensors, first {bad[:5]}')
    sys.exit(1)
print(f'BITWISE IDENTICAL: {len(ka)} tensors')
