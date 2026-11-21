#!/usr/bin/env python3
"""Lane rocm102-bisect: print a self-golden dump's provenance and its plan's kernel map (self_meta.exec.kernels), one per line."""
import json, struct, sys
for f in sys.argv[1:]:
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    m = json.loads(h["__metadata__"]["self_meta"])
    print(f"== {f}: code {m.get('code')}")
    for k, v in sorted(m["exec"]["kernels"].items()):
        print(f"{k} {v}")
