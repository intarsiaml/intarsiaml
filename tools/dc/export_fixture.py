"""Lane dc: turn a fixture_<id>.npz from the MLX reference (tools/dc/ref) into raw little-endian files a Rust test reads.
usage: export_fixture.py fixture.npz outdir   (run with a python that has numpy, e.g. /opt/code/TensorFold/.venv/bin/python)"""
import json, os, sys
import numpy as np

z = np.load(sys.argv[1])
out = sys.argv[2]
os.makedirs(out, exist_ok=True)
P = int(z["P"])
meta = {"P": P, "anchor": int(z["anchor_token"]), "tokens": [int(t) for t in z["tokens"]]}
for k in z.files:
    a = z[k]
    if k in ("tokens", "P", "anchor_token") or a.dtype.kind not in "fiu" or a.ndim == 0:
        continue
    dt = np.int32 if a.dtype.kind in "iu" else np.float32
    a.astype(dt).tofile(os.path.join(out, k + (".i32" if dt == np.int32 else ".f32")))
    meta.setdefault("shapes", {})[k] = list(a.shape)
json.dump(meta, open(os.path.join(out, "meta.json"), "w"))
print(out, P, sorted(meta["shapes"]))
