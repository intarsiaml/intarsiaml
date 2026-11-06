#!/usr/bin/env python3
"""Weight-space error of the head formats on the real lm_head (sampled rows): per-tensor fp8 (what is built), per-row fp8, mxfp4 (per-32 e8m0).
uv run --with numpy --with ml_dtypes --with safetensors python scripts/lmhead/wstats.py MODEL_DIR [rows=20000]"""
import json, struct, sys
import numpy as np, ml_dtypes
d = sys.argv[1]; nrows = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
f = open(f"{d}/model.safetensors", "rb"); n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); base = 8 + n
t = h["lm_head.weight"]; V, K = t["shape"]; b, e = t["data_offsets"]
mm = np.memmap(f"{d}/model.safetensors", dtype=np.uint16, mode="r", offset=base + b, shape=(V, K))
amax_all = 0.0
for s in range(0, V, 8192):
    amax_all = max(amax_all, float(np.abs(mm[s:s+8192].view(ml_dtypes.bfloat16).astype(np.float32)).max()))
rng = np.random.default_rng(1); idx = np.sort(rng.choice(V, nrows, replace=False))
w = mm[idx].view(ml_dtypes.bfloat16).astype(np.float32)
print(f"V={V} K={K} global amax={amax_all:.4f}  sample rms={np.sqrt((w**2).mean()):.5f}  median |w|={np.median(np.abs(w)):.5f}  p99.9 |w|={np.percentile(np.abs(w),99.9):.4f}")
def fp8(x, scale):
    q = (x / scale).astype(ml_dtypes.float8_e4m3fn).astype(np.float32) * scale
    return q
def rep(name, q):
    err = q - w
    print(f"{name:<22} rel rms err {np.sqrt((err**2).sum()/(w**2).sum()):.4%}   per-row rel err p50/p99/max {np.percentile(np.sqrt((err**2).sum(1)/(w**2).sum(1)),[50,99,100]).round(4)}")
rep("fp8 per-tensor", fp8(w, amax_all / 448))
ra = np.abs(w).max(1, keepdims=True) / 448; rep("fp8 per-row", fp8(w, ra))
# mxfp4: e2m1 grid, scale 2^ceil(log2(amax/6)) ("even" rule approximated by floor(log2(amax))-2)
grid = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
g = w.reshape(nrows, K // 32, 32); am = np.abs(g).max(2, keepdims=True)
sc = 2.0 ** (np.floor(np.log2(np.maximum(am, 1e-30))) - 2)
x = g / sc; a = np.abs(x); q = grid[np.abs(a[..., None] - grid).argmin(-1)] * np.sign(x) * sc
rep("mxfp4 per-32 (approx)", q.reshape(nrows, K))
