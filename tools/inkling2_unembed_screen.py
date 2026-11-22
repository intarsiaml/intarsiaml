"""Unembed as an exact screen (item 6 of Fable's decode review): a quantised copy Wq of the unembed plus a per-row error
bound picks the candidate rows; only those need the exact bf16 rows. usage: inkling2_unembed_screen.py H.npz (from inkling2_dump_hidden.py; INK_MODEL = checkpoint dir; lane ink-fp8)

For a row n, |h . W_n - h . Wq_n| <= ||h||_2 * e_n with e_n = ||W_n - Wq_n||_2 (Cauchy-Schwarz; e_n precomputed at load).
Greedy: L = max_n (a_n - b_n), candidates {n : a_n + b_n >= L}, the exact argmax is among them. Top-k: L_k = k-th largest
(a_n - b_n). Also a block bound per 32 columns: sum_blocks ||h_b||_2 * e_{n,b} (tighter, 72 x more bound data).
Softcap (30 tanh(x / 30)) is monotone, so the screen runs on raw logits. Reports candidates per token."""
import sys
import numpy as np
from safetensors import safe_open

import os
SNAP = os.environ.get("INK_MODEL", "/Users/josef/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/b85c3cfa615fc413cc9af3379f491b963f3b586a")
V = 200058

def load_unembed():
    """the first V rows of the bf16 unembed as f32, parsed from the safetensors header (no torch)"""
    import glob, json, os, struct
    for fn in sorted(glob.glob(os.path.join(SNAP, "*.safetensors"))):
        with open(fn, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
            if "model.llm.unembed.weight" not in hdr:
                continue
            e = hdr["model.llm.unembed.weight"]
            s, t = e["data_offsets"]
            fh.seek(8 + n + s)
            raw = np.frombuffer(fh.read(t - s), dtype=np.uint16).reshape(e["shape"])[:V]
            return (raw.astype(np.uint32) << 16).view(np.float32)
    raise SystemExit("unembed not found")

E4M3 = None

def e4m3_grid():
    vals = []
    for b in range(256):
        s, e, m = b >> 7, (b >> 3) & 15, b & 7
        if e == 15 and m == 7:
            continue
        v = (m / 8) * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
        vals.append(-v if s else v)
    return np.unique(np.array(vals, dtype=np.float32))

def q_fp8(w):
    """per-tensor amax / 448, round to nearest e4m3 value"""
    ws = np.abs(w).max() / 448.0
    g = e4m3_grid()
    out = np.empty_like(w)
    for r0 in range(0, w.shape[0], 8192):
        x = w[r0:r0 + 8192] / ws
        i = np.clip(np.searchsorted(g, x), 1, len(g) - 1)
        lo, hi = g[i - 1], g[i]
        out[r0:r0 + 8192] = np.where(x - lo <= hi - x, lo, hi) * ws
    return out

def q_mxfp4(w):
    """e2m1 with an e8m0 scale per 32 columns (OCP: scale 2^(floor(log2 amax) - 2)), nearest on the grid, saturating"""
    g = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    n, k = w.shape
    out = np.empty_like(w)
    for r0 in range(0, n, 8192):
        x = w[r0:r0 + 8192].reshape(-1, k // 32, 32)
        amax = np.abs(x).max(-1, keepdims=True)
        sc = np.where(amax > 0, 2.0 ** (np.floor(np.log2(np.maximum(amax, 1e-30))) - 2), 1.0).astype(np.float32)
        y = np.abs(x) / sc
        i = np.clip(np.searchsorted(g, y), 1, 7)
        lo, hi = g[i - 1], g[i]
        q = np.where(y - lo <= hi - y, lo, hi)
        out[r0:r0 + 8192] = (np.sign(x) * q * sc).reshape(-1, k)
    return out

def screen(h, W, Wq, e_row, e_blk, ks=(1, 20, 50)):
    """candidates per token for greedy / top-k with the row and block bounds; also the true max |error| / row bound ratio"""
    res = {f"{b}_{k}": [] for b in ("row", "blk") for k in ks}
    ratio = []
    for r0 in range(0, h.shape[0], 64):
        hb = h[r0:r0 + 64]
        a = hb @ Wq.T
        ex = hb @ W.T
        hn = np.linalg.norm(hb, axis=1, keepdims=True)
        brow = hn * e_row[None, :]
        hblk = np.linalg.norm(hb.reshape(hb.shape[0], -1, 32), axis=2)  # [T, 72]
        bblk = hblk @ e_blk.T  # [T, V]
        ratio.append((np.abs(ex - a) / np.maximum(brow, 1e-30)).max(1))
        for name, b in (("row", brow), ("blk", bblk)):
            lo, up = a - b, a + b
            for k in ks:
                lk = -np.partition(-lo, k - 1, axis=1)[:, k - 1:k]
                c = (up >= lk).sum(1)
                # exactness check: the true top-k are all candidates
                tk = np.argpartition(-ex, k - 1, axis=1)[:, :k]
                assert (np.take_along_axis(up, tk, 1) >= lk).all()
                res[f"{name}_{k}"].append(c)
    return {k: np.concatenate(v) for k, v in res.items()}, np.concatenate(ratio)

def main():
    z = np.load(sys.argv[1])
    h = np.concatenate([z[k] for k in sorted(z.files) if k.startswith("h_")], 0).astype(np.float32)
    print(f"{h.shape[0]} positions")
    W = load_unembed()
    for name, qf, bytes_per in (("fp8", q_fp8, 1.0), ("mxfp4", q_mxfp4, 0.5 + 1 / 32)):
        Wq = qf(W)
        d = W - Wq
        e_row = np.linalg.norm(d, axis=1).astype(np.float32)
        e_blk = np.linalg.norm(d.reshape(V, -1, 32), axis=2).astype(np.float32)  # [V, 72]
        res, ratio = screen(h, W, Wq, e_row, e_blk)
        print(f"== {name} ({bytes_per:.3f} B/weight, {V * 2304 * bytes_per / 1e9:.3f} GB): true |err| / row bound max {ratio.max():.3f} median {np.median(ratio):.3f}")
        for k, c in res.items():
            print(f"  {k:8s} candidates mean {c.mean():9.1f}  p50 {np.percentile(c, 50):8.0f}  p99 {np.percentile(c, 99):8.0f}  max {c.max():8d}  "
                  f"exact-row bytes mean {c.mean() * 2304 * 2 / 1e6:.2f} MB")
        del Wq, d

if __name__ == "__main__":
    main()
