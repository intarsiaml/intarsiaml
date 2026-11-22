"""Histogram of greedy candidates per position for the fp8 unembed screen (row bound ||h|| * ||W_n - Wq_n||), lane ink-fp8.
usage: INK_MODEL=<checkpoint> python inkling2_unembed_screen_hist.py H.npz"""
import sys
import numpy as np
import inkling2_unembed_screen as S

z = np.load(sys.argv[1])
h = np.concatenate([z[k] for k in sorted(z.files) if k.startswith("h_")], 0).astype(np.float32)
W = S.load_unembed()
Wq = S.q_fp8(W)
e = np.linalg.norm(W - Wq, axis=1).astype(np.float32)
cnt = []
for r0 in range(0, h.shape[0], 64):
    hb = h[r0:r0 + 64]
    a = hb @ Wq.T
    b = np.linalg.norm(hb, axis=1, keepdims=True) * e[None, :]
    cnt.append(((a + b) >= (a - b).max(1, keepdims=True)).sum(1))
c = np.concatenate(cnt)
edges = [1, 2, 3, 5, 9, 17, 33, 65, 129, 257, 513, 1025]
print(f"positions {len(c)} mean {c.mean():.1f} p50 {np.percentile(c, 50):.0f} p90 {np.percentile(c, 90):.0f} p99 {np.percentile(c, 99):.0f} max {c.max()}")
for lo, hi in zip(edges, edges[1:] + [10**9]):
    n = ((c >= lo) & (c < hi)).sum()
    print(f"{lo}-{hi - 1 if hi < 10**9 else 'inf'}\t{n}\t{100 * n / len(c):.1f}%")
for cap in (16, 32, 64, 128, 256, 512):
    print(f"cap {cap}: fallback on {(c > cap).mean() * 100:.1f}% of positions")
