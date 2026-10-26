"""Unit tests for dcoco_ref.py (run: python test_dcoco_ref.py). Synthetic weights, no model needed."""
import math

import mlx.core as mx
import numpy as np

import dcoco_ref as D

rng = np.random.default_rng(0)
d, r, K, B, M = 64, 16, 8, 6, 24


def make_weights(zero_out=False):
    t = {
        "dcoco.cand_proj.weight": rng.normal(size=(r, d)) * 0.2,
        "dcoco.ctx_h_proj.weight": rng.normal(size=(r, d)) * 0.2,
        "dcoco.cand_ctx_proj.weight": rng.normal(size=(r, 2 * r)) * 0.3,
        "dcoco.ctx_out_norm.weight": 1 + 0.1 * rng.normal(size=r),
        "dcoco.ctx_out_norm.bias": 0.1 * rng.normal(size=r),
        "dcoco.pair_proj.weight": rng.normal(size=(r, 2 * d)) * 0.1,
        "dcoco.trans_mlp.0.weight": rng.normal(size=(M, 3 * r + 2)) * 0.2,
        "dcoco.trans_mlp.0.bias": rng.normal(size=M) * 0.1,
        "dcoco.trans_mlp.2.weight": rng.normal(size=(1, M)) * 0.3,
        "dcoco.trans_mlp.2.bias": rng.normal(size=1) * 0.1,
    }
    if zero_out:
        t["dcoco.trans_mlp.2.weight"][:] = 0
        t["dcoco.trans_mlp.2.bias"][:] = 0
    return D.load_dcoco_weights({k: mx.array(v.astype(np.float32)) for k, v in t.items()}), t


def make_inputs():
    h = rng.normal(size=(B, d)).astype(np.float32)
    emb = rng.normal(size=(B, K, d)).astype(np.float32)
    l = -np.sort(-rng.normal(size=(B, K)).astype(np.float32) * 3, axis=-1)  # descending
    return h, emb, l


def naive(t, h, emb, l, var):
    """Per-element float64 loops straight from the equations."""
    ln = lambda x: (x - x.mean()) / math.sqrt(((x - x.mean()) ** 2).mean() + var.eps)
    g_ = lambda x: 0.5 * x * (1 + math.erf(x / math.sqrt(2))) if var.gelu == "erf" else \
        0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
    T = {k: np.asarray(v, dtype=np.float64) for k, v in t.items()}
    z = np.zeros((B, K, r)); g = np.zeros((B, r))
    for i in range(B):
        g[i] = ln(T["dcoco.ctx_h_proj.weight"] @ h[i])
        for k in range(K):
            b = ln(T["dcoco.cand_proj.weight"] @ emb[i, k])
            u = T["dcoco.cand_ctx_proj.weight"] @ np.concatenate([b, g[i]])
            z[i, k] = ln(u) * T["dcoco.ctx_out_norm.weight"] + T["dcoco.ctx_out_norm.bias"]
    S = np.zeros((B - 1, K, K))
    for p in range(B - 1):
        s = T["dcoco.pair_proj.weight"] @ np.concatenate([h[p], h[p + 1]])
        for a in range(K):
            for b in range(K):
                blk = {"s": s, "za": z[p, a], "zb": z[p + 1, b], "la": [l[p, a]], "lb": [l[p + 1, b]]}
                x = np.concatenate([np.asarray(blk[n], dtype=np.float64) for n in D.ORDERS[var.order]])
                hid = np.array([g_(v) for v in T["dcoco.trans_mlp.0.weight"] @ x + T["dcoco.trans_mlp.0.bias"]])
                S[p, a, b] = l[p + 1, b] + (T["dcoco.trans_mlp.2.weight"] @ hid + T["dcoco.trans_mlp.2.bias"])[0]
    return S


def test_zero_mlp_equals_argmax():
    W, _ = make_weights(zero_out=True)
    for _ in range(5):
        h, emb, l = make_inputs()
        out = D.dcoco_scores(W, mx.array(h), mx.array(emb), mx.array(l))
        S = np.array(out["S"])
        assert np.array_equal(S, np.broadcast_to(l[1:][:, None, :], S.shape))
        assert D.select_path(S, l) == [0] * B           # candidates sorted descending => argmax = index 0


def test_matches_naive():
    W, t = make_weights()
    for order in D.ORDERS:
        for gelu in ("erf", "tanh"):
            var = D.Variant(eps=1e-6, gelu=gelu, order=order)
            h, emb, l = make_inputs()
            out = D.dcoco_scores(W, mx.array(h), mx.array(emb), mx.array(l), var)
            ref = naive(t, h.astype(np.float64), emb.astype(np.float64), l.astype(np.float64), var)
            err = np.abs(np.array(out["S"]) - ref).max()
            assert err < 2e-4, (order, gelu, err)


def test_nonzero_mlp_changes_path():
    W, _ = make_weights()
    changed = 0
    for _ in range(10):
        h, emb, l = make_inputs()
        S = np.array(D.dcoco_scores(W, mx.array(h), mx.array(emb), mx.array(l))["S"])
        changed += D.select_path(S, l) != [0] * B
    assert changed > 0


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
