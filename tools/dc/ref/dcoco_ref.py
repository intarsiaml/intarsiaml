"""DCOCO corrector reference (float32, MLX arrays), written from the paper's equations only
(arXiv 2610.12570 sec 4.2-4.3, as transcribed in the lane note section 3). No code from
github.com/two-delta/DCOCO was read or used.

Slot index i = 0..B-1 here is the paper's i = 1..B (B = 15 drafted slots); transition p = 0..B-2
couples slot p and slot p+1.

  b_ik = LN(W_cand E[c_ik])                 LN = non-affine LayerNorm over 128 dims
  g_i  = LN(W_ctx h_i)
  z_ik = LN_gb(W_fuse [b_ik ; g_i])         LN_gb = affine (ctx_out_norm.weight/bias)
  s_p  = W_pair [h_p ; h_{p+1}]
  S_p(a,b) = l_{p+1}(b) + MLP([s_p ; z_pa ; z_{p+1,b} ; l_p(a) ; l_{p+1}(b)])
  MLP = Linear(386,256) -> GELU -> Linear(256,1)
  path: k_1 = argmax_k l_1(k); k_{p+1} = argmax_b S_p(k_p, b)
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import mlx.core as mx
import numpy as np

ORDERS = {
    # name: block order of the 386-wide MLP input; s=pair state, za/zb = z of the left/right slot
    # candidate, la/lb = logit of the left/right slot candidate
    "A": ("s", "za", "zb", "la", "lb"),   # default, the paper's listing
    "B": ("za", "zb", "s", "la", "lb"),
    "C": ("s", "zb", "za", "lb", "la"),
}


@dataclass(frozen=True)
class Variant:
    eps: float = 1e-5          # LayerNorm eps (both LN and LN_gb)
    gelu: str = "erf"          # "erf" | "tanh"
    order: str = "A"           # key of ORDERS
    hnorm: str = "post"        # "post" = h_i after final `norm` (feeds lm_head); "pre" = residual before norm
    logit: str = "raw"         # "raw" = lm_head logits; "logsoftmax" = log-softmax over the vocab; "lmax" = l - top-1 logit

    def name(self) -> str:
        d = Variant()
        parts = [f"{k}={v}" for k, v in asdict(self).items() if v != getattr(d, k)]
        return ",".join(parts) if parts else "default"

    @staticmethod
    def parse(spec: str) -> "Variant":
        if spec in ("", "default"):
            return Variant()
        kw = {}
        for item in spec.split(","):
            k, v = item.split("=")
            kw[k.strip()] = float(v) if k.strip() == "eps" else v.strip()
        return Variant(**kw)


DCOCO_KEYS = (
    "dcoco.cand_proj.weight", "dcoco.ctx_h_proj.weight", "dcoco.cand_ctx_proj.weight",
    "dcoco.ctx_out_norm.weight", "dcoco.ctx_out_norm.bias", "dcoco.pair_proj.weight",
    "dcoco.trans_mlp.0.weight", "dcoco.trans_mlp.0.bias",
    "dcoco.trans_mlp.2.weight", "dcoco.trans_mlp.2.bias",
)


def load_dcoco_weights(tensors: dict) -> dict:
    """tensors: name -> array (bf16 ok). Returns float32 mx arrays keyed by the short names."""
    return {k[len("dcoco."):]: mx.array(tensors[k]).astype(mx.float32) for k in DCOCO_KEYS}


def _ln(x, eps):
    m = mx.mean(x, axis=-1, keepdims=True)
    v = mx.mean(mx.square(x - m), axis=-1, keepdims=True)
    return (x - m) * mx.rsqrt(v + eps)


def _gelu(x, kind):
    if kind == "erf":
        return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))
    if kind == "tanh":
        return 0.5 * x * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))
    raise ValueError(kind)


def dcoco_scores(W: dict, h, cand_emb, l, var: Variant = Variant()) -> dict:
    """h [B,d] f32; cand_emb [B,K,d] f32 (E[c_ik]); l [B,K] f32 (candidate logits, descending).
    Returns dict of mx arrays: b [B,K,r], g [B,r], z [B,K,r], s [B-1,r], S [B-1,K,K]."""
    with mx.stream(mx.cpu):
        # MLX f32 matmul on the Apple GPU (M5 Max) is NOT full f32 (about 1e-3 relative error vs
        # f64), so the corrector runs on the CPU stream (Accelerate sgemm, true f32).
        return _dcoco_scores(W, h, cand_emb, l, var)


def _dcoco_scores(W, h, cand_emb, l, var):
    B, K = l.shape
    mm = lambda x, w: x @ w.T  # w is [out, in]
    b = _ln(mm(cand_emb, W["cand_proj.weight"]), var.eps)                     # [B,K,r]
    g = _ln(mm(h, W["ctx_h_proj.weight"]), var.eps)                           # [B,r]
    r = g.shape[-1]
    bg = mx.concatenate([b, mx.broadcast_to(g[:, None, :], (B, K, r))], axis=-1)  # [B,K,2r]
    z = _ln(mm(bg, W["cand_ctx_proj.weight"]), var.eps) * W["ctx_out_norm.weight"] + W["ctx_out_norm.bias"]
    s = mm(mx.concatenate([h[:-1], h[1:]], axis=-1), W["pair_proj.weight"])   # [B-1,r]
    P = B - 1
    blocks = {
        "s": mx.broadcast_to(s[:, None, None, :], (P, K, K, r)),
        "za": mx.broadcast_to(z[:-1][:, :, None, :], (P, K, K, r)),
        "zb": mx.broadcast_to(z[1:][:, None, :, :], (P, K, K, r)),
        "la": mx.broadcast_to(l[:-1][:, :, None, None], (P, K, K, 1)),
        "lb": mx.broadcast_to(l[1:][:, None, :, None], (P, K, K, 1)),
    }
    x = mx.concatenate([blocks[n] for n in ORDERS[var.order]], axis=-1)       # [P,K,K,386]
    hid = _gelu(mm(x, W["trans_mlp.0.weight"]) + W["trans_mlp.0.bias"], var.gelu)
    delta = (mm(hid, W["trans_mlp.2.weight"]) + W["trans_mlp.2.bias"])[..., 0]   # [P,K,K]
    S = l[1:][:, None, :] + delta
    out = {"b": b, "g": g, "z": z, "s": s, "S": S, "delta": delta}
    mx.eval(out)
    return out


def select_path(S: np.ndarray, l: np.ndarray) -> list[int]:
    """Prefix-first greedy: k_1 = argmax l_1; k_{p+1} = argmax_b S_p(k_p, b). Ties -> lowest index."""
    k = [int(np.argmax(l[0]))]
    for p in range(S.shape[0]):
        k.append(int(np.argmax(S[p, k[-1]])))
    return k
