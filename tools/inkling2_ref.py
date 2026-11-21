#!/usr/bin/env python3
"""fp32 / fp64 reference forward of the Inkling-2-Mini TEXT model (`inkling2_loop`).

The numerics authority is vLLM's release image (vllm 0.1.dev25855+g7cb5dae22,
`vllm/models/inkling2/nvidia/{model,loop,attention,moe,sconv}.py` and `ops/`);
docs/plans/2026-11-21-inkling2-port.md section a has the formulas, and every
step here follows that section's numbering in its comments ("a3", "a7", ...).

Two numerics knobs:

* `--rounding none` (default): every activation stays in the compute dtype
  (fp32 or `--dtype float64`); weights are the checkpoint's bf16 values.
  This is the oracle the Rust ports test against.
* `--rounding vllm`: activations are rounded to bf16 at the points vLLM stores
  them in bf16 (residual stack, projections, conv outputs, ...), to get closer
  to vLLM's logits.
* `--act-quant w4a4`: fake-quantize the expert activations like vLLM's SM100 /
  SM12x paths (NVFP4 per-token amax / 1536 with the "four over six" block
  scale choice; MXFP8 per-32 e8m0). The scale *formulas* are from
  `ops/moe.py:_quantize_x` / `ops/activation.py`; the per-block 4-vs-6 choice
  and the e2m1 / e4m3 rounding come from `ops/nvfp4.py`, which is NOT in the
  extracted sources, so that part is inferred (spec section c).

Experts stay quantized (read per expert from the mmapped safetensors) and are
dequantized only when selected, through a small LRU.

Modes:
  run       prompt -> logits (+ --dump DIR per-visit activations, --max-layers N)
  gen       greedy generation (raw text or --chat, effort 0 by default)
  selftest  prefill vs chunked prefill vs token-by-token decode agreement
  tiny      write a tiny random checkpoint (real names + formats) and its logits
  cmp       top-1 / KL against a vLLM top-k golden (.json / .jsonl)
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

torch.set_grad_enabled(False)

SNAPSHOT = Path(
    "~/.cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/"
    "b85c3cfa615fc413cc9af3379f491b963f3b586a"
).expanduser()
P = "model.llm."

# --------------------------------------------------------------------------------------
# config


@dataclass
class Cfg:
    hidden: int
    n_layers: int
    vocab: int  # padded rows of embed / unembed
    vocab_real: int
    n_heads: int
    n_kv: int
    head_dim: int
    swa_heads: int
    swa_kv: int
    swa_head_dim: int
    d_rel: int
    rel_extent: int
    window: int
    local: set
    k_conv: int
    eps: float
    qk_clip: float
    rel_clip: float
    n_floor: int | None
    alpha: float
    block: int
    n_experts: int
    topk: int
    inter: int
    gate_cap: float
    up_cap: float
    route_scale: float
    softcap: float | None
    core: range
    passes: int
    gate_rank: int
    mxfp8_layers: set = field(default_factory=set)

    @staticmethod
    def load(model_dir: Path) -> "Cfg":
        raw = json.loads((model_dir / "config.json").read_text())
        t = raw["text_config"]
        loop = t.get("loop_config")
        mx = set()
        qpath = model_dir / "hf_quant_config.json"
        if qpath.exists():
            q = json.loads(qpath.read_text())
            for g in q["model"]["parameter_groups"]:
                if g["compute_dtype"] == "mxfp8":
                    for name in g["tensors"]:
                        mx.add(int(name.split(".layers.")[1].split(".")[0]))
        assert t.get("use_value_norm_no_gamma", True) and t.get("attn_res_normalize_values", True)
        assert t.get("n_shared_experts", 0) == 0 and t.get("use_gate_bias", True)
        return Cfg(
            hidden=t["hidden_size"],
            n_layers=t["num_hidden_layers"],
            vocab=t["vocab_size"],
            vocab_real=t.get("unpadded_vocab_size") or t["vocab_size"],
            n_heads=t["num_attention_heads"],
            n_kv=t["num_key_value_heads"],
            head_dim=t["head_dim"],
            swa_heads=t.get("swa_num_attention_heads") or t["num_attention_heads"],
            swa_kv=t.get("swa_num_key_value_heads") or t["num_key_value_heads"],
            swa_head_dim=t.get("swa_head_dim") or t["head_dim"],
            d_rel=t["d_rel"],
            rel_extent=t["rel_extent"],
            window=t["sliding_window_size"],
            local=set(t["local_layer_ids"]),
            k_conv=t["sconv_kernel_size"],
            eps=t["rms_norm_eps"],
            qk_clip=t["qk_l2_clip_threshold"],
            rel_clip=t["rel_l2_clip_threshold"],
            n_floor=t.get("log_scaling_n_floor"),
            alpha=t.get("log_scaling_alpha", 0.1),
            block=t["n_layers_per_attn_res_block"],
            n_experts=t["n_routed_experts"],
            topk=t["num_experts_per_tok"],
            inter=t["intermediate_size"],
            gate_cap=t["mlp_gate_softcap"],
            up_cap=t["mlp_up_softcap"],
            route_scale=t["route_scale"],
            softcap=t.get("final_logit_softcapping"),
            core=range(loop["core_start"], loop["core_end"]) if loop else range(0),
            passes=int(loop["num_passes"]) if loop else 1,
            gate_rank=int(loop["gate_rank"]) if loop else 0,
            mxfp8_layers=mx,
        )

    def visits(self) -> list[tuple[int, int]]:
        """(layer, pass) in execution order: prelude, each core pass, coda (a12)."""
        c = self.core
        if not c:
            return [(i, 0) for i in range(self.n_layers)]
        return (
            [(i, 0) for i in range(c.start)]
            + [(i, p) for p in range(self.passes) for i in c]
            + [(i, 0) for i in range(c.stop, self.n_layers)]
        )


# --------------------------------------------------------------------------------------
# quant formats (spec section c)

# e2m1: sign bit 3, magnitude index bits 0-2.
FP4_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float64,
)


def e4m3_to_f(u8: torch.Tensor, dtype) -> torch.Tensor:
    return u8.view(torch.float8_e4m3fn).to(torch.float32).to(dtype)


def e8m0_to_f(u8: torch.Tensor, dtype) -> torch.Tensor:
    # 2^(e - 127); 255 is NaN in OCP MX (never seen in this checkpoint).
    return torch.exp2(u8.to(torch.float64) - 127.0).to(dtype)


def deq_nvfp4(w: torch.Tensor, s: torch.Tensor, global_scale: float, dtype) -> torch.Tensor:
    """w u8 [N, K/2] (low nibble = even column), s e4m3 [N, K/16], W = e2m1 * s * amax/denom."""
    n, kh = w.shape
    lo = (w & 0x0F).long()
    hi = (w >> 4).long()
    codes = torch.stack((lo, hi), dim=-1).reshape(n, 2 * kh)
    vals = FP4_LUT.to(dtype)[codes]
    scale = e4m3_to_f(s, dtype).repeat_interleave(16, dim=1)
    return vals * scale * global_scale


def deq_mxfp8(w: torch.Tensor, s: torch.Tensor, dtype) -> torch.Tensor:
    """w e4m3 [N, K], s e8m0 [N, K/32] (one scale per row per 32 columns)."""
    return e4m3_to_f(w, dtype) * e8m0_to_f(s, dtype).repeat_interleave(32, dim=1)


E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)


def round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Nearest e2m1 value (ties to the even code), saturating at 6."""
    a = x.abs().clamp(max=6.0)
    grid = E2M1_GRID.to(x.dtype)
    d = (a[..., None] - grid).abs()
    idx = d.argmin(-1)  # first (lower) index on a tie
    mid = d[..., :-1] == d[..., 1:]  # exact midpoint between code j and j + 1
    tie = torch.gather(mid, -1, idx.clamp(max=6)[..., None])[..., 0] & (idx % 2 == 1)
    idx = idx + tie.long()  # a midpoint goes to the even code
    return grid[idx] * torch.sign(x)


def round_e4m3(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32).clamp(-448, 448).to(torch.float8_e4m3fn).to(torch.float32).to(x.dtype)


def fake_quant_nvfp4_act(x: torch.Tensor) -> torch.Tensor:
    """Per-token NVFP4 activation QDQ as in vLLM `_quantize_x` (ops/moe.py:307-343): outer
    scale amax/1536, e4m3 block scales per 16. INFERRED (ops/nvfp4.py not extracted): the
    `fouroversix_mae` rule picks, per block, the block scale bmax/6 or bmax/4 (in the
    encoder-scaled domain) with the smaller absolute error."""
    amax = x.abs().amax(-1, keepdim=True)
    enc = torch.clamp(1536.0 / amax, max=torch.finfo(torch.float32).max)
    xs = x * enc
    b = xs.reshape(*x.shape[:-1], -1, 16)
    bmax = b.abs().amax(-1, keepdim=True)
    best, best_err = None, None
    for div in (6.0, 4.0):
        sc = round_e4m3(bmax / div)
        safe = torch.where(sc == 0, torch.ones_like(sc), sc)
        q = round_e2m1(b / safe) * sc
        err = (q - b).abs().sum(-1, keepdim=True)
        if best is None:
            best, best_err = q, err
        else:
            take = err < best_err
            best = torch.where(take, q, best)
            best_err = torch.where(take, err, best_err)
    return (best.reshape(x.shape) / enc).to(x.dtype)


def fake_quant_mxfp8_act(x: torch.Tensor) -> torch.Tensor:
    """Per-32 MXFP8 QDQ, OCP MX rule (INFERRED: ops/mxfp8.py not extracted): scale
    2^(floor(log2 bmax) - 8), values rounded to e4m3."""
    b = x.reshape(*x.shape[:-1], -1, 32)
    bmax = b.abs().amax(-1, keepdim=True)
    e = torch.floor(torch.log2(torch.where(bmax > 0, bmax, torch.ones_like(bmax)))) - 8
    sc = torch.exp2(e)
    q = round_e4m3(b / sc) * sc
    return q.reshape(x.shape).to(x.dtype)


# --------------------------------------------------------------------------------------
# weights


class Weights:
    def __init__(self, model_dir: Path, cfg: Cfg, dtype, expert_cache_gb: float = 6.0):
        self.cfg, self.dtype = cfg, dtype
        files = sorted(model_dir.glob("model*.safetensors"))
        self.handles = [safe_open(str(f), "pt") for f in files]
        self.where = {}
        for h in self.handles:
            for k in h.keys():
                self.where[k] = h
        self.dense: dict[str, torch.Tensor] = {}
        for k, h in self.where.items():
            if not k.startswith(P) or ".mlp.experts." in k:
                continue
            t = h.get_tensor(k)
            if k == P + "embed.weight":
                self.dense[k] = t  # bf16, gathered per token
            else:
                self.dense[k] = t.to(dtype)
        self._check_layout()
        self.cache: collections.OrderedDict = collections.OrderedDict()
        self.cache_bytes = 0
        self.cache_cap = int(expert_cache_gb * 2**30)
        self.dequants = 0

    def _check_layout(self):
        c = self.cfg
        for i in range(c.n_layers):
            full = i not in c.local
            assert (P + f"layers.{i}.k_input_norm.weight" in self.dense) == full, f"layer {i} pattern"
            assert (P + f"layers.{i}.attn.k_sconv.static_w" in self.dense) == full
            if i > 0:
                q = self.dense[P + f"attn_res.attn_queries.{i - 1}"]
                assert q.numel() == (3 if full else 1) * c.hidden
            fmt = "mxfp8" if i in c.mxfp8_layers else "nvfp4"
            has_amax = P + f"layers.{i}.mlp.experts.w13_weight_amax" in self.where
            assert has_amax == (fmt == "nvfp4"), f"layer {i} expert format"

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.dense[P + name]

    def has(self, name: str) -> bool:
        return P + name in self.dense

    def expert(self, layer: int, e: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = (layer, e)
        hit = self.cache.get(key)
        if hit is not None:
            self.cache.move_to_end(key)
            return hit
        out = []
        base = P + f"layers.{layer}.mlp.experts."
        for w in ("w13", "w2"):
            h = self.where[base + f"{w}_weight"]
            wt = h.get_slice(base + f"{w}_weight")[e]
            st = h.get_slice(base + f"{w}_weight_scale")[e]
            if layer in self.cfg.mxfp8_layers:
                out.append(deq_mxfp8(wt.view(torch.uint8), st.view(torch.uint8), self.dtype))
            else:
                amax = h.get_slice(base + f"{w}_weight_amax")[e].item()
                den = h.get_tensor(base + f"{w}_weight_outer_scale_denominator").item()
                out.append(deq_nvfp4(wt, st.view(torch.uint8), amax / den, self.dtype))
        self.dequants += 1
        val = (out[0], out[1])
        nbytes = sum(t.numel() * t.element_size() for t in val)
        self.cache[key] = val
        self.cache_bytes += nbytes
        while self.cache_bytes > self.cache_cap and len(self.cache) > 1:
            _, old = self.cache.popitem(last=False)
            self.cache_bytes -= sum(t.numel() * t.element_size() for t in old)
        return val


# --------------------------------------------------------------------------------------
# state (spec section b)


class VisitState:
    """Cross-token state of one layer visit: KV (SWA visits: an explicit ring of `window`
    rows indexed by position % window; full visits: append-only) and the four conv input
    windows (last K-1 inputs of the k / v / attn-out / mlp-out streams)."""

    def __init__(self, ring: int | None, k_conv: int):
        self.ring = ring
        self.k = self.v = None  # [S, Hkv, D]
        self.pos = None  # [S] absolute positions (ring rows: -1 = empty)
        self.conv = {}  # stream -> [K-1, C] inputs, oldest first
        self.k_conv = k_conv

    def keys(self):
        if self.k is None:
            return None, None, None
        if self.ring is None:
            return self.k, self.v, self.pos
        ok = self.pos >= 0
        return self.k[ok], self.v[ok], self.pos[ok]

    def append(self, k, v, pos):
        if self.ring is None:
            if self.k is None:
                self.k, self.v, self.pos = k, v, pos
            else:
                self.k = torch.cat((self.k, k))
                self.v = torch.cat((self.v, v))
                self.pos = torch.cat((self.pos, pos))
            return
        if self.k is None:
            self.k = k.new_zeros((self.ring, *k.shape[1:]))
            self.v = v.new_zeros((self.ring, *v.shape[1:]))
            self.pos = torch.full((self.ring,), -1, dtype=torch.long)
        rows = pos % self.ring
        self.k[rows] = k
        self.v[rows] = v
        self.pos[rows] = pos

    def conv_hist(self, stream: str, c: int, like: torch.Tensor):
        h = self.conv.get(stream)
        if h is None:
            return like.new_zeros((self.k_conv - 1, c))
        return h

    def set_conv(self, stream: str, xp: torch.Tensor):
        self.conv[stream] = xp[-(self.k_conv - 1):].clone()


class SeqState:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.visits = [
            VisitState(cfg.window if i in cfg.local else None, cfg.k_conv) for i, _ in cfg.visits()
        ]
        self.len = 0


# --------------------------------------------------------------------------------------
# model


class Ref:
    def __init__(self, w: Weights, rounding: str = "none", act_quant: str = "w4a16"):
        self.w, self.c = w, w.cfg
        self.dtype = w.dtype
        self.vllm = rounding == "vllm"
        self.a4 = act_quant == "w4a4"
        self.dump: dict[str, torch.Tensor] | None = None
        self.capture_router = False
        self.router_log: list = []
        D = self.c.hidden
        # a2: checkpoint queries are raw; vLLM pre-scales them by D**-0.5 at load.
        s = D**-0.5
        self.attn_q = [None] + [w[f"attn_res.attn_queries.{i - 1}"].reshape(-1, D) * s for i in range(1, self.c.n_layers)]
        self.mlp_q = [w[f"attn_res.mlp_queries.{i}"].reshape(-1, D) * s for i in range(self.c.n_layers)]
        self.out_q = w["attn_res.output_query"].reshape(-1, D) * s
        if self.c.core:
            n = len(self.c.core)
            qa = w["middle_cycle_cross_pass_queries"].reshape(n, 2, D)
            qkv = w["middle_cycle_cross_pass_kv_queries"].reshape(n, 2, D)
            # a12: streams (attn=q, k, v, mlp) = (queries[:,0], kv[:,0], kv[:,1], queries[:,1])
            self.msg_q = torch.stack((qa[:, 0], qkv[:, 0], qkv[:, 1], qa[:, 1]), 1) * s

    # -- helpers ---------------------------------------------------------------------
    def r(self, x):
        """Round to bf16 at a vLLM storage point (--rounding vllm only)."""
        if self.vllm:
            return x.to(torch.bfloat16).to(self.dtype)
        return x

    def d(self, name, x):
        if self.dump is not None:
            self.dump[name] = x.detach().to(torch.float32).clone()

    def rms(self, x, w=None, eps=None):
        eps = self.c.eps if eps is None else eps
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return y if w is None else y * w

    def lin(self, x, w):
        return self.r(x @ w.t())

    def attn_res(self, blocks, last, queries, norm_w=None):
        """a2: block attention residual. sources = [*blocks, last]; score_n = (q . x_n) *
        rsqrt(mean(x_n^2) + 1e-6); out = sum softmax_n * x_n * rsqrt(...); then the next
        sublayer's RMSNorm (rows of norm_w) if given, else the fp32 mix. -> [Q, T, D]"""
        src = torch.stack([*blocks, last], 0)  # [N, T, D]
        inv = torch.rsqrt(src.pow(2).mean(-1) + 1e-6)  # [N, T]
        logits = torch.einsum("qd,ntd->qnt", queries, src) * inv[None]
        p = torch.softmax(logits, dim=1) * inv[None]  # normalize_values
        out = torch.einsum("qnt,ntd->qtd", p, src)
        if norm_w is None:
            return out
        return self.r(self.rms(out, norm_w[:, None, :]))

    def conv_static(self, x, wt, hist):
        """a6: y_t = (sum_k w[c,k] x[t-(K-1)+k] + x_t) / sqrt(K+1); w [C, K], tap K-1 = current."""
        K = wt.shape[1]
        xp = torch.cat((hist, x), 0)
        T = x.shape[0]
        acc = sum(wt[:, k] * xp[k : k + T] for k in range(K))
        return (acc + x) * (K + 1) ** -0.5, xp

    def conv_dynamic(self, x, static_w, dyn, hist, heads):
        """a6: y_t = (sum_l (static_w[l,c] + dyn[t, head(c), l]) x[t-l] + x_t) / sqrt(2K+1)."""
        K = static_w.shape[0]
        T, C = x.shape
        xp = torch.cat((hist, x), 0)
        dyn = dyn.reshape(T, heads, 1, K)
        acc = 0
        for lag in range(K):
            win = xp[K - 1 - lag : K - 1 - lag + T].reshape(T, heads, C // heads)
            tap = static_w[lag].reshape(heads, C // heads) + dyn[..., lag]
            acc = acc + win * tap
        return (acc.reshape(T, C) + x) * (2 * K + 1) ** -0.5, xp

    def l2clip(self, x, thr, scale=None):
        """a7: x * scale, then * min(thr / (||x|| + 1e-6), 1) per head row."""
        if scale is not None:
            x = x * scale
        n = x.pow(2).sum(-1, keepdim=True).sqrt()
        return x * torch.clamp(thr / (n + 1e-6), max=1.0)

    # -- attention (a3-a11) -------------------------------------------------------------
    def attention(self, L, xq, xk, xv, pos, st: VisitState, tag):
        c, w = self.c, self.w
        local = L in c.local
        H = c.swa_heads if local else c.n_heads
        Hk = c.swa_kv if local else c.n_kv
        Dh = c.swa_head_dim if local else c.head_dim
        T = xq.shape[0]
        pre = f"layers.{L}.attn."
        q = self.lin(xq, w[pre + "wq_du.weight"]).reshape(T, H, Dh)
        rr = self.lin(xq, w[pre + "wr_du.weight"]).reshape(T, H, c.d_rel)
        gate = self.lin(xq, w[pre + "gate_attn.weight"])
        sink = self.lin(xq, w[pre + "attn_sink_proj.weight"])  # [T, H]
        k = self.lin(xk, w[pre + "wk_dv.weight"])
        v = self.lin(xv, w[pre + "wv_dv.weight"])
        # a6: K/V short convs over the projected k / v (vLLM rounds the conv input to bf16)
        if local:
            kc, kxp = self.conv_static(k, w[pre + "k_sconv.weight"].reshape(Hk * Dh, -1), st.conv_hist("k", Hk * Dh, k))
            vc, vxp = self.conv_static(v, w[pre + "v_sconv.weight"].reshape(Hk * Dh, -1), st.conv_hist("v", Hk * Dh, v))
        else:
            # dynamic taps are projected from the NORMED attention inputs x_k / x_v
            dk = self.lin(xk, w[pre + "k_sconv.weight_proj"])
            dv = self.lin(xv, w[pre + "v_sconv.weight_proj"])
            kc, kxp = self.conv_dynamic(k, w[pre + "k_sconv.static_w"], dk, st.conv_hist("k", Hk * Dh, k), Hk)
            vc, vxp = self.conv_dynamic(v, w[pre + "v_sconv.static_w"], dv, st.conv_hist("v", Hk * Dh, v), Hk)
        st.set_conv("k", kxp)
        st.set_conv("v", vxp)
        kc = self.r(kc).reshape(T, Hk, Dh)
        vc = self.r(vc).reshape(T, Hk, Dh)
        k = self.r(self.l2clip(kc, c.qk_clip))  # a7: k clip, no scale
        v = self.r(self.rms(vc))  # a7: v RMSNorm without gamma
        q = self.r(self.l2clip(q, c.qk_clip, w[pre + "q_l2_clip.scale"]))
        rr = self.r(self.l2clip(rr, c.rel_clip, w[pre + "rel_l2_clip.scale"]))
        if not local and c.n_floor is not None:  # a8: log-position scaling of q and r
            tau = 1.0 + c.alpha * torch.log(torch.clamp((pos + 1).to(self.dtype) / c.n_floor, min=1.0))
            q = self.r(q * tau[:, None, None])
            rr = self.r(rr * tau[:, None, None])
        relp = w[pre + "rel_logits_proj.proj"]  # [d_rel, extent]
        rel = self.r(torch.einsum("thr,re->the", rr, relp))  # [T, H, E]
        self.d(f"{tag}.q", q)
        self.d(f"{tag}.k", k)
        self.d(f"{tag}.v", v)
        self.d(f"{tag}.rel", rel)
        self.d(f"{tag}.sink", sink)
        # keys: cached (ring / list) then this chunk
        ck, cv, cp = st.keys()
        if ck is None:
            K_, V_, Pk = k, v, pos
        else:
            K_, V_, Pk = torch.cat((ck, k)), torch.cat((cv, v)), torch.cat((cp, pos))
        st.append(k, v, pos)
        G = H // Hk
        Kh = K_.repeat_interleave(G, dim=1)  # [S, H, D]
        Vh = V_.repeat_interleave(G, dim=1)
        s = torch.einsum("thd,shd->hts", q, Kh) * Dh**-0.5
        dist = pos[:, None] - Pk[None, :]  # [T, S]
        E = relp.shape[1]
        vis = dist >= 0
        if local:
            vis = vis & (dist < c.window)
        # a9: bias rel[t,h,dist] for dist < extent, 0 beyond (full layers only reach there)
        idx = dist.clamp(0, E - 1)
        bias = torch.gather(rel.permute(1, 0, 2), 2, idx[None].expand(H, -1, -1))
        bias = torch.where((dist < E)[None], bias, torch.zeros_like(bias))
        s = torch.where(vis[None], s + bias, torch.full_like(s, -math.inf))
        # a10: per-query sink logit of a zero-value key
        sk = sink.t()[:, :, None]  # [H, T, 1]
        m = torch.maximum(s.amax(-1, keepdim=True), sk)
        e = torch.exp(s - m)
        den = e.sum(-1, keepdim=True) + torch.exp(sk - m)
        o = torch.einsum("hts,shd->thd", e / den, Vh).reshape(T, H * Dh)
        o = self.r(o * torch.sigmoid(gate))  # a11: sigmoid output gate
        return self.lin(o, w[pre + "wo_ud.weight"])

    # -- MoE (a13-a14) ------------------------------------------------------------------
    def moe(self, L, x, tag):
        c, w = self.c, self.w
        logits = x @ w[f"layers.{L}.mlp.gate.weight"].t()  # fp32 router
        sel = torch.sigmoid(logits) + w[f"layers.{L}.mlp.gate.bias"]
        # top-k by sigmoid + bias; ties prefer the lower expert index (stable sort)
        order = torch.sort(sel, dim=-1, descending=True, stable=True).indices[:, : c.topk]
        ls = torch.nn.functional.logsigmoid(torch.gather(logits, 1, order))
        gam = torch.softmax(ls, dim=-1) * c.route_scale  # = sigmoid_i / sum_sel sigmoid * 8
        self.d(f"{tag}.router_logits", logits)
        self.d(f"{tag}.expert_ids", order.to(torch.float32))
        self.d(f"{tag}.expert_w", gam)
        if self.capture_router:
            self.router_log.append((tag, order.clone()))
        mx = L in c.mxfp8_layers
        xin = x
        if self.a4:
            xin = fake_quant_mxfp8_act(x) if mx else fake_quant_nvfp4_act(x)
        out = torch.zeros_like(x)
        for e in torch.unique(order).tolist():
            rows, slot = (order == e).nonzero(as_tuple=True)
            w13, w2 = self.w.expert(L, e)
            h = self.r(xin[rows] @ w13.t())  # [n, 2I], interleaved (gate, up)
            g, u = h[:, 0::2], h[:, 1::2]
            act = c.gate_cap * torch.tanh(g / c.gate_cap) * torch.sigmoid(g)
            act = act * (c.up_cap * torch.tanh(u / c.up_cap)) * gam[rows, slot][:, None]
            act = self.r(act)
            if self.a4:
                act = fake_quant_mxfp8_act(act) if mx else fake_quant_nvfp4_act(act)
            out.index_add_(0, rows, self.r(act @ w2.t()))
        return self.r(out)

    # -- loop controller (a12) --------------------------------------------------------------
    def control(self, L, stream, dest, msg):
        """(1 - tanh(forget(h))) * dest + tanh(gain(h)) * msg, h = down([rms(dest), rms(msg)])."""
        w = self.w
        pre = f"layers.{L}.controllers.{stream}."
        pair = torch.cat((self.rms(dest, eps=1e-6), self.rms(msg, eps=1e-6)), -1)
        h = self.r(self.r(pair) @ w[pre + "down_weight"].t())
        gain = h @ w[pre + "gain_weight"].t()
        forget = h @ w[pre + "forget_weight"].t()
        return (1.0 - torch.tanh(forget)) * dest + torch.tanh(gain) * msg

    # -- one layer visit ------------------------------------------------------------------
    def visit(self, vi, L, p, blocks, partial, bank, pos, st, tag):
        c, w = self.c, self.w
        local = L in c.local
        first = L % c.block == 0
        n_in = 1 if local else 3
        if local:
            nw = w[f"layers.{L}.attn_norm.weight"][None]
        else:
            nw = torch.stack([w[f"layers.{L}.{s}_input_norm.weight"] for s in "qkv"])
        msgs = None
        if p > 0:  # a12: messages from the previous pass's bank
            msgs = self.attn_res(bank[:-1], bank[-1], self.msg_q[L - c.core.start])  # [4, T, D] fp32
        if not blocks:  # layer 0: no mix, direct norm of the embedding
            dests = partial[None].expand(n_in, -1, -1)
        else:
            dests = self.attn_res(blocks, partial, self.attn_q[L])  # [1|3, T, D] fp32
        if msgs is not None:
            d3 = dests.expand(3, -1, -1)
            ins = [self.control(L, s, d3[j], msgs[j]) for j, s in enumerate(("attn", "k", "v"))]
            nw3 = nw.expand(3, -1)
            ins = [self.r(self.rms(x, nw3[j])) for j, x in enumerate(ins)]
        else:
            ins = [self.r(self.rms(dests[j], nw[j])) for j in range(n_in)]
            if local:
                ins = ins * 3
        self.d(f"{tag}.attn_in_q", ins[0])
        self.d(f"{tag}.attn_in_k", ins[1])
        self.d(f"{tag}.attn_in_v", ins[2])
        a = self.attention(L, ins[0], ins[1], ins[2], pos, st, tag)
        self.d(f"{tag}.attn_proj", a)
        # a6: attention-output short conv (no residual inside; static taps)
        ac, axp = self.conv_static(a, w[f"layers.{L}.attn_sconv.weight"].reshape(c.hidden, -1), st.conv_hist("attn", c.hidden, a))
        st.set_conv("attn", axp)
        attn_out = self.r(ac)
        self.d(f"{tag}.attn_out", attn_out)
        # a2 / a15: MLP input mix: the first layer of a block pushes its input partial
        if first:
            mix = self.attn_res(blocks + [partial], attn_out, self.mlp_q[L])
        else:
            mix = self.attn_res(blocks, partial + attn_out, self.mlp_q[L])
        y = mix[0]
        if msgs is not None:
            y = self.control(L, "mlp", y, msgs[3])
        mlp_in = self.r(self.rms(y, w[f"layers.{L}.mlp_norm.weight"]))
        self.d(f"{tag}.mlp_in", mlp_in)
        m = self.moe(L, mlp_in, tag)
        self.d(f"{tag}.moe_out", m)
        mc, mxp = self.conv_static(m, w[f"layers.{L}.mlp_sconv.weight"].reshape(c.hidden, -1), st.conv_hist("mlp", c.hidden, m))
        st.set_conv("mlp", mxp)
        mc = self.r(mc)
        if first:
            blocks = blocks + [partial]
            new = self.r(attn_out + mc)
        else:
            new = self.r((partial + attn_out) + mc)
        self.d(f"{tag}.partial_out", new)
        return blocks, new

    # -- full forward over a chunk ------------------------------------------------------
    def forward(self, ids: torch.Tensor, state: SeqState, max_visits: int | None = None) -> torch.Tensor:
        c, w = self.c, self.w
        T = ids.shape[0]
        pos = torch.arange(state.len, state.len + T)
        emb = w["embed.weight"][ids].to(self.dtype)
        x = self.r(self.rms(emb, w["embed_norm.weight"]))  # a1
        self.d("embed", x)
        blocks: list = []
        partial = x
        bank = prelude = None
        visits = c.visits()
        n = len(visits) if max_visits is None else min(max_visits, len(visits))
        for vi in range(n):
            L, p = visits[vi]
            if c.core and L == c.core.start:
                if p == 0:
                    prelude = (list(blocks), partial)
                else:
                    # a12: restart from the prelude; bank = previous pass's blocks after the
                    # prelude's (+1: the prelude partial the first core layer pushed) + partial
                    bank = blocks[len(prelude[0]) + 1 :] + [partial]
                    blocks, partial = list(prelude[0]), prelude[1]
            tag = f"v{vi:02d}"
            self.d(f"{tag}.partial_in", partial)
            blocks, partial = self.visit(vi, L, p, blocks, partial, bank if p > 0 else None, pos, state.visits[vi], tag)
        # a16: output mix, final norm, unembed, softcap, drop the padded vocab
        h = self.attn_res(blocks, partial, self.out_q, w["norm.weight"][None])[0]
        self.d("final_hidden", h)
        logits = h @ w["unembed.weight"].t()
        if c.softcap:
            logits = c.softcap * torch.tanh(logits / c.softcap)
        logits = logits[:, : c.vocab_real]
        self.d("logits", logits)
        state.len += T
        return logits


# --------------------------------------------------------------------------------------
# tokenizer / chat


def tokenizer(model_dir: Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(model_dir / "tokenizer.json"))


def chat_prompt(user: str, effort: str = "0") -> str:
    # chat_template.jinja: effort system block, user turn; no generation prompt (the model
    # opens <|message_model|> itself).
    return (
        f"<|message_system|><|content_text|>Thinking effort level: {effort}<|end_message|>"
        f"<|message_user|><|content_text|>{user}<|end_message|>"
    )


# --------------------------------------------------------------------------------------
# tiny random checkpoint


def tiny_config() -> dict:
    return {
        "architectures": ["Inkling2LoopForConditionalGeneration"],
        "model_type": "inkling2_loop",
        "eos_token_id": 5,
        "text_config": {
            "model_type": "inkling2_loop_text",
            "hidden_size": 64,
            "num_hidden_layers": 8,
            "vocab_size": 320,
            "unpadded_vocab_size": 300,
            "num_attention_heads": 6,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "swa_num_attention_heads": 6,
            "swa_num_key_value_heads": 2,
            "swa_head_dim": 16,
            "d_rel": 16,
            "rel_extent": 12,
            "sliding_window_size": 8,
            "local_layer_ids": [0, 1, 2, 4, 5, 6],
            "sconv_kernel_size": 4,
            "rms_norm_eps": 1e-6,
            "qk_l2_clip_threshold": 22.627416997969522,
            "rel_l2_clip_threshold": 8.0,
            "log_scaling_n_floor": 4,
            "log_scaling_alpha": 0.1,
            "use_value_norm_no_gamma": True,
            "n_layers_per_attn_res_block": 2,
            "attn_res_normalize_values": True,
            "n_routed_experts": 16,
            "num_experts_per_tok": 4,
            "n_shared_experts": 0,
            "intermediate_size": 32,
            "mlp_gate_softcap": 4.0,
            "mlp_up_softcap": 25.0,
            "route_scale": 8.0,
            "use_gate_bias": True,
            "final_logit_softcapping": 30.0,
            "loop_config": {"core_start": 2, "core_end": 6, "num_passes": 3, "gate_rank": 8, "is_varloop": False},
        },
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"},
    }


def quant_nvfp4(W: torch.Tensor):
    """W [E, N, K] fp32 -> (u8 [E,N,K/2], e4m3 [E,N,K/16], amax [E], denom [1]) (ModelOpt)."""
    E, N, K = W.shape
    amax = W.abs().amax(dim=(1, 2))
    gs = amax / 2688.0
    b = W.reshape(E, N, K // 16, 16) / gs[:, None, None, None]
    sc = (b.abs().amax(-1, keepdim=True) / 6.0).clamp(max=448).to(torch.float8_e4m3fn)
    scf = sc.to(torch.float32)
    q = round_e2m1(b / torch.where(scf == 0, torch.ones_like(scf), scf))
    mag = (q.abs()[..., None] == E2M1_GRID.to(torch.float32)).float().argmax(-1)
    codes = mag + 8 * (q < 0).long()
    codes = codes.reshape(E, N, K).to(torch.uint8)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return packed.contiguous(), sc.reshape(E, N, K // 16).contiguous(), amax.float(), torch.tensor([2688.0])


def quant_mxfp8(W: torch.Tensor):
    E, N, K = W.shape
    b = W.reshape(E, N, K // 32, 32)
    bmax = b.abs().amax(-1, keepdim=True)
    ex = torch.floor(torch.log2(torch.where(bmax > 0, bmax, torch.ones_like(bmax)))) - 8
    q = (b / torch.exp2(ex)).clamp(-448, 448).to(torch.float8_e4m3fn)
    s = (ex + 127).to(torch.uint8)
    return q.reshape(E, N, K).contiguous(), s.reshape(E, N, K // 32).contiguous()


def make_tiny(out: Path, seed: int = 0):
    out.mkdir(parents=True, exist_ok=True)
    cfgd = tiny_config()
    t = cfgd["text_config"]
    g = torch.Generator().manual_seed(seed)

    def rn(*shape, s=1.0):
        return torch.randn(*shape, generator=g) * s

    def un(*shape, lo=0.5, hi=1.5):
        return lo + (hi - lo) * torch.rand(*shape, generator=g)

    D, L, V = t["hidden_size"], t["num_hidden_layers"], t["vocab_size"]
    H, Hk, Dh, R = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"], t["d_rel"]
    E, I, K = t["n_routed_experts"], t["intermediate_size"], t["sconv_kernel_size"]
    local = set(t["local_layer_ids"])
    loop = t["loop_config"]
    core = range(loop["core_start"], loop["core_end"])
    mx_layers = {6, 7}
    bf = torch.bfloat16
    T: dict[str, torch.Tensor] = {}
    T[P + "embed.weight"] = rn(V, D).to(bf)
    T[P + "embed_norm.weight"] = un(D).to(bf)
    T[P + "unembed.weight"] = rn(V, D, s=D**-0.5 * 4).to(bf)
    T[P + "norm.weight"] = un(D).to(bf)
    for i in range(1, L):
        T[P + f"attn_res.attn_queries.{i - 1}"] = rn(*((D,) if i in local else (3, D)), s=1.5)
    for i in range(L):
        T[P + f"attn_res.mlp_queries.{i}"] = rn(D, s=1.5)
    T[P + "attn_res.output_query"] = rn(D, s=1.5)
    T[P + "middle_cycle_cross_pass_queries"] = rn(2 * len(core), D, s=1.5)
    T[P + "middle_cycle_cross_pass_kv_queries"] = rn(2 * len(core), D, s=1.5)
    for i in range(L):
        a = P + f"layers.{i}."
        ext = t["sliding_window_size"] if i in local else t["rel_extent"]
        T[a + "attn.wq_du.weight"] = rn(H * Dh, D, s=D**-0.5 * 2).to(bf)
        T[a + "attn.wr_du.weight"] = rn(H * R, D, s=D**-0.5 * 2).to(bf)
        T[a + "attn.gate_attn.weight"] = rn(H * Dh, D, s=D**-0.5).to(bf)
        T[a + "attn.attn_sink_proj.weight"] = rn(H, D, s=D**-0.5).to(bf)
        T[a + "attn.wk_dv.weight"] = rn(Hk * Dh, D, s=D**-0.5 * 2).to(bf)
        T[a + "attn.wv_dv.weight"] = rn(Hk * Dh, D, s=D**-0.5).to(bf)
        T[a + "attn.wo_ud.weight"] = rn(D, H * Dh, s=(H * Dh) ** -0.5).to(bf)
        T[a + "attn.q_l2_clip.scale"] = un(H, Dh, lo=0.5, hi=8.0).to(bf)  # some rows clip, some not
        T[a + "attn.rel_l2_clip.scale"] = un(H, R, lo=0.5, hi=4.0).to(bf)
        T[a + "attn.rel_logits_proj.proj"] = rn(R, ext, s=0.5).to(bf)
        if i in local:
            T[a + "attn_norm.weight"] = un(D).to(bf)
            T[a + "attn.k_sconv.weight"] = rn(Hk * Dh, 1, K, s=0.5).to(bf)
            T[a + "attn.v_sconv.weight"] = rn(Hk * Dh, 1, K, s=0.5).to(bf)
        else:
            for s_ in "qkv":
                T[a + f"{s_}_input_norm.weight"] = un(D).to(bf)
            for s_ in "kv":
                T[a + f"attn.{s_}_sconv.static_w"] = rn(K, Hk * Dh, s=0.5).to(bf)
                T[a + f"attn.{s_}_sconv.weight_proj"] = rn(Hk * K, D, s=D**-0.5 * 0.5).to(bf)
        T[a + "attn_sconv.weight"] = rn(D, 1, K, s=0.5).to(bf)
        T[a + "mlp_sconv.weight"] = rn(D, 1, K, s=0.5).to(bf)
        T[a + "mlp_norm.weight"] = un(D).to(bf)
        T[a + "mlp.gate.weight"] = rn(E, D, s=D**-0.5 * 2).to(bf)
        T[a + "mlp.gate.bias"] = rn(E, s=0.1)
        w13 = rn(E, 2 * I, D, s=D**-0.5 * 3)
        w2 = rn(E, D, I, s=I**-0.5)
        x = a + "mlp.experts."
        if i in mx_layers:
            T[x + "w13_weight"], T[x + "w13_weight_scale"] = quant_mxfp8(w13)
            T[x + "w2_weight"], T[x + "w2_weight_scale"] = quant_mxfp8(w2)
        else:
            for nm, W in (("w13", w13), ("w2", w2)):
                q, s_, am, den = quant_nvfp4(W)
                T[x + f"{nm}_weight"], T[x + f"{nm}_weight_scale"] = q, s_
                T[x + f"{nm}_weight_amax"], T[x + f"{nm}_weight_outer_scale_denominator"] = am, den
        if i in core:
            for st_ in ("attn", "k", "v", "mlp"):
                c_ = a + f"controllers.{st_}."
                T[c_ + "down_weight"] = rn(loop["gate_rank"], 2 * D, s=(2 * D) ** -0.5).to(bf)
                T[c_ + "gain_weight"] = rn(D, loop["gate_rank"], s=loop["gate_rank"] ** -0.5).to(bf)
                T[c_ + "forget_weight"] = rn(D, loop["gate_rank"], s=loop["gate_rank"] ** -0.5).to(bf)
    save_file(T, str(out / "model-00001-of-00001.safetensors"), metadata={"format": "pt"})
    (out / "config.json").write_text(json.dumps(cfgd, indent=1))
    groups = [
        {"compute_dtype": "nvfp4", "tensors": [P + f"layers.{i}.mlp.experts.{w}_weight" for i in range(L) if i not in mx_layers for w in ("w13", "w2")],
         "weight_block_shape": [1, 16], "weight_scale_format": "amax_denominator", "weight_transform": "nvfp4_nearest",
         "output_dtype": "uint8",
         "required_activation_quantization": {"amax_type": "per_token", "block_size": 16, "format": "nvfp4_e2m1",
                                              "rounding": "nearest", "scale_dtype": "float8_e4m3fn",
                                              "scale_selection_rule": "fouroversix_mae"}},
        {"compute_dtype": "mxfp8", "tensors": [P + f"layers.{i}.mlp.experts.{w}_weight" for i in sorted(mx_layers) for w in ("w13", "w2")],
         "weight_block_shape": [32, 32], "weight_transform": "mxfp8_nearest", "output_dtype": "float8_e4m3fn",
         "required_activation_quantization": {"block_size": 32, "format": "mxfp8_e4m3", "rounding": "nearest", "scale_dtype": "ue8m0"}},
    ]
    (out / "hf_quant_config.json").write_text(json.dumps({"quant_method": "inkling2", "model": {"parameter_groups": groups}}, indent=1))
    return out


# --------------------------------------------------------------------------------------
# drivers


def build(model_dir: Path, dtype, rounding="none", act_quant="w4a16", cache_gb=6.0):
    cfg = Cfg.load(model_dir)
    w = Weights(model_dir, cfg, dtype, cache_gb)
    return Ref(w, rounding, act_quant)


def run_chunks(ref: Ref, ids: list[int], chunks: list[int]) -> torch.Tensor:
    st = SeqState(ref.c)
    outs, i = [], 0
    for n in chunks:
        outs.append(ref.forward(torch.tensor(ids[i : i + n]), st))
        i += n
    assert i == len(ids)
    return torch.cat(outs)


def selftest(ref: Ref, ids: list[int]) -> float:
    """Whole-sequence prefill vs chunked prefill (crossing a boundary) vs token-by-token."""
    n = len(ids)
    a = run_chunks(ref, ids, [n])
    split = [n // 3, n // 3 + 1, n - 2 * (n // 3) - 1]
    b = run_chunks(ref, ids, split)
    c = run_chunks(ref, ids, [1] * n)
    worst = 0.0
    for name, x in (("chunked", b), ("decode", c)):
        diff = (x - a).abs().max().item()
        rel = diff / a.abs().max().item()
        same = (x.argmax(-1) == a.argmax(-1)).float().mean().item()
        print(f"selftest {name:8s} vs prefill: max|d|={diff:.3e} rel={rel:.3e} top1 agree={same:.3f}")
        worst = max(worst, rel)
    return worst


def greedy(ref: Ref, ids: list[int], n_new: int, eos: set[int]) -> list[int]:
    st = SeqState(ref.c)
    logits = ref.forward(torch.tensor(ids), st)
    out = []
    for _ in range(n_new):
        nxt = int(logits[-1].argmax())
        out.append(nxt)
        if nxt in eos:
            break
        logits = ref.forward(torch.tensor([nxt]), st)
    return out


def kl_top(ref_logits: torch.Tensor, gold: dict) -> tuple[float, float]:
    """KL(gold || ref) over gold's top-k support (renormalized) and top-1 agreement."""
    lp = torch.log_softmax(ref_logits.double(), -1)
    ids = torch.tensor(gold["ids"])
    glp = torch.tensor(gold["logprobs"], dtype=torch.float64)
    gp = torch.softmax(glp, -1)
    kl = (gp * (torch.log(gp) - torch.log_softmax(lp[ids], -1))).sum().item()
    return kl, float(int(ref_logits.argmax()) == int(ids[int(glp.argmax())]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["run", "gen", "selftest", "tiny", "cmp"])
    ap.add_argument("--model", type=Path, default=SNAPSHOT)
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--rounding", choices=["none", "vllm"], default="none")
    ap.add_argument("--act-quant", choices=["w4a16", "w4a4"], default="w4a16")
    ap.add_argument("--prompt", action="append", default=[])
    ap.add_argument("--ids", type=str, default=None, help="comma-separated token ids")
    ap.add_argument("--chat", action="store_true", help="wrap prompts in the chat template (effort --effort)")
    ap.add_argument("--effort", default="0")
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--max-layers", "--max-visits", dest="max_visits", type=int, default=None,
                    help="run only the first N layer visits (execution order), then the output head")
    ap.add_argument("--dump", type=Path, default=None, help="write per-visit activations (safetensors) here")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--golden", type=Path, default=None)
    ap.add_argument("--expert-cache-gb", type=float, default=6.0)
    a = ap.parse_args()
    dtype = getattr(torch, a.dtype)
    t0 = time.time()

    if a.mode == "tiny":
        out = a.out or Path(__file__).resolve().parent / "inkling2_tiny"
        make_tiny(out)
        ids = [(7 * i + 3) % 300 for i in range(40)]  # > window (8) and > extent (12)
        res = {"prompt_ids": ids}
        for dt in ("float64", "float32"):
            ref = build(out, getattr(torch, dt))
            worst = selftest(ref, ids)
            # f32: a near-tie in the top-k routing can flip between chunkings; f64 must agree
            assert worst < (1e-10 if dt == "float64" else 5e-3), worst
            res[dt] = run_chunks(ref, ids, [len(ids)]).to(torch.float32)
        save_file({"prompt_ids": torch.tensor(ids, dtype=torch.int64), "logits_f64": res["float64"],
                   "logits_f32": res["float32"]}, str(out / "ref_logits.safetensors"))
        d = (res["float64"] - res["float32"]).abs().max().item()
        print(f"tiny checkpoint + logits written to {out} (f32 vs f64 max|d| {d:.2e})")
        return

    ref = build(a.model, dtype, a.rounding, a.act_quant, a.expert_cache_gb)
    print(f"loaded in {time.time() - t0:.0f}s", file=sys.stderr)
    tok = None
    prompts: list[list[int]] = []
    if a.ids:
        prompts.append([int(x) for x in a.ids.split(",")])
    if a.prompt:
        tok = tokenizer(a.model)
        for p in a.prompt:
            text = chat_prompt(p, a.effort) if a.chat else p
            prompts.append(tok.encode(text, add_special_tokens=False).ids)

    if a.mode == "selftest":
        for ids in prompts:
            worst = selftest(ref, ids)
            print(f"selftest worst rel {worst:.3e}")
    elif a.mode == "run":
        for n, ids in enumerate(prompts):
            if a.dump:
                ref.dump = {}
            st = SeqState(ref.c)
            logits = ref.forward(torch.tensor(ids), st, a.max_visits)
            top = torch.topk(logits[-1], 5)
            print(f"prompt {n}: {len(ids)} tokens, last-token top5 ids {top.indices.tolist()} logits {[round(x, 3) for x in top.values.tolist()]}")
            if a.dump:
                a.dump.mkdir(parents=True, exist_ok=True)
                save_file({k: v.contiguous() for k, v in ref.dump.items()}, str(a.dump / f"prompt{n}.safetensors"),
                          metadata={"ids": json.dumps(ids), "dtype": a.dtype, "rounding": a.rounding, "act_quant": a.act_quant})
            if a.out:
                save_file({"prompt_ids": torch.tensor(ids), "logits": logits.to(torch.float32).contiguous()}, str(a.out))
    elif a.mode == "gen":
        tok = tok or tokenizer(a.model)
        eos = {200006}
        for ids in prompts:
            t1 = time.time()
            out = greedy(ref, ids, a.max_new, eos)
            text = tok.decode(out, skip_special_tokens=False)
            print(f"--- {len(ids)} prompt tokens, {len(out)} new in {time.time() - t1:.0f}s (expert dequants {ref.w.dequants})")
            print(repr(tok.decode(ids, skip_special_tokens=False)))
            print(">>>", repr(text))
    elif a.mode == "cmp":
        gold = json.loads(a.golden.read_text())
        print(json.dumps(gold)[:300])
    print(f"done in {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
