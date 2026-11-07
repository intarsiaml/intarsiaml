"""PyTorch port of intarsiaml's Qwen3.5 / 3.8 MTP draft head (`kind = "qwen3_5_mtp"`), dense SwiGLU variant (27B).

Ground truth is the engine, not vLLM; every convention below cites the line it was taken from:

  x  = fc_e(rmsnorm_e(embed(token_{p+1}))) + fc_h(rmsnorm_h(hidden_p))   mtp.rs:1062-1066 (fc = [H, 2H] split by columns:
                                                                          [0, H) takes the embedding, [H, 2H) the hidden row)
  r  = x;  r += attn(input_layernorm(r))                                 mtp.rs:1067-1085
  r += swiglu(post_attention_layernorm(r))                               mtp.rs:1086-1100 (RmsNormAdd, then ffn_block)
  g  = mtp.norm(r)         -> lm_head(g) is the draft for token p+2;     mtp.rs:1101 (RmsNormAdd with head.norm)

  * hidden_p is the main model's FINAL-NORMED hidden row (mtp.rs:5-6); the dump stores exactly that, so only
    pre_fc_norm_hidden is applied here.
  * RMSNorm everywhere (also q_norm / k_norm and mtp.norm) is x * rsqrt(mean(x^2) + eps) * (1 + w): recipes/qwen3_5_vl.toml
    `norm_offset = 1.0`, passed as `w_offset` in `OpAttrs` (mtp.rs:1060, model.rs:4073 for QkNorm), crates/intarsia-backend-cpu/src/fast.rs:106-110.
    Checkpoint weights stay in the stored convention (never save 1 + w). eps = text_config.rms_norm_eps (1e-6).
  * Gated full attention: q_proj emits [n_heads, 2 * head_dim] per row; per head the FIRST head_dim values are q and the NEXT
    head_dim the gate (gdn.rs:259 `gate_split`, model.rs:3975 reshape [t, nh, 2*hd]); output = attn * sigmoid(gate)
    (model.rs:4236-4246 `SigmoidMul`, gdn.rs:270). config.json says `output_gate_type: "swish"`; the engine uses sigmoid, so does this.
  * q_norm / k_norm: RMSNorm over head_dim per head, before rope (model.rs:4108-4130).
  * Rope: partial, rotary_dim = round(partial_rotary_factor * head_dim) = 64 of 256 (model.rs:1092-1096); NeoX half-split pairs
    (i, i + rd/2) over the first rd values only, inv_freq_i = theta^(-2 i / rd), theta = rope_parameters.rope_theta = 1e7
    (backend-cpu fast.rs:184-215); text-only so mrope reduces to plain rope (recipe `rope = "default"`). The position of
    row p is p: the head's own cache starts empty and appends row p at kv.len = p (mtp.rs:497, model.rs:4076 `pos0`).
  * Attention: scale 1/sqrt(head_dim), causal over the head's own rows, GQA with contiguous groups (q head i -> kv head
    i // (n_heads / n_kv_heads)); model.rs:4143-4150.
  * Embedding and lm_head are the main model's, frozen. The drafter's argmax runs over the first `draft_vocab` lm_head rows
    (recipes `mtp_draft_vocab = 65536`) when that is set.

lm_head sources (`--lm-head`):
  nvfp4 (default)  dequantise `lm_head.{weight,weight_scale,weight_scale_2}` of the checkpoint (ModelOpt NVFP4: u8 [V, H/2], low
                   nibble = even element, e2m1 * e4m3 scale per 16 * weight_scale_2; quant/nvfp4.rs:58-69). The e2m1 * e4m3 product
                   is exact in bf16, so the weights are kept as that exact bf16 and `weight_scale_2` is applied to the f32 logits.
                   This is what the engine computes on the NVFP4 checkpoint (CUDA).
  bf16:<file>      a bf16 `lm_head.weight` (or `model.language_model.lm_head`-like single tensor) read from a safetensors file.
  fp8              per-tensor e4m3 emulation of a bf16 head (lm_head_quant = "fp8" in the recipe): w -> e4m3(w / s) * s,
                   s = amax / 448. Applied on top of whichever source was loaded above (use with bf16:<file>).
"""
from __future__ import annotations

import glob
import json
import math
import os
import struct
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from safetensors import safe_open

# the 15 tensors of the head, in the checkpoint's names
MTP_NAMES = [
    "mtp.fc.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.norm.weight",
]
EMBED_NAME = "model.language_model.embed_tokens.weight"
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


@dataclass
class HeadConfig:
    hidden: int = 5120
    vocab: int = 248320
    n_heads: int = 24
    n_kv_heads: int = 4
    head_dim: int = 256
    inter: int = 17408
    rotary_dim: int = 64
    theta: float = 1e7
    eps: float = 1e-6

    @staticmethod
    def from_checkpoint(ckpt: str) -> "HeadConfig":
        c = json.load(open(os.path.join(ckpt, "config.json")))
        t = c.get("text_config", c)
        rp = t.get("rope_parameters", {})
        hd = t["head_dim"]
        prf = rp.get("partial_rotary_factor", t.get("partial_rotary_factor", 1.0))
        return HeadConfig(
            hidden=t["hidden_size"],
            vocab=t["vocab_size"],
            n_heads=t["num_attention_heads"],
            n_kv_heads=t["num_key_value_heads"],
            head_dim=hd,
            inter=t["intermediate_size"],
            rotary_dim=0 if abs(prf - 1.0) < 1e-6 else int(round(hd * prf)),
            theta=float(rp.get("rope_theta", t.get("rope_theta", 1e7))),
            eps=float(t.get("rms_norm_eps", 1e-6)),
        )


# ----------------------------------------------------------------------------------------------------------------------
# checkpoint access


class Checkpoint:
    """Tensor index of a (sharded) safetensors checkpoint directory; opens a shard only to read the tensors asked for."""

    def __init__(self, path: str):
        self.path = path
        idx = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(idx):
            self.weight_map = json.load(open(idx))["weight_map"]
        else:
            self.weight_map = {}
            for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
                with safe_open(f, "pt") as s:
                    for k in s.keys():
                        self.weight_map[k] = os.path.basename(f)

    def get(self, name: str, device="cpu") -> torch.Tensor:
        with safe_open(os.path.join(self.path, self.weight_map[name]), "pt", device=str(device)) as s:
            return s.get_tensor(name)

    def header(self, name: str) -> dict:
        """The safetensors header entry (dtype, shape) of a tensor, without reading it."""
        with open(os.path.join(self.path, self.weight_map[name]), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            return json.loads(f.read(n))[name]


def dequant_nvfp4(w: torch.Tensor, scale: torch.Tensor, gs: float, rows_per_chunk: int = 16384) -> torch.Tensor:
    """ModelOpt NVFP4 `[N, K/2]` u8 + `[N, K/16]` e4m3 -> bf16 `[N, K]` WITHOUT the tensor scale (`gs` is returned to the caller
    to apply on the f32 logits): e2m1 * e4m3 is exact in bf16. Element 2i is the low nibble, 2i+1 the high nibble (nvfp4.rs:58-69)."""
    dev = w.device
    lut = torch.tensor(E2M1 + [-v for v in E2M1], dtype=torch.float32, device=dev)
    n, kh = w.shape
    out = torch.empty(n, kh * 2, dtype=torch.bfloat16, device=dev)
    for r0 in range(0, n, rows_per_chunk):
        wc = w[r0 : r0 + rows_per_chunk]
        codes = torch.stack([wc & 0xF, wc >> 4], dim=-1).reshape(wc.shape[0], kh * 2).long()
        vals = lut[codes]
        s = scale[r0 : r0 + rows_per_chunk].float().repeat_interleave(16, dim=1)
        out[r0 : r0 + rows_per_chunk] = (vals * s).to(torch.bfloat16)
    return out


def load_lm_head(ck: Checkpoint, spec: str, device) -> tuple[torch.Tensor, float]:
    """Returns (weight bf16 [V, H], scalar applied to the f32 logits)."""
    fp8 = False
    if spec.endswith("+fp8"):
        spec, fp8 = spec[: -len("+fp8")], True
    if spec == "fp8":
        spec, fp8 = "nvfp4", True  # per-tensor fp8 of the dequantised head: only a stand-in, say so in the report
    if spec == "auto":  # the checkpoint's own head: NVFP4 when it carries NVFP4 scales, else its float tensor
        spec = "nvfp4" if "lm_head.weight_scale" in ck.weight_map else "ckpt"
    if spec == "ckpt":
        W, s = ck.get("lm_head.weight", device).to(torch.bfloat16), 1.0
    elif spec == "nvfp4":
        w = ck.get("lm_head.weight", device)
        scale = ck.get("lm_head.weight_scale", device)
        gs = float(ck.get("lm_head.weight_scale_2"))
        W, s = dequant_nvfp4(w, scale, gs), gs
    elif spec.startswith("bf16:"):
        with safe_open(spec[5:], "pt", device=str(device)) as f:
            keys = [k for k in f.keys() if k.endswith("lm_head.weight") or k == "lm_head"]
            if len(keys) != 1:
                raise ValueError(f"{spec[5:]}: expected exactly one lm_head tensor, found {keys}")
            W, s = f.get_tensor(keys[0]).to(torch.bfloat16), 1.0
    else:
        raise ValueError(f"--lm-head {spec!r}: auto | ckpt | nvfp4 | bf16:<file> | <any>+fp8")
    if fp8:
        W = W.float() * s
        sc = W.abs().max().item() / 448.0
        W = ((W / sc).to(torch.float8_e4m3fn).float() * sc).to(torch.bfloat16)
        s = 1.0
    return W, s


# ----------------------------------------------------------------------------------------------------------------------
# the head


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """x * rsqrt(mean(x^2) + eps) * (1 + w), computed in f32, returned in x's dtype."""
    xf = x.float()
    r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * r * (1.0 + w.float())).to(x.dtype)


class _Logits(torch.autograd.Function):
    """bf16 matmul with f32 output (torch.mm out_dtype where this torch has it), frozen weight: grad flows to g only."""

    @staticmethod
    def forward(ctx, g, W, scale):
        ctx.save_for_backward(W)
        ctx.scale = scale
        try:
            out = torch.mm(g, W.t(), out_dtype=torch.float32)
        except (TypeError, RuntimeError):
            out = torch.mm(g, W.t()).float()
        return out * scale if scale != 1.0 else out

    @staticmethod
    def backward(ctx, grad):
        (W,) = ctx.saved_tensors
        gr = grad * ctx.scale if ctx.scale != 1.0 else grad
        return torch.mm(gr.to(W.dtype), W), None, None


class MtpHead(torch.nn.Module):
    """Trainable head (the 15 mtp.* tensors, as parameters) plus frozen buffers: embed table and lm_head."""

    def __init__(self, cfg: HeadConfig, dtype=torch.bfloat16):
        super().__init__()
        self.cfg = cfg
        H, nh, nkv, hd, I = cfg.hidden, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.inter
        P = lambda *s: torch.nn.Parameter(torch.zeros(*s, dtype=dtype))
        self.fc = P(H, 2 * H)
        self.norm_e, self.norm_h = P(H), P(H)
        self.input_ln, self.post_ln = P(H), P(H)
        self.q_proj, self.k_proj, self.v_proj = P(nh * hd * 2, H), P(nkv * hd, H), P(nkv * hd, H)
        self.o_proj = P(H, nh * hd)
        self.q_norm, self.k_norm = P(hd), P(hd)
        self.gate_proj, self.up_proj, self.down_proj = P(I, H), P(I, H), P(H, I)
        self.final_norm = P(H)
        self.embed: torch.Tensor | None = None  # [V, H] bf16, frozen
        self.lm_w: torch.Tensor | None = None  # [V, H] bf16, frozen
        self.lm_scale = 1.0
        self.rope_cache: dict = {}

    # name <-> parameter ---------------------------------------------------------------------------------------------
    def named_mtp(self) -> dict[str, torch.nn.Parameter]:
        m = dict(zip(MTP_NAMES, [
            self.fc, self.norm_e, self.norm_h, self.input_ln, self.post_ln, self.q_proj, self.k_proj, self.v_proj,
            self.o_proj, self.q_norm, self.k_norm, self.gate_proj, self.up_proj, self.down_proj, self.final_norm,
        ]))
        return m

    def load_mtp(self, tensors: dict[str, torch.Tensor]) -> None:
        for name, p in self.named_mtp().items():
            t = tensors[name]
            if tuple(t.shape) != tuple(p.shape):
                raise ValueError(f"{name}: shape {tuple(t.shape)} != expected {tuple(p.shape)}")
            with torch.no_grad():
                p.copy_(t.to(p.dtype))

    # forward --------------------------------------------------------------------------------------------------------
    def _rope(self, T: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        key = (T, str(device))
        if key not in self.rope_cache:
            rd = self.cfg.rotary_dim or self.cfg.head_dim
            inv = torch.tensor([self.cfg.theta ** (-2.0 * i / rd) for i in range(rd // 2)], dtype=torch.float64, device=device)
            ang = torch.arange(T, dtype=torch.float64, device=device)[:, None] * inv[None, :]
            self.rope_cache = {key: (ang.cos().float(), ang.sin().float())}  # keep one length only
        return self.rope_cache[key]

    def _apply_rope(self, x: torch.Tensor, cs) -> torch.Tensor:
        """x [B, T, n, hd]; NeoX pairs (i, i + rd/2) over the first rd dims; position = row index."""
        cos, sin = cs
        rd = self.cfg.rotary_dim or self.cfg.head_dim
        h = rd // 2
        xf = x.float()
        x1, x2 = xf[..., :h], xf[..., h:rd]
        c, s = cos[None, :, None, :], sin[None, :, None, :]
        rot = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s, xf[..., rd:]], dim=-1)
        return rot.to(x.dtype)

    def forward(self, hidden: torch.Tensor, next_tokens: torch.Tensor) -> torch.Tensor:
        """Teacher-forced pass. hidden [B, T, H] (row p = final-normed main hidden at position p), next_tokens [B, T] (token p+1).
        Returns g [B, T, H], the head's final normed output rows (the lm_head input)."""
        cfg = self.cfg
        B, T, H = hidden.shape
        dt = hidden.dtype
        W = lambda p: p.to(dt)
        e = F.embedding(next_tokens, self.embed)
        x = F.linear(rmsnorm(e, self.norm_e, cfg.eps), W(self.fc)[:, :H]) + F.linear(rmsnorm(hidden, self.norm_h, cfg.eps), W(self.fc)[:, H:])
        # attention
        n = rmsnorm(x, self.input_ln, cfg.eps)
        qg = F.linear(n, W(self.q_proj)).view(B, T, cfg.n_heads, 2 * cfg.head_dim)
        q, gate = qg[..., : cfg.head_dim], qg[..., cfg.head_dim :]
        k = F.linear(n, W(self.k_proj)).view(B, T, cfg.n_kv_heads, cfg.head_dim)
        v = F.linear(n, W(self.v_proj)).view(B, T, cfg.n_kv_heads, cfg.head_dim)
        q, k = rmsnorm(q, self.q_norm, cfg.eps), rmsnorm(k, self.k_norm, cfg.eps)
        cs = self._rope(T, hidden.device)
        q, k = self._apply_rope(q, cs), self._apply_rope(k, cs)
        a = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True, enable_gqa=True
        ).transpose(1, 2)
        a = (a * torch.sigmoid(gate.float()).to(dt)).reshape(B, T, cfg.n_heads * cfg.head_dim)
        r = x + F.linear(a, W(self.o_proj))
        # SwiGLU
        n = rmsnorm(r, self.post_ln, cfg.eps)
        m = F.silu(F.linear(n, W(self.gate_proj))) * F.linear(n, W(self.up_proj))
        r = r + F.linear(m, W(self.down_proj))
        return rmsnorm(r, self.final_norm, cfg.eps)

    # logits ---------------------------------------------------------------------------------------------------------
    def logits(self, g: torch.Tensor, vocab: int | None = None) -> torch.Tensor:
        """f32 logits [N, vocab or V] of rows g [N, H] over the first `vocab` lm_head rows (differentiable w.r.t. g)."""
        W = self.lm_w if not vocab else self.lm_w[:vocab]
        return _Logits.apply(g, W, self.lm_scale)

    @torch.no_grad()
    def argmax(self, g: torch.Tensor, draft_vocab: int = 0, chunk: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
        """(argmax over all rows, argmax over the first draft_vocab rows) for g [N, H]; the second is empty when draft_vocab == 0."""
        full, dv = [], []
        for i in range(0, g.shape[0], chunk):
            lg = self.logits(g[i : i + chunk])
            full.append(lg.argmax(-1))
            if draft_vocab:
                dv.append(lg[:, :draft_vocab].argmax(-1))
        return torch.cat(full), (torch.cat(dv) if draft_vocab else torch.empty(0, dtype=torch.long, device=g.device))


def load_head(
    ckpt: str,
    device="cuda",
    lm_head: str = "nvfp4",
    mtp_file: str | None = None,
    param_dtype=torch.bfloat16,
    cfg: HeadConfig | None = None,
) -> MtpHead:
    """MtpHead with mtp.* (from the checkpoint, or from `mtp_file` where that holds a tensor), frozen embed and lm_head on `device`.
    Loads only mtp.*, embed_tokens and lm_head; never the 27B body."""
    ck = Checkpoint(ckpt)
    cfg = cfg or HeadConfig.from_checkpoint(ckpt)
    head = MtpHead(cfg, dtype=param_dtype)
    tens = {n: ck.get(n) for n in MTP_NAMES}
    if mtp_file:
        with safe_open(mtp_file, "pt") as f:
            have = set(f.keys())
            missing = [n for n in MTP_NAMES if n not in have]
            if missing:
                raise ValueError(f"{mtp_file} lacks {missing}")
            for n in MTP_NAMES:
                tens[n] = f.get_tensor(n)
    head.load_mtp(tens)
    head.embed = ck.get(EMBED_NAME, device).to(torch.bfloat16)
    head.lm_w, head.lm_scale = load_lm_head(ck, lm_head, device)
    for p in head.parameters():
        p.requires_grad_(False)
    return head.to(device)


def save_mtp(head: MtpHead, path: str, ckpt: str | None = None) -> None:
    """Writes the 15 mtp.* tensors as bf16 under the checkpoint's names; with `ckpt`, asserts names / dtypes / shapes against its headers."""
    from safetensors.torch import save_file

    out = {n: p.detach().to(torch.bfloat16).cpu().contiguous() for n, p in head.named_mtp().items()}
    if ckpt:
        ck = Checkpoint(ckpt)
        for n, t in out.items():
            h = ck.header(n)
            assert h["dtype"] == "BF16", (n, h["dtype"])
            assert list(t.shape) == h["shape"], (n, list(t.shape), h["shape"])
        assert set(out) == {k for k in ck.weight_map if k.startswith("mtp.")}, "tensor name set differs from the checkpoint's mtp.*"
    save_file(out, path, metadata={"format": "pt"})


# ----------------------------------------------------------------------------------------------------------------------
# dumps (`intarsia mtp-dump`)


def read_dump(path: str, device="cpu") -> dict:
    """tokens I64 [n]; hidden BF16 [n-1, H]; head_top1 I64 [n-1]; optional head_top1_dv, head_out; metadata dict."""
    out = {}
    with safe_open(path, "pt", device=str(device)) as f:
        out["meta"] = f.metadata() or {}
        for k in f.keys():
            out[k] = f.get_tensor(k)
    n = out["tokens"].shape[0]
    assert out["hidden"].shape[0] == n - 1, f"{path}: hidden has {out['hidden'].shape[0]} rows for {n} tokens (expected n-1)"
    return out


def dump_files(d: str) -> list[str]:
    return sorted(glob.glob(os.path.join(d, "*.safetensors")))


def file_id(path: str) -> str:
    return os.path.basename(path)[: -len(".safetensors")]


def category_of(fid: str) -> str:
    """Ids written by regen.py are `<category>__<n>`; anything else is category 'all'."""
    return fid.split("__", 1)[0] if "__" in fid else "all"


def load_meta(path: str | None) -> dict:
    """regen.py's sidecar: {id: {"prompt_len": int, "category": str, ...}}."""
    return json.load(open(path)) if path else {}


def draft_vocab_of(dump: dict, override: int | None) -> int:
    if override is not None:
        return override
    return int(dump["meta"].get("draft_vocab", 0) or 0)


@torch.no_grad()
def dump_g(head: MtpHead, d: dict, device, rows: int | None = None) -> torch.Tensor:
    """The head's output rows g [n-1, H] (or the first `rows`) for one dump, bf16."""
    hid = d["hidden"][:rows].to(device).to(torch.bfloat16)[None]
    nxt = d["tokens"][1 : 1 + hid.shape[1]].to(device)[None]
    return head(hid, nxt)[0]
