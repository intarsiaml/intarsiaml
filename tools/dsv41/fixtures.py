# /// script
# requires-python = ">=3.10"
# dependencies = ["torch>=2.5", "numpy", "safetensors", "tokenizers", "sympy"]
# ///
"""DeepSeek-V4.1 Flash reference fixtures: DeepSeek's official torch code, run on CPU.

Runs DeepSeek's own `model.py` and `engram.py` (MIT, Copyright (c) 2023 DeepSeek; the inference/ directory of
deepseek-ai/DeepSeek-V4.1-Flash-1017, a copy sits in .worktrees/dsv41-official) with
  * `kernel.py` (tilelang/triton GPU kernels) replaced by the pure-torch CPU equivalents below (`CpuKernels`):
    act_quant, fp4_act_quant, sparse_attn, hc_split_sinkhorn; written from kernel.py line by line, NOT bit-checked
    against the GPU kernels (none is available here);
  * every EXL3 linear of MiaAI-Lab's 2.9 bpw checkpoint decoded with TensorFold's MIT numpy decoder
    (`format.py`, `unpack`) and applied in float64 in the order of the kernels (rotate, W_q, rotate): the
    "dequantised weights" are therefore never materialised except wo_a, which the official code needs dense;
  * the Engram tables read row by row from the official shards 47/48 (FP8 E4M3 rows + E8M0 scales).

Two numeric flows: `bf16` (the official GPU dtype flow: bf16 activations, fp32 where model.py says fp32) and
`fp32` (default dtype float32 everywhere: the parity target for a CPU reference engine).

Quantisation flavour (`--quant`), the official activation fake-quant sites of the FP8/FP4 checkpoint:
  `tf` (default)  `o` before wo_a (model.py:787) is OFF: neither the EXL3 weights nor TensorFold has it. Everything
                  TensorFold reproduces stays: window KV fp8 (block 32, E8M0), compressed KV fp4 (block 16, E4M3
                  scale), indexer key fp4 (block 32, E8M0) and the indexer query fp4 (model.py:553; TensorFold runs
                  `_fp4qd_p2` on it: round.zig:956, prompt.zig:1236).
  `nofake`        additionally switches the indexer query fp4 off (the "bf16 mode" of the coordinator's review).
  `official`      the literal official code path (all sites on).

Beyond the official code (which has no chunked prefill and no multi-token decode: Compressor.forward,
_window_kv and the Indexer mask branch on start_pos == 0), `attn_chunk` below generalises Attention.forward to
arbitrary chunks; `--verify` proves it equals the official code where the official code applies and that
chunked == whole bitwise (for chunk % ratio == 0) and prefill(N) == prefill(N-3) + 3 decode steps.

Commands (run on spark1 inside a fleetq slot, see docs/research/2026-10-31-dsv41-porting-map.md):
  uv run tools/dsv41/fixtures.py selftest                       # tiny synthetic model, any machine, < 2 min
  uv run tools/dsv41/fixtures.py layers  --out DIR [options]    # layers 0-2, 16-token prefill + 2 decode steps
  uv run tools/dsv41/fixtures.py attn4k  --out DIR [options]    # layer-2 attention, ~4K-token prompt, chunk checks
  uv run tools/dsv41/fixtures.py ops     --out DIR              # synthetic op fixtures (quant, select, rope, hc)
  uv run tools/dsv41/fixtures.py ci --out DIR DST                # cut the small committed subset (golden/dsv41/)
  uv run tools/dsv41/fixtures.py manifest DIR                   # sha256 + sizes of every file under DIR
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import struct
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROW_BLOCK = 16  # EXL3 linears are applied in blocks of exactly this many rows (zero padded): row results do not
# depend on how a prompt was chunked


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# --------------------------------------------------------------------------------------------------------------
# CPU equivalents of kernel.py
# --------------------------------------------------------------------------------------------------------------
FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def _round_scale(amax: torch.Tensor, inv: float) -> torch.Tensor:
    """kernel.py fast_round_scale: 2 ** ceil(log2(amax * inv)) by exponent bits (fp32)."""
    v = amax * np.float32(inv)
    bits = v.view(torch.int32)
    e = (bits >> 23) & 0xFF
    man = bits & 0x7FFFFF
    k = (e - 127 + (man != 0).to(torch.int32)).to(torch.float32)
    return torch.exp2(k)


def _fp4_round(x: torch.Tensor) -> torch.Tensor:
    """e2m1 round-to-nearest-even (a tie goes to the even code) of |x| <= 6, sign kept."""
    a = x.abs()
    g = FP4_GRID.to(x.device)
    low = (torch.bucketize(a, g, right=True) - 1).clamp(0, 6)
    mid = (g[low] + g[low + 1]) / 2
    up = (a > mid) | ((a == mid) & (low % 2 == 1))
    return torch.copysign(g[low + up.to(low.dtype)], x)


class CpuKernels(types.ModuleType):
    """Drop-in for kernel.py (the tilelang kernels), pure torch on CPU."""

    def __init__(self):
        super().__init__("kernel")

    @staticmethod
    def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
        N = x.size(-1)
        assert N % block_size == 0
        z = x.contiguous().float().view(-1, N // block_size, block_size)
        amax = z.abs().amax(-1, keepdim=True).clamp_min(1e-4)
        if scale_fmt is not None:
            s = _round_scale(amax, 1 / 448.0)
        else:
            s = amax * np.float32(1 / 448.0)
        y = (z / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * s
        if inplace:
            x.copy_(y.view(x.shape).to(x.dtype))
            return x
        raise NotImplementedError("only the inplace fake-quant is needed")

    @staticmethod
    def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
        N = x.size(-1)
        assert N % block_size == 0
        z = x.contiguous().float().view(-1, N // block_size, block_size)
        amax = z.abs().amax(-1, keepdim=True)
        if scale_dtype == torch.float8_e4m3fn:
            amax = amax.clamp_min(6 * 2.0**-9)
            s = (amax / 6.0).to(torch.float8_e4m3fn).float()
        else:
            amax = amax.clamp_min(6 * 2.0**-126)
            s = _round_scale(amax, 1 / 6.0)
        y = _fp4_round((z / s).clamp(-6.0, 6.0)) * s
        if inplace:
            x.copy_(y.view(x.shape).to(x.dtype))
            return x
        raise NotImplementedError("only the inplace fake-quant is needed")

    @staticmethod
    def fp8_gemm(*a, **k):
        raise NotImplementedError("EXL3 weights only: no FP8 weight GEMM")

    fp4_gemm = fp8_gemm

    @staticmethod
    def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
        hc = hc_mult
        m = mixes.float()
        pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
        post = 2 * torch.sigmoid(m[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
        c = (m[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
        c = torch.softmax(c, dim=-1) + eps
        c = c / (c.sum(-2, keepdim=True) + eps)
        for _ in range(sinkhorn_iters - 1):
            c = c / (c.sum(-1, keepdim=True) + eps)
            c = c / (c.sum(-2, keepdim=True) + eps)
        return pre, post, c

    @staticmethod
    def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
        """kernel.py sparse_attn_kernel: online softmax over blocks of 64 gathered keys, probabilities cast to q's
        dtype before P@V, running max starts at -1e30 (a row without a valid key gives zeros), sink added to the
        denominator at the end. q [b,m,h,d], kv [b,n,d], idxs [b,m,topk] int32 (-1 = none)."""
        b, m, h, d = q.shape
        pad = -topk_idxs.size(-1) % 64  # the kernel always works on whole blocks of 64 slots (padding = -1)
        topk_idxs = F.pad(topk_idxs, (0, pad), value=-1)
        topk = topk_idxs.size(-1)
        out = torch.empty_like(q)
        sink = attn_sink.float()
        tile = 128
        for r0 in range(0, m, tile):
            r1 = min(m, r0 + tile)
            qq = q[:, r0:r1].float()  # [b,t,h,d]
            ix = topk_idxs[:, r0:r1].long()  # [b,t,topk]
            T = r1 - r0
            mx = torch.full((b, T, h), -1e30)
            sm = torch.zeros(b, T, h)
            acc = torch.zeros(b, T, h, d)
            for k0 in range(0, topk, 64):
                blk = ix[..., k0 : k0 + 64]
                valid = blk != -1
                g = torch.gather(
                    kv.float().unsqueeze(1).expand(b, T, kv.size(1), d), 2, blk.clamp_min(0).unsqueeze(-1).expand(-1, -1, -1, d)
                )  # [b,T,B,d]
                g = torch.where(valid.unsqueeze(-1), g, torch.zeros(()))
                s = torch.einsum("bthd,btkd->bthk", qq, g)
                s = torch.where(valid.unsqueeze(2), s, torch.full((), -math.inf))
                s = s * np.float32(softmax_scale)
                prev = mx
                mx = torch.maximum(prev, s.amax(-1))
                scale = torch.exp(prev - mx)
                p = torch.exp(s - mx.unsqueeze(-1))
                sm = sm * scale + p.sum(-1)
                pc = p.to(q.dtype).float()
                acc = acc * scale.unsqueeze(-1) + torch.einsum("bthk,btkd->bthd", pc, g)
            sm = sm + torch.exp(sink - mx)
            out[:, r0:r1] = (acc / sm.unsqueeze(-1)).to(q.dtype)
        return out


# --------------------------------------------------------------------------------------------------------------
# Loading the official model.py with the CPU kernels
# --------------------------------------------------------------------------------------------------------------
def load_official(official_dir: Path, quant: str):
    """Import model.py / engram.py from official_dir with stubs; returns (model_module, engram_module)."""
    sys.path.insert(0, str(official_dir))
    k = CpuKernels()
    sys.modules["kernel"] = k
    ip = types.ModuleType("image_processor")
    ip.IMAGE, ip.IMAGE_END, ip.IMAGE_NEW_LINE, ip.IMAGE_START = 0, 1, 2, 3
    sys.modules["image_processor"] = ip
    vi = types.ModuleType("vision")
    vi.Aligner = vi.ViT = type("Stub", (), {})
    sys.modules["vision"] = vi
    import engram as engram_mod  # noqa: E402  (official engram.py; needs sympy)

    src = (official_dir / "model.py").read_text()
    if quant in ("tf", "nofake"):
        n_o = src.count("act_quant(o, fp8_block_size, scale_fmt, scale_dtype, True)")
        n_q = src.count("fp4_act_quant(q, fp4_block_size, True)")
        assert n_o == 2 and n_q == 1, (n_o, n_q)
        src = src.replace("act_quant(o, fp8_block_size, scale_fmt, scale_dtype, True)", "pass")
        if quant == "nofake":
            src = src.replace("fp4_act_quant(q, fp4_block_size, True)", "pass")
    mod = types.ModuleType("model")
    mod.__file__ = str(official_dir / "model.py")
    sys.modules["model"] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod, engram_mod


# --------------------------------------------------------------------------------------------------------------
# Weight sources
# --------------------------------------------------------------------------------------------------------------
class Handler:
    """x [..., K] -> y [..., N]."""

    def __call__(self, x):  # pragma: no cover
        raise NotImplementedError


class DenseHandler(Handler):
    def __init__(self, w):  # w [out, in] float32
        self.w = w

    def __call__(self, x):
        xs = x.reshape(-1, x.size(-1))
        outs = []
        for r in range(0, xs.size(0), ROW_BLOCK):
            blk = xs[r : r + ROW_BLOCK].double()
            pad = ROW_BLOCK - blk.size(0)
            if pad:
                blk = F.pad(blk, (0, 0, 0, pad))
            outs.append((blk @ self.w.double().T)[: ROW_BLOCK - pad])
        return torch.cat(outs).reshape(*x.shape[:-1], -1)


class Exl3Cache:
    def __init__(self, cap_bytes):
        self.cap, self.used, self.d = cap_bytes, 0, {}

    def get(self, k):
        return self.d.get(k)

    def put(self, k, t):
        if self.used + t.numel() * t.element_size() > self.cap:
            self.clear()
        self.d[k] = t
        self.used += t.numel() * t.element_size()

    def clear(self):
        self.d.clear()
        self.used = 0


class Exl3Linear(Handler):
    def __init__(self, src, prefix):
        self.src, self.prefix = src, prefix
        self.name = prefix

    def _meta(self):
        sh, f = self.src.shape_of(self.prefix + ".trellis")
        self.kt, self.nt, w = sh
        self.K, self.N = 16 * self.kt, 16 * self.nt
        self.bits = self.src.fmt.bits_of(sh)
        self.cb = "mul1" if self.src.has(self.prefix + ".mul1") else "3inst"

    def wq(self):
        """W_q [K, N] float16 in the rotated basis (cached)."""
        w = self.src.cache.get(self.prefix)
        if w is None:
            self._meta()
            tr = self.src.tensor(self.prefix + ".trellis")
            w = torch.from_numpy(self.src.fmt.unpack(tr.numpy(), self.bits, self.cb))
            self.src.cache.put(self.prefix, w)
        return w

    def signs(self):
        return (self.src.tensor(self.prefix + ".suh").double(), self.src.tensor(self.prefix + ".svh").double())

    def __call__(self, x):
        suh, svh = self.signs()
        wq = self.wq().double()
        Hm = self.src.H
        xs = x.reshape(-1, x.size(-1)).double() * suh
        K, N = wq.shape
        outs = []
        for r in range(0, xs.size(0), ROW_BLOCK):
            blk = xs[r : r + ROW_BLOCK]
            pad = ROW_BLOCK - blk.size(0)
            if pad:
                blk = F.pad(blk, (0, 0, 0, pad))
            xh = (blk.reshape(ROW_BLOCK, K // 128, 128) @ Hm).reshape(ROW_BLOCK, K)
            y = (xh @ wq).reshape(ROW_BLOCK, N // 128, 128) @ Hm
            outs.append((y.reshape(ROW_BLOCK, N) * svh)[: ROW_BLOCK - pad])
        return torch.cat(outs).reshape(*x.shape[:-1], N)

    def dense(self):
        """W^T [N, K] float64 (the weight of F.linear)."""
        self._meta()
        suh, svh = self.signs()
        tr = self.src.tensor(self.prefix + ".trellis").numpy()
        W = self.src.fmt.dequantize(tr, suh.numpy(), svh.numpy(), self.bits, self.cb)  # [K, N] float64
        return torch.from_numpy(W).T.contiguous()


class Exl3Source:
    """The Mia EXL3 checkpoint (safetensors shards + index)."""

    def __init__(self, mdir: Path, fmt_path: Path, cache_gb: float):
        from safetensors import safe_open

        self._so = safe_open
        self.dir = mdir
        self.wm = json.loads((mdir / "model.safetensors.index.json").read_text())["weight_map"]
        spec = importlib.util.spec_from_file_location("tf_exl3_format", fmt_path)
        self.fmt = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.fmt
        spec.loader.exec_module(self.fmt)
        self.H = torch.from_numpy(self.fmt.hadamard() / np.sqrt(128)).double()
        self.cache = Exl3Cache(int(cache_gb * 2**30))
        self._h = {}
        self._wo = {}
        self.pool = ThreadPoolExecutor(8)

    def _f(self, name):
        sh = self.wm[name]
        if sh not in self._h:
            self._h[sh] = self._so(str(self.dir / sh), framework="pt")
        return self._h[sh]

    def has(self, name):
        return name in self.wm

    def tensor(self, name):
        return self._f(name).get_tensor(name)

    def shape_of(self, name):
        f = self._f(name)
        return tuple(f.get_slice(name).get_shape()), f

    def linear(self, prefix, shape=None):
        return Exl3Linear(self, prefix)

    def plain(self, name, shape=None):
        return self.tensor(name)

    def embed_rows(self, ids):
        f = self._f("embed.weight")
        sl = f.get_slice("embed.weight")
        return torch.stack([sl[int(i) : int(i) + 1][0] for i in ids])

    def wo_a_dense(self, prefix, groups):
        if prefix in self._wo:
            return self._wo[prefix]
        outs = []
        for g in range(groups):
            lin = Exl3Linear(self, f"{prefix}.slice.{g}")
            outs.append(lin.dense())  # [1024, 4096]
        self._wo[prefix] = torch.cat(outs, 0)
        return self._wo[prefix]

    def prefetch(self, prefixes):
        todo = [p for p in prefixes if self.cache.get(p) is None]
        list(self.pool.map(lambda p: Exl3Linear(self, p).wq(), todo))


class SynthSource:
    """Tiny random model for `selftest`: dense linears, random plain tensors."""

    def __init__(self, seed=0):
        self.seed = seed
        self.cache = Exl3Cache(1)

    def _g(self, name):
        h = int(hashlib.sha256(f"{self.seed}/{name}".encode()).hexdigest()[:8], 16)
        return torch.Generator().manual_seed(h)

    def has(self, name):
        return False

    def linear(self, prefix, shape):
        out, inp = shape
        return DenseHandler(torch.randn(out, inp, generator=self._g(prefix)) / math.sqrt(inp))

    def plain(self, name, shape):
        g = self._g(name)
        if name.endswith("norm.weight"):
            return 1 + 0.1 * torch.randn(shape, generator=g)
        if name.endswith("attn_sink"):
            return torch.randn(shape, generator=g)
        if name.endswith(("hc_attn_fn", "hc_ffn_fn")):
            return 0.05 * torch.randn(shape, generator=g)
        if name.endswith(("hc_attn_base", "hc_ffn_base")):
            return 0.5 * torch.randn(shape, generator=g)
        if name.endswith(("hc_attn_scale", "hc_ffn_scale")):
            return 0.3 + 0.1 * torch.rand(shape, generator=g)
        if name.endswith(("gate.bias", "gate.bias_vl")):
            return 0.1 * torch.randn(shape, generator=g)
        if name.endswith(("q_weight", "k_weight")):
            return 1 + 0.1 * torch.randn(shape, generator=g)
        return torch.randn(shape, generator=g) / math.sqrt(shape[-1])

    def embed_rows(self, ids, dim=None):
        return torch.stack([torch.randn(self.dim, generator=self._g(f"embed/{int(i)}")) for i in ids])

    def wo_a_dense(self, prefix, groups, shape=None):
        return torch.randn(shape, generator=self._g(prefix)) / math.sqrt(shape[1])

    def prefetch(self, prefixes):
        pass


class EngramTable:
    """Rows of the official shards 47/48 (FP8 E4M3 [rows,256] then E8M0 [rows,8])."""

    def __init__(self, path: Path, layer_key: str):
        self.path = str(path)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        self.base = 8 + n
        w, s = h[f"{layer_key}.embed.weight"], h[f"{layer_key}.embed.scale"]
        assert w["dtype"] == "F8_E4M3" and s["dtype"] == "F8_E8M0"
        self.rows, self.dim = w["shape"]
        self.woff, self.soff = self.base + w["data_offsets"][0], self.base + s["data_offsets"][0]
        self.sdim = s["shape"][1]
        self.fd = os.open(self.path, os.O_RDONLY)

    def read(self, idx: torch.Tensor):
        flat = idx.reshape(-1).tolist()
        W = torch.empty(len(flat), self.dim, dtype=torch.uint8)
        S = torch.empty(len(flat), self.sdim, dtype=torch.uint8)
        for i, r in enumerate(flat):
            assert 0 <= r < self.rows, (r, self.rows)
            W[i] = torch.frombuffer(bytearray(os.pread(self.fd, self.dim, self.woff + r * self.dim)), dtype=torch.uint8)
            S[i] = torch.frombuffer(bytearray(os.pread(self.fd, self.sdim, self.soff + r * self.sdim)), dtype=torch.uint8)
        return W.reshape(*idx.shape, self.dim), S.reshape(*idx.shape, self.sdim)


class SynthTable:
    def __init__(self, rows, dim, seed):
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randint(0, 0x70, (rows, dim), generator=g, dtype=torch.uint8)
        self.S = torch.randint(120, 130, (rows, dim // 32), generator=g, dtype=torch.uint8)

    def read(self, idx):
        return self.W[idx], self.S[idx]


def engram_embed_forward(table, block_size=32):
    """ParallelEngramEmbedding.forward (world_size 1) with the rows fetched from `table`."""

    def fwd(indices):
        W, S = table.read(indices)
        values = W.view(torch.float8_e4m3fn).float()
        scales = torch.exp2(S.to(torch.float32) - 127.0)
        scales = torch.where(S == 255, torch.full((), float("nan")), scales)
        values = values.unflatten(-1, (-1, block_size)) * scales.unsqueeze(-1)
        return values.flatten(-2).to(torch.bfloat16)

    return fwd


# --------------------------------------------------------------------------------------------------------------
# Model construction (official classes on the meta device, real weights bound afterwards)
# --------------------------------------------------------------------------------------------------------------
FP32_NAMES = ("hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base", "hc_attn_scale", "hc_ffn_scale", "attn_sink",
              "gate.bias", "gate.bias_vl", "gate.weight", "q_weight", "k_weight")


class Env:
    """Everything one run needs: official modules, args, weight source, engram layout."""

    def __init__(self, M, E, args, src, layout, tables, quant):
        self.M, self.E, self.args, self.src, self.layout, self.tables, self.quant = M, E, args, src, layout, tables, quant
        self.handlers: dict[int, Handler] = {}
        orig = M.linear
        handlers = self.handlers

        def linear(x, weight, bias=None):
            ent = handlers.get(id(weight))
            if ent is None or ent[0] is not weight:
                return orig(x, weight, bias)
            h = ent[1]
            out_dtype = torch.float32 if x.dtype == torch.float32 else torch.get_default_dtype()
            return h(x).to(out_dtype)

        M.linear = linear


def make_args(M, cfg: dict, max_seq_len: int):
    d = dict(cfg)
    for k in ("compress_ratios", "kv_source_layers", "index_source_layers", "engram_layer_ids", "engram_num_embeddings",
              "dspark_target_layer_ids"):
        if k in d:
            d[k] = tuple(d[k])
    d["max_batch_size"], d["max_seq_len"] = 1, max_seq_len
    fields = M.ModelArgs.__dataclass_fields__
    return M.ModelArgs(**{k: v for k, v in d.items() if k in fields})


def build_block(env: Env, layer_id: int, flow_dtype: torch.dtype):
    M, args, src = env.M, env.args, env.src
    torch.set_default_dtype(flow_dtype)
    M.world_size, M.rank, M.default_dtype = 1, 0, torch.float8_e4m3fn
    M.precompute_freqs_cis.cache_clear()
    with torch.device("meta"):
        blk = M.Block(layer_id, args, None)
        if env.layout is not None and layer_id in env.layout.layer_ids:
            blk.engram = M.Engram(args, layer_id, env.layout)
    M.precompute_freqs_cis.cache_clear()
    # buffers
    for mod in blk.modules():
        for n, b in list(mod._buffers.items()):
            if b is not None and b.device.type == "meta":
                if n == "score_state":
                    mod._buffers[n] = torch.full(b.shape, -torch.inf, dtype=b.dtype)
                elif n == "freqs_cis":
                    ratio = mod.compress_ratio
                    o, th = (args.original_seq_len, args.compress_rope_theta) if ratio else (0, args.rope_theta)
                    mod._buffers[n] = M.precompute_freqs_cis(
                        mod.rope_head_dim, args.max_seq_len, o, th, args.rope_factor, args.beta_fast, args.beta_slow
                    )
                else:
                    mod._buffers[n] = torch.zeros(b.shape, dtype=b.dtype)
    lin_mods = {n: m for n, m in blk.named_modules() if isinstance(m, M.Linear)}
    for name, mod in lin_mods.items():
        pref = f"layers.{layer_id}.{name}"
        if name == "attn.wo_a":
            w = src.wo_a_dense(pref, args.o_groups, (mod.out_features, mod.in_features)) if isinstance(src, SynthSource) else src.wo_a_dense(pref, args.o_groups)
            mod._parameters["weight"] = torch.nn.Parameter(w.to(flow_dtype), requires_grad=False)
        elif (src.has(pref + ".trellis")) or (isinstance(src, SynthSource) and not name.endswith("weights_proj")):
            env.handlers[id(mod.weight)] = (mod.weight, src.linear(pref, (mod.out_features, mod.in_features)) if isinstance(src, SynthSource) else src.linear(pref))
        else:  # plain weight stored as is (indexer.weights_proj)
            t = src.plain(pref + ".weight", (mod.out_features, mod.in_features))
            mod._parameters["weight"] = torch.nn.Parameter(t.to(flow_dtype), requires_grad=False)
    for pname, p in list(blk.named_parameters()):
        modpath, _, attr = pname.rpartition(".")
        if modpath in lin_mods or pname.startswith("engram.embed."):
            continue  # linears are bound above; the engram table is read row by row from disk
        t = src.plain(f"layers.{layer_id}.{pname}", tuple(p.shape))
        keep32 = any(pname.endswith(s) for s in FP32_NAMES)
        t = t.to(torch.float32 if keep32 else flow_dtype)
        blk.get_submodule(modpath)._parameters[attr] = torch.nn.Parameter(t, requires_grad=False)
    if blk.engram is not None:
        blk.engram.embed.forward = engram_embed_forward(env.tables[layer_id])
    return blk


# --------------------------------------------------------------------------------------------------------------
# Generalised attention (arbitrary chunks) + sublayer-level block forward with taps
# --------------------------------------------------------------------------------------------------------------
def compressor_chunk(M, c, x, s):
    """Compressor.forward for any chunk [s, s+n): returns the pre-RoPE latent of the groups this chunk completes."""
    b, n, _ = x.size()
    ratio, dtype = c.compress_ratio, x.dtype
    if ratio == 1:
        return c.norm(c.wkv(x))
    xf = x.float()
    kv, score = c.wkv(xf), c.wgate(xf)
    r0 = s % ratio
    if r0:
        kv = torch.cat([c.kv_state[:b, :r0], kv], 1)
        score = torch.cat([c.score_state[:b, :r0], score], 1)
    L = kv.size(1)
    ng, rem = L // ratio, L % ratio
    cut = ng * ratio
    if rem:
        c.kv_state[:b, :rem] = kv[:, cut:]
        c.score_state[:b, :rem] = score[:, cut:]
    if ng == 0:
        return None
    kvg = kv[:, :cut].unflatten(1, (ng, ratio))
    sg = score[:, :cut].unflatten(1, (ng, ratio))
    pooled = (kvg * sg.softmax(dim=2)).sum(dim=2)
    return c.norm(pooled.to(dtype))


def attn_chunk(env: Env, attn, x, s, tap=None):
    """Attention.forward generalised to a chunk starting at absolute position `s` (any size)."""
    M = env.M
    tap = tap or (lambda *_: None)
    b, n, _ = x.size()
    E = s + n
    win, rd, ratio = attn.window_size, attn.rope_head_dim, attn.compress_ratio
    freqs = attn.freqs_cis[s:E]
    qr = attn.q_norm(attn.wq_a(x))
    q = attn.wq_b(qr).unflatten(-1, (attn.n_local_heads, attn.head_dim))
    M.apply_rotary_emb(q[..., -rd:], freqs)
    tap("attn.q", q)
    kv = attn.kv_norm(attn.wkv(x))
    M.apply_rotary_emb(kv[..., -rd:], freqs)
    M.act_quant(kv, M.fp8_block_size, M.scale_fmt, M.scale_dtype, True)
    tap("attn.kv_window", kv)
    ring = attn.window_kv_cache[:b].clone()
    kv_all = torch.cat([ring, kv], 1)  # [b, win + n, d]
    t = torch.arange(n).unsqueeze(1)
    pos = (s + t) - (win - 1) + torch.arange(win).unsqueeze(0)  # [n, win] absolute positions, ascending
    widx = torch.where(pos >= s, win + (pos - s), pos % win)
    widx = torch.where(pos < 0, torch.full_like(widx, -1), widx).int()
    idxs = widx.unsqueeze(0).expand(b, -1, -1)
    if ratio:
        total_len = win + n
        clen = E // ratio
        latent = None
        if attn.is_kv_source:
            latent = compressor_chunk(M, attn.compressor, x, s)
            M.shared_attn.compress_kv = attn.compress_kv_cache
        g0 = s // ratio
        if latent is not None:
            tap("attn.latent_pre_rope", latent)
        # index
        if not attn.is_index_source:
            cidx = M.shared_attn.topk_idxs
        elif clen == 0:
            cidx = torch.empty(b, n, 0, dtype=torch.int32)
            M.shared_attn.topk_idxs = cidx
        else:
            cidx = indexer_chunk(env, attn, x, qr, latent, s, total_len, tap)
            M.shared_attn.topk_idxs = cidx
        if latent is not None:
            ng = latent.size(1)
            gf = attn.freqs_cis[g0 * ratio : (g0 + ng) * ratio : ratio]
            M.apply_rotary_emb(latent[..., -rd:], gf)
            M.fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)
            attn.compress_kv_cache[:b, g0 : g0 + ng] = latent
            tap("attn.latent", latent)
        kv_all = torch.cat([kv_all, M.shared_attn.compress_kv[:b, :clen]], 1)
        idxs = torch.cat([idxs, cidx], -1)
    tap("attn.idxs", idxs)
    # ring update (after attention inputs were assembled)
    for j in range(max(s, E - win), E):
        attn.window_kv_cache[:b, j % win] = kv[:, j - s]
    o = M.sparse_attn(q, kv_all, attn.attn_sink, idxs.contiguous(), attn.softmax_scale)
    tap("attn.o_sparse", o)
    M.apply_rotary_emb(o[..., -rd:], freqs, True)
    tap("attn.o_unrot", o)
    o = o.view(b, n, attn.n_local_groups, -1)
    if env.quant == "official":
        M.act_quant(o, M.fp8_block_size, M.scale_fmt, M.scale_dtype, True)
    wo_a = attn.wo_a.weight.view(attn.n_local_groups, attn.o_lora_rank, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    tap("attn.o_lora", o)
    return attn.wo_b(o.flatten(2))


def indexer_chunk(env, attn, x, qr, latent, s, offset, tap):
    M = env.M
    ix = attn.indexer
    b, n, _ = x.size()
    ratio, rd, E = ix.compress_ratio, ix.rope_head_dim, s + n
    if ix.freqs_cis is None:
        ix.freqs_cis = attn.freqs_cis
    if ix.owns_k and latent is not None:
        g0, ng = s // ratio, latent.size(1)
        gf = ix.freqs_cis[g0 * ratio : (g0 + ng) * ratio : ratio]
        k = ix.k_norm(ix.wk(latent))
        M.apply_rotary_emb(k[..., -rd:], gf)
        M.fp4_act_quant(k, M.fp4_block_size, True)
        ix.k_cache[:b, g0 : g0 + ng] = k
        tap("indexer.k", k)
    if ix.owns_k:
        M.shared_attn.index_k = ix.k_cache
    q = ix.wq_b(qr).unflatten(-1, (ix.n_local_heads, ix.index_head_dim))
    M.apply_rotary_emb(q[..., -rd:], ix.freqs_cis[s:E])
    if env.quant != "nofake":
        M.fp4_act_quant(q, M.fp4_block_size, True)
    index_k = M.shared_attn.index_k[:b, : E // ratio]
    weights = ix.weights_proj(x) * (ix.softmax_scale * ix.n_heads**-0.5)
    tap("indexer.q", q)
    tap("indexer.w", weights)
    tap("indexer.index_k", index_k)
    score = torch.einsum("bshd,btd->bsht", q, index_k)
    score = (score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
    clens = (torch.arange(s + 1, E + 1) // ratio).unsqueeze(-1)
    score.masked_fill_(torch.arange(E // ratio) >= clens, -torch.inf)
    tap("indexer.score", score)
    if ix.is_candidate_source or ix.uses_candidates:
        raise NotImplementedError("candidate blocks only exist from layer 20; use the select_candidate_blocks fixture")
    topk = min(ix.index_topk, E // ratio)
    # stable descending sort = ties go to the lower position, independent of the score width (the official
    # topk(sorted=False) leaves ties unspecified, and bf16 scores tie often): needed for chunk-invariant results
    idxs = torch.sort(score, dim=-1, descending=True, stable=True).indices[..., :topk].sort(dim=-1).values
    return torch.where(idxs < clens, idxs + offset, -1).int()


class Taps:
    def __init__(self):
        self.d = {}
        self.prefix = ""

    def __call__(self, name, t):
        if t is None:
            return
        if isinstance(t, torch.Tensor):
            self.d[self.prefix + name] = t.detach().clone().contiguous()


def hc_mixes_tap(env, blk, x, fn, scale, base, tag, tap):
    M = env.M
    xf = x.flatten(2).float()
    rs = torch.rsqrt(xf.square().mean(-1, keepdim=True) + blk.norm_eps)
    mixes = F.linear(xf, fn) * rs
    tap(f"{tag}.mixes", mixes)
    pre, post, comb = M.hc_split_sinkhorn(mixes, scale, base, blk.hc_mult, blk.hc_sinkhorn_iters, blk.hc_eps)
    tap(f"{tag}.pre", pre), tap(f"{tag}.post", post), tap(f"{tag}.comb", comb)
    return pre, post, comb


def moe_tap(env, blk, x, tap):
    ffn = blk.ffn
    shape = x.size()
    xv = x.view(-1, ffn.dim)
    weights, indices = ffn.gate(xv, None)
    tap("ffn.gate_w", weights), tap("ffn.gate_idx", indices.to(torch.int32))
    counts = torch.bincount(indices.flatten(), minlength=ffn.n_routed_experts).tolist()
    used = [i for i, c in enumerate(counts) if c]
    pre = [f"layers.{blk.layer_id}.ffn.experts.{i}.{w}" for i in used for w in ("w1", "w2", "w3")]
    if hasattr(env.src, "prefetch"):
        env.src.prefetch(pre)
    y = torch.zeros_like(xv, dtype=torch.float32)
    for i in used:
        idx, top = torch.where(indices == i)
        y[idx] += ffn.experts[i](xv[idx], weights[idx, top, None])
    tap("ffn.routed_y", y)
    sh = ffn.shared_experts(xv)
    tap("ffn.shared", sh)
    y += sh
    tap("ffn.combine_f32", y)
    return y.type_as(x).view(shape)


def block_step(env, blk, h, s, pre_mix, tap, attn_impl=None):
    """Block.forward with taps at every sublayer boundary; the attention is the chunk-general one."""
    M = env.M
    attn_impl = attn_impl or (lambda a, x, st: attn_chunk(env, a, x, st, tap))
    tap("in.h", h), tap("in.pre_mix", pre_mix)
    residual = h
    a_pre, a_post, a_comb = hc_mixes_tap(env, blk, h, blk.hc_attn_fn, blk.hc_attn_scale, blk.hc_attn_base, "hc_attn", tap)
    x = blk.hc_pre(h, pre_mix)
    tap("hc_pre_attn", x)
    x = blk.attn_norm(x)
    tap("attn_norm", x)
    x = attn_impl(blk.attn, x, s)
    tap("attn.out", x)
    h = blk.hc_post(x, residual, a_post, a_comb)
    tap("hc_post_attn", h)
    residual = h
    f_pre, f_post, f_comb = hc_mixes_tap(env, blk, h, blk.hc_ffn_fn, blk.hc_ffn_scale, blk.hc_ffn_base, "hc_ffn", tap)
    x = blk.hc_pre(h, a_pre)
    tap("hc_pre_ffn", x)
    x = blk.ffn_norm(x)
    tap("ffn_norm", x)
    x = moe_tap(env, blk, x, tap)
    tap("ffn.out", x)
    h = blk.hc_post(x, residual, f_post, f_comb)
    tap("out.h", h), tap("out.pre_mix", f_pre)
    return h, f_pre


def attach_engram_taps(blk, tap):
    e = blk.engram
    wkv_fwd, emb_fwd = e.wkv.forward, e.embed.forward

    def wkv(x):
        y = wkv_fwd(x)
        tap("engram.wkv_in", x), tap("engram.kv", y)
        return y

    def emb(i):
        r = emb_fwd(i)
        tap("engram.hash_ids", i.to(torch.int32)), tap("engram.rows", r)
        return r

    e.wkv.forward, e.embed.forward = wkv, emb


# --------------------------------------------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------------------------------------------
FLOWS = {"bf16": torch.bfloat16, "fp32": torch.float32}


def rel(a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs().max().item()
    return d, d / max(b.abs().max().item(), 1e-30)


def hash_ids_for(env, steps):
    hs = env.E.NgramHashState(env.args, env.layout, None)
    return [hs(torch.tensor([ids]), s) for s, ids in steps]  # each [1, n, n_engram_layers, n_cols]


def run_layers(env, layer_ids, steps, flows, embed_rows, check_official=(0,), log_prefix=""):
    """Layer-major run. Returns {flow: Taps.d}. steps = [(start_pos, ids list), ...]."""
    M = env.M
    hashes = hash_ids_for(env, steps) if env.layout is not None else None
    out = {}
    state = {}
    snap = {fl: {} for fl in flows}  # SharedAttentionRuntime per (flow, step): layer-major order must not mix them
    for fl in flows:
        torch.set_default_dtype(FLOWS[fl])
        hs, pm = [], []
        for si, (s, ids) in enumerate(steps):
            h = embed_rows(ids).unsqueeze(0).to(FLOWS[fl])
            h = h.unsqueeze(2).repeat(1, 1, env.args.hc_mult, 1)
            hs.append(h)
            pm.append(M.make_identity_pre_mix(h, env.args.hc_mult))
        state[fl] = (hs, pm)
        out[fl] = Taps()
    for L in layer_ids:
        for fl in flows:
            tap = out[fl]
            hs, pm = state[fl]
            torch.set_default_dtype(FLOWS[fl])
            blk = build_block(env, L, FLOWS[fl])
            if blk.engram is not None:
                attach_engram_taps(blk, tap)
            for si, (s, ids) in enumerate(steps):
                tap.prefix = f"{fl}/L{L}/s{si}/"
                h = hs[si]
                if blk.engram is not None:
                    hid = hashes[si][:, :, blk.engram.layer_hash_index, :]
                    tap("engram.in", h)
                    h = blk.engram(h, hid, None)
                    tap("engram.out", h)
                h_in, pm_in = h, pm[si]
                for k, v in snap[fl].get(si, {}).items():
                    setattr(M.shared_attn, k, v)
                h2, p2 = block_step(env, blk, h, s, pm[si], tap)
                snap[fl][si] = {k: getattr(M.shared_attn, k) for k in ("compress_kv", "index_k", "topk_idxs", "candidates")}
                if si == 0 and L in check_official:
                    rh, rp = blk.forward(h_in, s, pm_in, None)
                    d, r = rel(h2, rh)
                    log(f"{log_prefix}{fl} L{L}: sublayer-tapped block vs official Block.forward: max|d|={d:.3e} rel={r:.3e}")
                    lim = 1e-4 if fl == "fp32" else 5e-2
                    assert r < lim, (fl, L, r)
                hs[si], pm[si] = h2, p2
            log(f"{log_prefix}{fl} layer {L} done")
        env.src.cache.clear()
    return {fl: out[fl].d for fl in flows}


def attn_state(attn):
    st = {"window_kv_cache": attn.window_kv_cache}
    if attn.is_kv_source:
        st["compress_kv_cache"] = attn.compress_kv_cache
        if attn.compressor.compress_ratio > 1:
            st["kv_state"], st["score_state"] = attn.compressor.kv_state, attn.compressor.score_state
    if attn.is_index_source and attn.indexer.owns_k:
        st["index_k_cache"] = attn.indexer.k_cache
    return st


def valid_state(attn, total):
    """The meaningful part of the caches after `total` tokens."""
    r = attn.compress_ratio
    st = attn_state(attn)
    out = {"window_kv_cache": st["window_kv_cache"].clone()}
    if r:
        n = total // r
        out["compress_kv_cache"] = st["compress_kv_cache"][:, :n].clone()
        if "index_k_cache" in st:
            out["index_k_cache"] = st["index_k_cache"][:, :n].clone()
        if "kv_state" in st:
            rem = total % r
            out["kv_state"] = st["kv_state"][:, :rem].clone()
            out["score_state"] = st["score_state"][:, :rem].clone()
    return out


def bit_equal(a, b):
    return a.shape == b.shape and torch.equal(a.view(torch.int16) if a.dtype == torch.bfloat16 else a, b.view(torch.int16) if b.dtype == torch.bfloat16 else b)


def verify_chunking(env, layer_id, x, flow, chunk_schedules, outdir=None, tag="verify"):
    """x [1,N,dim] = the attention input. Returns (report dict, saved tensors)."""
    M = env.M
    dt = FLOWS[flow]
    torch.set_default_dtype(dt)
    N = x.size(1)
    x = x.to(dt)
    rep, saved = {"N": N, "flow": flow, "layer": layer_id}, {}

    def fresh():
        torch.set_default_dtype(dt)
        blk = build_block(env, layer_id, dt)
        return blk, blk.attn

    # official whole prefill
    _, a_off = fresh()
    o_off = a_off.forward(x, 0)
    st_off = valid_state(a_off, N)
    off_idx = M.shared_attn.topk_idxs.clone() if a_off.compress_ratio else None  # compressed picks, offset N
    # mine whole
    _, a_my = fresh()
    tp = Taps()
    o_my = attn_chunk(env, a_my, x, 0, tp)
    st_my = valid_state(a_my, N)
    d, r = rel(o_my, o_off)
    rep["mine_whole_vs_official_whole_out"] = {"max_abs": d, "rel": r}
    rep["mine_whole_vs_official_whole_state_bitequal"] = {k: bit_equal(st_my[k], st_off[k]) for k in st_off}
    log(f"{tag}: mine-whole vs official-whole out rel={r:.3e} state bitequal={rep['mine_whole_vs_official_whole_state_bitequal']}")
    assert all(rep["mine_whole_vs_official_whole_state_bitequal"].values()), "cache state differs from the official code"
    if flow == "fp32":
        assert r < 1e-4, r  # bf16: indexer score ties may pick another (equally good) key than the official topk
    if off_idx is not None and "indexer.score" in tp.d:
        # do the indexer selections differ from the official topk only by ties? (equal-score keys swapped)
        my_idx = tp.d["attn.idxs"][..., a_my.window_size :]
        sc = tp.d["indexer.score"][0].float()
        diff_rows = tie_rows = 0
        for r in range(N):
            a_ = set(off_idx[0, r][off_idx[0, r] >= 0].tolist())
            a_ = {v - N for v in a_}
            b_ = {v - (a_my.window_size + N) for v in my_idx[0, r][my_idx[0, r] >= 0].tolist()}
            if a_ != b_:
                diff_rows += 1
                sa = torch.sort(sc[r, sorted(a_)]).values
                sb = torch.sort(sc[r, sorted(b_)]).values
                tie_rows += int(a_.__len__() == b_.__len__() and torch.equal(sa, sb))
        rep["indexer_selection_vs_official"] = {"rows": N, "rows_with_different_set": diff_rows, "of_which_equal_score_swaps_only": tie_rows}
        log(f"{tag}: indexer picks differ from the official topk in {diff_rows}/{N} rows, {tie_rows} of them only by equal-score swaps")
        assert tie_rows == diff_rows, "mine and the official indexer disagree beyond ties"
    # official prefill N-3 + 3 decode steps
    _, a_dec = fresh()
    a_dec.forward(x[:, : N - 3], 0)
    outs = [a_dec.forward(x[:, N - 3 + i : N - 2 + i], N - 3 + i) for i in range(3)]
    st_dec = valid_state(a_dec, N)
    rep["official_prefill_Nm3_plus_3_decode_state_bitequal_to_whole"] = {k: bit_equal(st_dec[k], st_off[k]) for k in st_off}
    d, r = rel(torch.cat(outs, 1), o_off[:, N - 3 :])
    rep["official_prefill_Nm3_plus_3_decode_last3_out"] = {"max_abs": d, "rel": r}
    log(f"{tag}: official (N-3)+3 decode: state bitequal={rep['official_prefill_Nm3_plus_3_decode_state_bitequal_to_whole']} last-3 out rel={r:.3e}")
    assert all(rep["official_prefill_Nm3_plus_3_decode_state_bitequal_to_whole"].values()), "prefill(N) != prefill(N-3)+3 decode"
    # mine, chunk schedules
    rep["chunked"] = {}
    ratio = a_my.compress_ratio
    for sched in chunk_schedules:
        assert sum(sched) == N, (sum(sched), N)
        _, a_c = fresh()
        pieces, s = [], 0
        for c in sched:
            pieces.append(attn_chunk(env, a_c, x[:, s : s + c], s))
            s += c
        oc = torch.cat(pieces, 1)
        stc = valid_state(a_c, N)
        out_eq = bit_equal(oc, o_my)
        st_eq = {k: bit_equal(stc[k], st_my[k]) for k in st_my}
        aligned = all(c % max(ratio, 1) == 0 for c in sched[:-1])
        d, r = rel(oc, o_my)
        rep["chunked"][",".join(map(str, sched[:6])) + ("..." if len(sched) > 6 else "")] = {
            "ratio_aligned": aligned, "out_bitequal_to_whole": out_eq, "out_rel": r, "state_bitequal": st_eq}
        log(f"{tag}: chunks {sched[:6]}{'...' if len(sched) > 6 else ''} aligned={aligned} out bitequal={out_eq} rel={r:.2e} state={st_eq}")
        if aligned:
            assert all(st_eq.values()), "chunked cache state != whole (aligned chunks)"
            # bf16 flow: bitwise. fp32 flow: float32 matmuls are not batch-shape invariant (BLAS blocking), so ~1e-7
            assert out_eq if flow == "bf16" else r < 1e-6, "chunked output != whole, aligned chunks"
        else:
            assert r < (1e-5 if flow == "fp32" else 1e-2), r
    saved.update({f"state_whole/{k}": v for k, v in st_my.items()})
    saved["out_whole"], saved["out_official_whole"], saved["x"] = o_my, o_off, x
    saved["out_official_decode_last3"] = torch.cat(outs, 1)
    return rep, saved


# --------------------------------------------------------------------------------------------------------------
# Real-model setup and commands
# --------------------------------------------------------------------------------------------------------------
class TokShim:
    """The slice of a transformers fast tokenizer that engram.build_compressed_token_map uses."""

    def __init__(self, tok):
        self.backend_tokenizer = tok
        self._n = tok.get_vocab_size(with_added_tokens=True)

    def __len__(self):
        return self._n


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def setup_real(a):
    from tokenizers import Tokenizer

    official, mia = Path(a.official), Path(a.mia)
    cfg = json.loads((official / "config.json").read_text())
    M, E = load_official(official, a.quant)
    args = make_args(M, cfg, a.max_seq_len)
    src = Exl3Source(mia, Path(a.exl3_format), a.cache_gb)
    layout = E.EngramLayout.from_args(args)
    tables = {}
    for lid, shard in zip(layout.layer_ids, a.engram_shards):
        tables[lid] = EngramTable(Path(a.engram_dir) / shard, f"layers.{lid}.engram")
        assert tables[lid].rows == layout.num_embeddings[layout.layer_ids.index(lid)], "engram rows != config"
    tok = Tokenizer.from_file(str(mia / "tokenizer.json"))
    # the compressed token map (engram.build_compressed_token_map) once, shared by every NgramHashState
    t0 = time.time()
    lookup, n = E.build_compressed_token_map(TokShim(tok))
    log(f"engram token map: {len(lookup)} ids -> {n} compressed ({time.time() - t0:.0f}s)")
    assert n == args.engram_compressed_vocab_size
    E.build_compressed_token_map = lambda _t: (lookup, n)
    env = Env(M, E, args, src, layout, tables, a.quant)
    env.tok, env.token_map = tok, lookup
    return env


def encode(env, text, n):
    ids = [0] + env.tok.encode(text, add_special_tokens=False).ids
    assert len(ids) >= n, (len(ids), n)
    return ids[:n]


def save_st(tensors, path, meta):
    from safetensors.torch import save_file

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().clone().contiguous() for k, v in tensors.items()}, str(path), metadata={k: str(v) for k, v in meta.items()})
    log(f"wrote {path} ({os.path.getsize(path) / 1e6:.2f} MB, {len(tensors)} tensors)")


DEFAULT_TEXT = (
    "The capital of France is Paris, and the capital of Germany is Berlin. In 1969, two astronauts walked on the "
    "surface of the Moon. def fibonacci(n):\n    return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)\n"
)


def cmd_layers(a):
    env = setup_real(a)
    text = Path(a.text_file).read_text() if a.text_file else DEFAULT_TEXT
    ids = encode(env, text, 18)
    steps = [(0, ids[:16]), (16, ids[16:17]), (17, ids[17:18])]
    log("ids", ids)
    src = env.src
    res = run_layers(env, [0, 1, 2], steps, list(FLOWS) if a.flows == "both" else [a.flows], src.embed_rows, check_official=(0, 1, 2))
    meta = {"generator": "tools/dsv41/fixtures.py layers", "quant": a.quant, "steps": json.dumps([(s, i) for s, i in steps]),
            "mia_snapshot": a.mia, "official": "deepseek-ai/DeepSeek-V4.1-Flash-1017 inference/{model,engram}.py (MIT)",
            "kernels": "pure-torch CPU emulation of kernel.py, not bit-checked against GPU", "exl3": "TensorFold format.py (MIT) float64"}
    out = Path(a.out)
    hid = env.E.NgramHashState(env.args, env.layout, None)
    extra = {}
    for si, (s, st) in enumerate(steps):
        extra[f"hash_ids/s{si}"] = hid(torch.tensor([st]), s).to(torch.int32)
    extra["ids"] = torch.tensor(ids, dtype=torch.int32)
    extra["token_map"] = torch.tensor(env.token_map, dtype=torch.int32)
    extra["engram/primes"] = hid.primes.to(torch.int64)
    extra["engram/offsets"] = hid.offsets.to(torch.int64)
    extra["engram/multipliers"] = hid.multipliers.to(torch.int64)
    extra["engram/pad_id_compressed"] = torch.tensor(int(hid.pad_id))
    # router with the vision bias (bias_vl) on layer 2, from the real ffn_norm output
    torch.set_default_dtype(torch.bfloat16)
    for fl, d in res.items():
        d.update(extra if fl == list(res)[0] else {})
    for fl, d in res.items():
        save_st(d, out / f"layers_{fl}.safetensors", {**meta, "flow": fl})
    gate = {}
    for lid in (2,):
        for n in ("weight", "bias", "bias_vl"):
            gate[f"layers.{lid}.ffn.gate.{n}"] = src.plain(f"layers.{lid}.ffn.gate.{n}")
    xg = res[list(res)[0]][f"{list(res)[0]}/L2/s0/ffn_norm"].view(-1, env.args.dim)
    blk2 = build_block(env, 2, FLOWS[list(res)[0]])
    mask = torch.zeros(xg.size(0), dtype=torch.bool)
    mask[4:11] = True
    w, ix = blk2.ffn.gate(xg, mask)
    gate.update({"x": xg, "image_mask": mask, "weights_vl": w, "indices_vl": ix.to(torch.int32)})
    save_st(gate, out / "router_gate_L2.safetensors", {**meta, "note": "gate with image_mask (bias_vl) on rows 4..10"})
    summ = {}
    for fl, d in res.items():
        for k, v in d.items():
            if k.endswith("/out.h") and "/s0/" in k:
                summ[f"{fl}:{k}"] = {"norm": float(v.float().norm()), "absmax": float(v.float().abs().max()), "has_nan": bool(torch.isnan(v.float()).any())}
    (out / "layers_summary.json").write_text(json.dumps(summ, indent=1))
    log(json.dumps(summ, indent=1))


def cmd_attn4k(a):
    env = setup_real(a)
    n = a.tokens
    text = Path(a.text_file).read_text()
    ids = encode(env, text, n)
    flow = a.flows if a.flows != "both" else "bf16"
    torch.set_default_dtype(FLOWS[flow])
    blk = build_block(env, 2, FLOWS[flow])
    emb = env.src.embed_rows(ids).unsqueeze(0).to(FLOWS[flow])
    x = blk.attn_norm(emb)  # attention input of layer 2 (embedding through the norm; the stream is not run through layers 0-1)
    sched = [[1024] * (n // 1024) + [n % 1024]] if n % 1024 else [[1024] * (n // 1024)]
    sched = [s for s in sched]
    sched.append([512] * (n // 512) + ([n % 512] if n % 512 else []))
    sched.append([1000] * (n // 1000) + ([n % 1000] if n % 1000 else []))
    sched.append([777] * (n // 777) + ([n % 777] if n % 777 else []))
    rep, saved = verify_chunking(env, 2, x, flow, sched, tag="attn4k")
    # compact sparse-attention fixture from a tapped whole run
    tap = Taps()
    torch.set_default_dtype(FLOWS[flow])
    blk = build_block(env, 2, FLOWS[flow])
    attn_chunk(env, blk.attn, x, 0, tap)
    d = tap.d
    win = blk.attn.window_size
    clen = n // blk.attn.compress_ratio
    kv_all = torch.cat([torch.zeros(1, win, 512, dtype=FLOWS[flow]), d["attn.kv_window"], blk.attn.compress_kv_cache[:, :clen]], 1)
    rows = [r for r in (0, 1, 2, 127, 128, 129, 1023, 2047, n - 2, n - 1) if r < n]
    q, idx, o = d["attn.q"][:, rows], d["attn.idxs"][:, rows], d["attn.o_sparse"][:, rows]
    u = torch.unique(idx[idx >= 0])
    remap = torch.full((kv_all.size(1),), -1, dtype=torch.int32)
    remap[u] = torch.arange(u.numel(), dtype=torch.int32)
    idx_c = torch.where(idx >= 0, remap[idx.clamp_min(0).long()], torch.full_like(idx, -1))
    # layout-free form of the same selection: absolute window positions and compressed-block numbers per row
    wp = torch.where(idx[..., :win] >= 0, idx[..., :win] - win, torch.full_like(idx[..., :win], -1))
    cp = torch.where(idx[..., win:] >= 0, idx[..., win:] - (win + n), torch.full_like(idx[..., win:], -1))
    fx = {"q": q[0], "kv": kv_all[0, u], "idxs": idx_c[0], "sink": blk.attn.attn_sink.detach(), "o": o[0],
          "rows": torch.tensor(rows, dtype=torch.int32), "window_n_idx": torch.tensor(win),
          "window_positions": wp[0], "compressed_positions": cp[0], "kv_rows_in_kv_all": u.to(torch.int32)}
    stats = {"rows": rows, "n_kv_rows": int(u.numel()), "minus1_per_row": [int((idx_c[0, i] < 0).sum()) for i in range(len(rows))],
             "topk_width": int(idx.size(-1)), "compressed_positions": clen, "ratio": blk.attn.compress_ratio,
             "scale": blk.attn.softmax_scale, "quant": a.quant, "flow": flow}
    meta = {"generator": "tools/dsv41/fixtures.py attn4k", **{k: json.dumps(v) for k, v in stats.items()}}
    out = Path(a.out)
    save_st(fx, out / f"sparse_attn_4k_{flow}.safetensors", meta)
    big = {k: v for k, v in saved.items()}
    save_st(big, out / f"attn4k_L2_{flow}_full.safetensors", {"generator": "attn4k", "report": json.dumps(rep)})
    (out / f"attn4k_L2_{flow}_report.json").write_text(json.dumps({"verify": rep, "stats": stats}, indent=1))
    ixs = d.get("indexer.score")
    if ixs is not None:
        sel = d["attn.idxs"][0, rows, win:]
        stats["indexer_selected_per_row"] = [int((sel[i] >= 0).sum()) for i in range(len(rows))]


def cmd_indexer4k(a):
    """Indexer fixture for the cpu reference's top-k parity: real q / weights / index keys / bf16 scores of layer 2 on the
    ~4K-token prompt, the official `topk(sorted=False)` picks and the stable lowest-index-wins picks for ~48 query rows
    (the rows where the two differ first). Run: uv run tools/dsv41/fixtures.py indexer4k --out DIR"""
    env = setup_real(a)
    M, n = env.M, a.tokens
    ids = encode(env, Path(a.text_file).read_text(), n)
    dt = FLOWS["bf16"]
    torch.set_default_dtype(dt)
    blk = build_block(env, 2, dt)
    x = blk.attn_norm(env.src.embed_rows(ids).unsqueeze(0).to(dt))
    blk_o = build_block(env, 2, dt)
    blk_o.attn.forward(x, 0)
    off = M.shared_attn.topk_idxs.clone()[0]  # [N, topk], offset N, -1 = none; sorted by position
    tap = Taps()
    blk_m = build_block(env, 2, dt)
    attn_chunk(env, blk_m.attn, x, 0, tap)
    d = tap.d
    win = blk_m.attn.window_size
    mine = d["attn.idxs"][0, :, win:]
    clen = n // blk_m.attn.compress_ratio
    off_c = torch.where(off >= 0, off - n, torch.full_like(off, -1))
    mine_c = torch.where(mine >= 0, mine - (win + n), torch.full_like(mine, -1))
    tie, other = [], []
    for r in range(1100, n):
        (other if torch.equal(off_c[r], mine_c[r]) else tie).append(r)
    rows = sorted(tie[:32] + other[:: max(1, len(other) // 16)][:16])
    fx = {"rows": torch.tensor(rows, dtype=torch.int32), "q": d["indexer.q"][0, rows], "w": d["indexer.w"][0, rows],
          "k": d["indexer.index_k"][0], "score": d["indexer.score"][0, rows], "official_idx": off_c[rows].to(torch.int32),
          "stable_idx": mine_c[rows].to(torch.int32)}
    stats = {"rows_total": n, "rows_where_official_differs_from_stable": len(tie), "compressed_rows": clen,
             "score_dtype": str(d["indexer.score"].dtype), "weights_dtype": str(d["indexer.w"].dtype), "q_dtype": str(d["indexer.q"].dtype),
             "scale": blk_m.attn.indexer.softmax_scale * blk_m.attn.indexer.n_heads ** -0.5}
    log(f"indexer4k: {stats}")
    save_st(fx, Path(a.out) / "indexer_4k_bf16.safetensors", {"generator": "tools/dsv41/fixtures.py indexer4k", **{k: json.dumps(v) for k, v in stats.items()}})


def cmd_ops(a):
    """Synthetic op fixtures: no weights needed."""
    out = Path(a.out)
    official, cfg = Path(a.official), None
    cfg = json.loads((official / "config.json").read_text())
    M, E = load_official(official, a.quant)
    g = torch.Generator().manual_seed(20261031)
    t = {}
    # fake-quant ops (CPU emulation of kernel.py)
    x = (torch.randn(8, 512, generator=g) * torch.exp(torch.randn(8, 1, generator=g) * 2)).to(torch.bfloat16)
    x[0, :32] = 0
    x[1, 5] = 3e4
    for name, fn in (("act_quant_fp8_b32_ue8m0", lambda z: M.act_quant(z, 32, "ue8m0", torch.float8_e8m0fnu, True)),
                     ("fp4_quant_b32_e8m0", lambda z: M.fp4_act_quant(z, 32, True)),
                     ("fp4_quant_b16_e4m3", lambda z: M.fp4_act_quant(z, 16, True, scale_dtype=torch.float8_e4m3fn))):
        z = x.clone()
        fn(z)
        t[f"{name}/in"], t[f"{name}/out"] = x, z
    # Sinkhorn / hc
    mixes = torch.randn(32, 24, generator=g)
    sc, base = torch.tensor([0.3, 0.2, 0.5]), torch.randn(24, generator=g) * 0.5
    pre, post, comb = M.hc_split_sinkhorn(mixes, sc, base, cfg["hc_mult"], cfg["hc_sinkhorn_iters"], cfg["hc_eps"])
    t.update({"sinkhorn/mixes": mixes, "sinkhorn/scale": sc, "sinkhorn/base": base, "sinkhorn/pre": pre, "sinkhorn/post": post, "sinkhorn/comb": comb})
    xs = torch.randn(1, 4, 4, cfg["dim"], generator=g).to(torch.bfloat16)
    pm = torch.softmax(torch.randn(1, 4, 4, generator=g), -1)
    res = torch.randn(1, 4, 4, cfg["dim"], generator=g).to(torch.bfloat16)
    ys = M.Block.hc_pre(None, xs, pm)
    sub = torch.randn(1, 4, cfg["dim"], generator=g).to(torch.bfloat16)
    yp = M.Block.hc_post(None, sub, res, post[:4].view(1, 4, 4), comb[:4].view(1, 4, 4, 4))
    t.update({"hc_pre/x": xs, "hc_pre/pre_mix": pm, "hc_pre/out": ys, "hc_post/x": sub, "hc_post/residual": res,
              "hc_post/post": post[:4].view(1, 4, 4), "hc_post/comb": comb[:4].view(1, 4, 4, 4), "hc_post/out": yp})
    # candidate blocks
    logits = torch.randn(4, 20000, generator=g)
    lens = torch.tensor([20000, 15000, 3000, 9]).unsqueeze(-1)
    logits = logits.masked_fill(torch.arange(20000) >= lens, -torch.inf)
    keep = M.select_candidate_blocks(logits, lens, cfg["candidate_topk_blocks"], cfg["candidate_block_size"])
    t.update({"candidates/logits": logits, "candidates/compress_lens": lens.to(torch.int32), "candidates/keep": keep})
    # sparse attention incl. -1 and an all-masked row
    h, d = cfg["n_heads"], cfg["head_dim"]
    q = torch.randn(1, 4, h, d, generator=g).to(torch.bfloat16)
    kv = torch.randn(1, 40, d, generator=g).to(torch.bfloat16)
    sink = torch.randn(h, generator=g)
    idx = torch.full((1, 4, 70), -1, dtype=torch.int32)
    idx[0, 0, :5] = torch.arange(5)
    idx[0, 2, 3:68] = torch.randint(0, 40, (65,), generator=g, dtype=torch.int32)  # crosses the 64-key block
    idx[0, 3, 60:70] = torch.arange(10)
    o = M.sparse_attn(q, kv, sink, idx, d**-0.5)
    assert o[0, 1].abs().max() == 0, "all-masked row must be zero"
    t.update({"sparse_attn/q": q, "sparse_attn/kv": kv, "sparse_attn/sink": sink, "sparse_attn/idxs": idx, "sparse_attn/out": o})
    # rope / yarn tables
    pos = torch.tensor(list(range(64)) + [1000, 4095, 4096, 65535, 65536, 65537, 131071, 262143, 524287, 1048575])
    for nm, orig, theta in (("plain", 0, cfg["rope_theta"]), ("compressed_yarn", cfg["original_seq_len"], cfg["compress_rope_theta"])):
        fc = M.precompute_freqs_cis(cfg["rope_head_dim"], 1048576, orig, theta, cfg["rope_factor"], cfg["beta_fast"], cfg["beta_slow"])
        t[f"rope_{nm}/positions"] = pos.to(torch.int32)
        t[f"rope_{nm}/cos"], t[f"rope_{nm}/sin"] = fc[pos].real.contiguous(), fc[pos].imag.contiguous()
        xr = torch.randn(1, 4, 2, cfg["rope_head_dim"], generator=g).to(torch.bfloat16)
        fq = fc[torch.tensor([5, 6, 7, 70000])].view(1, 4, cfg["rope_head_dim"] // 2)
        y = M.apply_rotary_emb(xr.clone(), fq)
        yi = M.apply_rotary_emb(y.clone(), fq, True)
        t[f"rope_{nm}/x"], t[f"rope_{nm}/y"], t[f"rope_{nm}/y_inverse"] = xr, y, yi
        t[f"rope_{nm}/positions_x"] = torch.tensor([5, 6, 7, 70000], dtype=torch.int32)
    save_st(t, out / "ops.safetensors", {"generator": "tools/dsv41/fixtures.py ops", "note": "synthetic inputs; outputs by the official torch functions / CPU kernel emulation"})


def cmd_ci(a):
    """Cut the small committed subset (golden/dsv41/) out of the big generator outputs."""
    from safetensors import safe_open

    src, dst = Path(a.out), Path(a.dir)
    cap = int(a.budget_mb * 1e6)
    total = 0
    for fl in ("bf16", "fp32"):
        f = safe_open(str(src / f"layers_{fl}.safetensors"), framework="pt")
        keep = {}
        for k in f.keys():
            sl = f.get_slice(k)
            nbytes = int(np.prod(sl.get_shape())) * {"BF16": 2, "F32": 4, "I32": 4, "I64": 8, "BOOL": 1, "F16": 2}[sl.get_dtype()]
            if not k.startswith(fl + "/"):
                keep[k] = nbytes if fl == "bf16" and (k in ("ids", "token_map") or k.startswith(("hash_ids", "engram/"))) else None
                continue
            _, L, st, name = k.split("/", 3)
            big = nbytes > (700_000 if fl == "bf16" else 1_300_000)
            if fl == "bf16":
                ok = (L == "L2" and st == "s0" and not big) or (L == "L1" and st == "s0" and name.startswith("engram.")) \
                    or name in ("out.h", "ffn.gate_idx", "ffn.gate_w") or (st != "s0" and L == "L2" and not big)
                ok = ok and not (name == "out.h" and big)
            else:
                ok = (L in ("L1", "L2") and st == "s0" and name in ("out.h", "attn.out", "ffn.out", "attn.o_sparse", "engram.out", "hc_attn.pre", "hc_attn.post", "hc_attn.comb", "hc_ffn.pre", "hc_ffn.post", "hc_ffn.comb", "ffn.gate_idx", "ffn.gate_w"))
                ok = ok and nbytes <= 1_400_000
            keep[k] = nbytes if ok else None
        sel = {k: f.get_tensor(k) for k, v in keep.items() if v is not None}
        sz = sum(t.numel() * t.element_size() for t in sel.values())
        total += sz
        log(f"{fl}: {len(sel)} tensors, {sz / 1e6:.2f} MB")
        save_st(sel, dst / f"layers_{fl}_ci.safetensors", {"note": "subset of layers_*.safetensors; full file sha256 in manifest.json (spark1:~/dsv41-fixtures/out)"})
    log(f"total {total / 1e6:.2f} MB (cap {cap / 1e6:.0f} MB)")
    assert total <= cap


def cmd_manifest(a):
    d = Path(a.dir)
    rows = {str(p.relative_to(d)): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(d.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    (d / "manifest.json").write_text(json.dumps(rows, indent=1))
    log(f"manifest of {len(rows)} files")


# --------------------------------------------------------------------------------------------------------------
# selftest: tiny synthetic model
# --------------------------------------------------------------------------------------------------------------
def synth_cfg():
    return dict(vocab_size=1000, dim=256, moe_inter_dim=128, n_layers=4, n_mtp_layers=0, n_heads=8, n_routed_experts=16,
                n_shared_experts=1, n_activated_experts=2, score_func="sqrtsoftplus", route_scale=1.5, swiglu_limit=10.0,
                q_lora_rank=64, head_dim=64, rope_head_dim=16, norm_eps=1e-20, o_groups=4, o_lora_rank=32, window_size=16,
                compress_ratios=[0, 0, 2, 2, 0], kv_source_layers=[2], index_source_layers=[2], original_seq_len=64,
                rope_theta=10000, rope_factor=16, beta_fast=32, beta_slow=1, compress_rope_theta=160000, index_n_heads=4,
                index_head_dim=32, index_topk=8, candidate_source_layer=-1, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
                engram_layer_ids=[1], engram_vocab_size=1000, engram_max_ngram_size=4, engram_pad_id=2,
                engram_n_heads=2, engram_head_dim=32, engram_compressed_vocab_size=500, engram_num_embeddings=[60000],
                vision_n_layers=0)


def cmd_selftest(a):
    torch.manual_seed(0)
    # 1. kernel emulation sanity
    K = CpuKernels
    g = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, -0.75, -5.0, 7.0])
    r = _fp4_round(g.clamp(-6, 6))
    assert r.tolist() == [0, 0, 0.5, 1, 1, 1, 1.5, 2, 2, 2, 3, 4, 4, 4, 6, -1, -4, 6], r.tolist()
    pre, post, comb = K.hc_split_sinkhorn(torch.randn(5, 24), torch.tensor([0.3, 0.2, 0.5]), torch.randn(24), 4, 20, 1e-6)
    assert torch.allclose(comb.sum(-1), torch.ones(5, 4), atol=1e-4) and torch.allclose(comb.sum(-2), torch.ones(5, 4), atol=1e-5)
    q = torch.randn(1, 3, 4, 32)
    kv = torch.randn(1, 20, 32)
    sink = torch.randn(4)
    idx = torch.randint(-1, 20, (1, 3, 70), dtype=torch.int32)
    idx[0, 1] = -1
    o = K.sparse_attn(q, kv, sink, idx, 32**-0.5)
    assert o[0, 1].abs().max() == 0
    for m in range(3):
        if m == 1:
            continue
        v = idx[0, m][idx[0, m] >= 0].long()
        s = (q[0, m] @ kv[0, v].T) * 32**-0.5
        w = torch.exp(torch.cat([s, sink[:, None]], 1)).float()
        ref = (w[:, :-1] / w.sum(1, keepdim=True)) @ kv[0, v]
        assert torch.allclose(o[0, m], ref, atol=1e-5), (o[0, m] - ref).abs().max()
    log("kernel emulation sanity ok")
    # 2. EXL3 forward == dense dequantisation (needs the TensorFold format.py)
    fp = Path(a.exl3_format)
    if fp.exists():
        spec = importlib.util.spec_from_file_location("tf_fmt", fp)
        fmt = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = fmt
        spec.loader.exec_module(fmt)
        rng = np.random.default_rng(1)
        for bits, cb in ((2, "mul1"), (3, "3inst"), (2.5, "mul1")):
            Kd, Nd = 256, 128
            tr = rng.integers(-32768, 32768, size=(Kd // 16, Nd // 16, int(16 * bits)), dtype=np.int16)
            suh = (rng.uniform(0.5, 1.5, Kd) * rng.choice([-1, 1], Kd) / np.sqrt(Kd)).astype(np.float16)
            svh = (rng.uniform(0.5, 1.5, Nd) * rng.choice([-1, 1], Nd)).astype(np.float16)
            xx = rng.standard_normal((5, Kd))
            ref = fmt.forward(xx, tr, suh, svh, bits, cb)
            Wq = torch.from_numpy(fmt.unpack(tr, bits, cb)).double()
            Hm = torch.from_numpy(fmt.hadamard() / np.sqrt(128)).double()
            xh = (torch.from_numpy(xx) * torch.from_numpy(suh.astype(np.float64))).reshape(5, Kd // 128, 128) @ Hm
            y = ((xh.reshape(5, Kd) @ Wq).reshape(5, Nd // 128, 128) @ Hm).reshape(5, Nd) * torch.from_numpy(svh.astype(np.float64))
            assert np.allclose(y.numpy(), ref, rtol=1e-12, atol=1e-12)
        log("EXL3 rotated forward == TensorFold forward ok")
    # 3. whole synthetic model through the harness, both flows, with checks against the official block
    M, E = load_official(Path(a.official), a.quant)
    cfg = synth_cfg()
    args = make_args(M, cfg, 256)
    src = SynthSource(1)
    src.dim = cfg["dim"]
    E.build_compressed_token_map = lambda _t: ([i % 500 for i in range(1000)], 500)
    layout = E.EngramLayout.from_args(args)
    tables = {1: SynthTable(60000, 32, 7)}
    env = Env(M, E, args, src, layout, tables, a.quant)
    ids = [int(v) for v in torch.randint(0, 1000, (30,))]
    steps = [(0, ids[:25]), (25, ids[25:26]), (26, ids[26:27])]
    res = run_layers(env, [0, 1, 2, 3], steps, ["bf16", "fp32"], lambda ii: src.embed_rows(ii), check_official=(0, 1, 2))
    for fl, d in res.items():
        h = d[f"{fl}/L3/s0/out.h"].float()
        assert torch.isfinite(h).all() and h.abs().max() > 0
    # 4. chunking
    for fl in ("fp32", "bf16"):
        torch.set_default_dtype(FLOWS[fl])
        x = torch.randn(1, 63, cfg["dim"]).to(FLOWS[fl])
        rep, _ = verify_chunking(env, 2, x, fl, [[8, 8, 8, 8, 8, 8, 8, 7], [32, 31], [63], [5, 7, 9, 11, 13, 18]], tag=f"selftest/{fl}")
        assert rep["chunked"]["32,31"]["out_bitequal_to_whole"]
    log("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["selftest", "layers", "attn4k", "indexer4k", "ops", "manifest", "ci"])
    ap.add_argument("dir", nargs="?")
    home = Path.home()
    base = home / "dsv41-fixtures"
    ap.add_argument("--out", default=str(base / "out"))
    ap.add_argument("--official", default=str(base / "in/official"))
    ap.add_argument("--mia", default=str(home / ".cache/huggingface/hub/models--Mia-AiLab--DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw/snapshots/45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b"))
    ap.add_argument("--engram-dir", default=str(home / ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/" ) )
    ap.add_argument("--engram-shards", nargs=2, default=["model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors"])
    ap.add_argument("--exl3-format", default=str(base / "in/tf_exl3_format.py"))
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--tokens", type=int, default=4095)
    ap.add_argument("--flows", default="both", choices=["both", "bf16", "fp32"])
    ap.add_argument("--quant", default="tf", choices=["tf", "nofake", "official"])
    ap.add_argument("--budget-mb", type=float, default=15)
    ap.add_argument("--cache-gb", type=float, default=16)
    ap.add_argument("--max-seq-len", type=int, default=4352)
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 8)
    if a.cmd == "ci":
        return cmd_ci(a)
    if a.cmd == "manifest":
        a.dir = a.dir or a.out
        return cmd_manifest(a)
    if a.cmd in ("layers", "attn4k", "indexer4k") and Path(a.engram_dir).is_dir() and not any(Path(a.engram_dir, s).exists() for s in a.engram_shards):
        sub = sorted(Path(a.engram_dir).glob("*"))
        if sub:
            a.engram_dir = str(sub[0])
    {"selftest": cmd_selftest, "layers": cmd_layers, "attn4k": cmd_attn4k, "indexer4k": cmd_indexer4k, "ops": cmd_ops}[a.cmd](a)


if __name__ == "__main__":
    main()
