#!/usr/bin/env python3
"""Inkling-2-Mini DSpark drafter (`mtp.safetensors`) oracle and acceptance measurement.

Numerics follow vLLM's `models/inkling2/nvidia/dspark.py`, `ops/dspark.py` and `speculator.py`
(`.worktrees/ink-official/`), with bf16 rounding at vLLM's storage points and fp32 math in between.

Modes (run with `uv run python inkling2_dspark_ref.py <mode>` from `tools/`):

  taps     run the target oracle (`inkling2_ref.py`, W4A16, rounding none) over each sequence, chunked, and
           save per position: the 4 DSpark taps (partial entering visits 5, 23, 41 and the final-norm output)
           as bf16, the target argmax and top-1 probability, and the top-8 expert ids of every visit.
  draft    for every anchor position t of each taps file, run the drafter exactly as vLLM does on a decode step
           (context K/V from the taps of positions < t, block = anchor + 7 masks at positions t..t+7,
           f_rnn from the taps of t-1, 7 sequential Markov steps) and save the 7 draft tokens and their
           top-1 probabilities.
  analyze  acceptance by draft position, accepted tokens per verify step for fixed k and confidence-gated k,
           expert union U(R) from the real routing, and the projected speedup from the Metal kernel timings.

Teacher forcing: sequences are fixed token streams (vLLM's greedy outputs on the greedy golden prompts,
and two human prose texts). At anchor t the target's next tokens are g_j = argmax at position t+j-1.
A draft prefix of length m counts as accepted when d_j == g_j for j <= m, and the text agrees with the
target (text[t+j] == g_j) for j < m, so every conditional is the one a real greedy verify would see.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inkling2_ref as ir  # noqa: E402

torch.set_grad_enabled(False)

TAP_VISITS = (5, 23, 41)  # partial entering these visits; tap 3 = final-norm output (target_layer_ids 48)


def rss(tag: str):
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS"):
                print(f"[rss] {tag}: {line.split()[1] / 2**20 if False else int(line.split()[1]) / 2**20:.1f} GB", flush=True)
    except OSError:
        pass


def bf(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(torch.float32)


# --------------------------------------------------------------------------------------
# sequences


def sequences(which: str, golden: Path) -> list[dict]:
    out = []
    if which in ("greedy", "all"):
        d = json.loads((golden / "inkling2-mini-vllm-greedy.json").read_text())
        for r in d["results"]:
            eff = r["reasoning_effort"]
            name = r["name"] + ("" if eff is None else f"_e{eff}")
            out.append({"name": name, "ids": r["prompt_ids"] + r["output_ids"], "start": len(r["prompt_ids"]),
                        "kind": "greedy"})
    if which in ("prose", "all"):
        f = safe_open(str(golden / "inkling2-mini-vllm-long-top20.safetensors"), "pt")
        names = [p.get("name") for p in json.loads(f.metadata()["prompts"])]
        for j, n in enumerate(names):
            if n.startswith("long_prose"):
                ids = f.get_tensor(f"ids_{j}").long().tolist()
                out.append({"name": n, "ids": ids, "start": 16, "kind": "prose"})
    return out


# --------------------------------------------------------------------------------------
# taps


class TapDump(dict):
    """`Ref.dump` that keeps only the DSpark taps and the routing."""

    keep = {f"v{v:02d}.partial_in" for v in TAP_VISITS} | {"final_hidden"}

    def __setitem__(self, k, v):
        if k in self.keep or k.endswith(".expert_ids"):
            super().__setitem__(k, v)


def run_taps(args):
    ref = ir.build(args.model, torch.float32, "none", "w4a16", args.expert_cache_gb)
    args.out.mkdir(parents=True, exist_ok=True)
    for s in sequences(args.set, args.golden):
        if args.only and s["name"] not in args.only.split(","):
            continue
        path = args.out / f"taps_{s['name']}.safetensors"
        if path.exists():
            print(f"skip {path}", flush=True)
            continue
        ids = s["ids"]
        st = ir.SeqState(ref.c)
        taps, am, p1, eids = [], [], [], []
        t0 = time.time()
        for i in range(0, len(ids), args.chunk):
            ref.dump = TapDump()
            logits = ref.forward(torch.tensor(ids[i : i + args.chunk]), st)
            d = ref.dump
            ref.dump = None
            taps.append(torch.stack([d[f"v{v:02d}.partial_in"] for v in TAP_VISITS] + [d["final_hidden"]], 1))
            pr = torch.softmax(logits.double(), -1)
            top = pr.max(-1)
            am.append(top.indices)
            p1.append(top.values.float())
            eids.append(torch.stack([d[f"v{v:02d}.expert_ids"] for v in range(len(ref.c.visits()))], 0))
        save_file({
            "ids": torch.tensor(ids, dtype=torch.int64),
            "taps": torch.cat(taps).to(torch.bfloat16).contiguous(),
            "argmax": torch.cat(am).to(torch.int64),
            "p1": torch.cat(p1),
            "expert_ids": torch.cat(eids, 1).to(torch.int16).contiguous(),  # [48, T, 8]
        }, str(path), metadata={"name": s["name"], "start": str(s["start"]), "kind": s["kind"]})
        print(f"{s['name']:24s} T={len(ids):5d}  {time.time() - t0:.0f}s", flush=True)


# --------------------------------------------------------------------------------------
# drafter


class DSpark:
    def __init__(self, model_dir: Path):
        cfg = json.loads((model_dir / "config.json").read_text())
        self.mc = mc = cfg["mtp_config"]
        text = cfg.get("text_config", cfg)
        self.softcap = float(text.get("final_logit_softcapping") or 30.0)
        self.vocab_real = int(text.get("unpadded_vocab_size") or 200058)
        self.D, self.H, self.KV, self.hd = mc["hidden_size"], mc["num_attention_heads"], mc["num_key_value_heads"], mc["head_dim"]
        self.eps = mc["rms_norm_eps"]
        self.nl = mc["num_hidden_layers"]
        self.local = set(mc["local_layer_ids"])
        self.block = mc["block_size"]
        self.window = mc["sliding_window_size"] + self.block  # keys a local layer sees, block included
        self.r = mc["markov_rank"]
        assert mc["target_layer_ids"][-1] == 48 and len(mc["target_layer_ids"]) == 4
        w = load_file(str(model_dir / "mtp.safetensors"))
        self.w = {k: v.to(torch.float32) for k, v in w.items()}
        rss("mtp weights")
        tf = safe_open(str(next(model_dir.glob("model*.safetensors"))), "pt")
        self.embed = tf.get_tensor(ir.P + "embed.weight")  # bf16, gathered
        self.embed_norm = tf.get_tensor(ir.P + "embed_norm.weight").float()
        unembed = tf.get_tensor(ir.P + "unembed.weight")  # bf16 [201024, D]
        self.unembed = unembed.float()
        rss("embed / unembed")
        # vLLM refresh_derived_weights: softmax(fuse_weight) fp32; fold hidden_out into the LM head in fp64.
        self.alpha = torch.softmax(self.w["fuse_weight"], -1)
        ho_w = self.w["markov_head.hidden_out.weight"].double()
        ho_b = self.w["markov_head.hidden_out.bias"].double()
        fw, fb = [], []
        for s in range(0, unembed.shape[0], 8192):
            h = unembed[s : s + 8192].double()
            fw.append((h @ ho_w).to(torch.bfloat16))
            fb.append((h @ ho_b).to(torch.bfloat16))
        self.fold_w = torch.cat(fw).float()  # [V, r]
        self.fold_b = torch.cat(fb).float()  # [V]
        rss("fold")
        for i in range(self.nl):
            p = f"layers.{i}.self_attn."
            self.w[p + "qkv"] = torch.cat([self.w[p + "q_proj.weight"], self.w[p + "k_proj.weight"], self.w[p + "v_proj.weight"]])
            self.w[p + "kv_ctx"] = torch.cat([self.w[p + "k_proj_ctx.weight"], self.w[p + "v_proj_ctx.weight"]])
            m = f"layers.{i}.mlp."
            self.w[m + "gate_up"] = torch.cat([self.w[m + "gate_proj.weight"], self.w[m + "up_proj.weight"]])
        inv = 1.0 / (mc["rope_theta"] ** (torch.arange(0, self.hd, 2, dtype=torch.float32) / self.hd))
        self.inv_freq = inv

    # -- helpers (vLLM op semantics) --------------------------------------------------------
    def rms(self, x, w, eps=None):
        eps = self.eps if eps is None else eps
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w

    def lin(self, x, w):
        sh = x.shape
        return bf(x.reshape(-1, sh[-1]).contiguous() @ w.t()).reshape(*sh[:-1], w.shape[0])  # bf16 GEMM, bf16 out

    def rope(self, x, pos):
        """neox RoPE, fp32 math, x [T, h, hd] -> bf16."""
        ang = pos.to(torch.float32)[:, None] * self.inv_freq[None]  # [T, hd/2]
        cos, sin = ang.cos()[:, None], ang.sin()[:, None]
        h = self.hd // 2
        x1, x2 = x[..., :h], x[..., h:]
        return bf(torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1))

    def mix(self, taps, row, norm_w):
        """mix_rmsnorm: fp32 alpha-weighted sum of the 4 taps, RMSNorm, bf16 out. taps [T, 4, D] (bf16 values)."""
        m = (taps * self.alpha[row][None, :, None]).sum(1)
        return bf(self.rms(m, norm_w))

    def context_kv(self, taps, pos):
        """Per draft layer: K = RoPE(k_norm_ctx(k_proj_ctx(ctx))), V = v_proj_ctx(ctx). -> list of (K, V) [T, KV, hd]."""
        out = []
        for i in range(self.nl):
            p = f"layers.{i}.self_attn."
            ctx = self.mix(taps, i, self.w[f"layers.{i}.dflare_norm.weight"])
            kv = self.lin(ctx, self.w[p + "kv_ctx"])
            k, v = kv.split([self.KV * self.hd] * 2, -1)
            k = bf(self.rms(k.reshape(-1, self.KV, self.hd), self.w[p + "k_norm_ctx.weight"]))
            out.append((self.rope(k, pos), v.reshape(-1, self.KV, self.hd)))
        return out

    def f_rnn(self, taps_row):
        return self.mix(taps_row[None], self.nl, self.w["markov_head.dflare_norm.weight"])[0]

    # -- draft block ------------------------------------------------------------------------
    def block_hidden(self, anchors: torch.Tensor, starts: torch.Tensor, ctx_kv):
        """Run B blocks at once. anchors [B] token ids, starts [B] anchor positions t (context = positions < t).
        Returns the final-norm hidden [B, 8, D] (bf16 values)."""
        B, S, D = anchors.shape[0], self.block, self.D
        x = self.w["mask_embed"][None, None].expand(B, S, D).clone()
        e = self.embed[anchors].float()
        x[:, 0] = bf(self.rms(e, self.embed_norm))
        x = bf(x)
        pos = starts[:, None] + torch.arange(S)[None]  # [B, S]
        residual = None
        hcur = x
        G = self.H // self.KV
        for i in range(self.nl):
            L = f"layers.{i}."
            if residual is None:
                residual = hcur
                xn = bf(self.rms(hcur, self.w[L + "input_layernorm.weight"]))
            else:
                s = residual + hcur
                residual = bf(s)
                xn = bf(self.rms(s, self.w[L + "input_layernorm.weight"]))
            # attention
            p = L + "self_attn."
            qkv = self.lin(xn, self.w[p + "qkv"])  # [B, S, 3072+2048]
            q, k, v = qkv.split([self.H * self.hd, self.KV * self.hd, self.KV * self.hd], -1)
            q = bf(self.rms(q.reshape(B, S, self.H, self.hd), self.w[p + "q_norm.weight"]))
            k = bf(self.rms(k.reshape(B, S, self.KV, self.hd), self.w[p + "k_norm.weight"]))
            q = self.rope(q.reshape(B * S, self.H, self.hd), pos.reshape(-1)).reshape(B, S, self.H, self.hd)
            k = self.rope(k.reshape(B * S, self.KV, self.hd), pos.reshape(-1)).reshape(B, S, self.KV, self.hd)
            v = v.reshape(B, S, self.KV, self.hd)
            Kc, Vc = ctx_kv[i]
            out = torch.empty(B, S, self.H, self.hd)
            for b in range(B):
                t = int(starts[b])
                lo = max(0, t + S - self.window) if i in self.local else 0
                kk = torch.cat([Kc[lo:t], k[b]], 0)  # [N, KV, hd]
                vv = torch.cat([Vc[lo:t], v[b]], 0)
                qh = q[b].reshape(S, self.KV, G, self.hd)
                sc = torch.einsum("skgd,nkd->kgsn", qh, kk) * self.hd**-0.5
                pr = torch.softmax(sc, -1)
                o = torch.einsum("kgsn,nkd->skgd", pr, vv)
                out[b] = o.reshape(S, self.H, self.hd)
            a = self.lin(bf(out.reshape(B, S, -1)), self.w[p + "o_proj.weight"])
            s = residual + a
            residual = bf(s)
            xn = bf(self.rms(s, self.w[L + "post_attention_layernorm.weight"]))
            gu = self.lin(xn, self.w[L + "mlp.gate_up"])
            g, u = gu.split([gu.shape[-1] // 2] * 2, -1)
            hcur = self.lin(bf(torch.nn.functional.silu(g) * u), self.w[L + "mlp.down_proj.weight"])
        s = residual + hcur
        return bf(self.rms(s, self.w["norm.weight"]))

    def markov(self, hidden, f_rnn, anchors, n_steps: int):
        """hidden [B, n, D] (mask rows 1..n), f_rnn [B, D], anchors [B]. -> tokens [B, n], p1 [B, n], margin."""
        B = hidden.shape[0]
        r = self.r
        W = self.w["markov_head.rnn_proj.weight"]
        n = hidden.shape[1]
        # 2-D contiguous GEMMs (a batched matmul against a 2-D weight would broadcast the weight per batch)
        hc = torch.cat([hidden, f_rnn[:, None].expand_as(hidden)], -1).reshape(B * n, -1).contiguous()
        ctx = (self.w["markov_head.rnn_proj.bias"][None] + hc @ W[:, 2 * r :].t()).reshape(B, n, -1)
        base = bf(hidden.reshape(B * n, -1).contiguous() @ self.unembed.t()).reshape(B, n, -1)  # bf16 GEMM
        state = torch.zeros(B, r)
        prev = anchors.clone()
        toks, p1s = [], []
        for i in range(n_steps):
            inp = torch.cat([state, bf(self.w["markov_head.markov_w1.weight"][prev])], -1)
            raw = ctx[:, i] + inp @ W[:, : 2 * r].t()
            gate, cand, outp = bf(raw[:, :r]), bf(raw[:, r : 2 * r]), bf(raw[:, 2 * r :])
            gate = bf(1.0 / (1.0 + torch.exp(-gate)))
            cand = bf(torch.tanh(cand))
            keep = bf(gate * state)
            upd = bf(bf(1.0 - gate) * cand)
            state = bf(keep + upd)
            latent = bf(torch.tanh(outp))
            bias = bf(self.fold_b[None] + latent @ self.fold_w.t())
            x = bf(base[:, i] + bias)
            logits = bf(self.softcap * torch.tanh(x / self.softcap))[:, : self.vocab_real]
            pr = torch.softmax(logits, -1)
            top = pr.max(-1)
            prev = logits.argmax(-1)
            toks.append(prev)
            p1s.append(top.values)
        return torch.stack(toks, 1), torch.stack(p1s, 1)


def run_draft(args):
    ds = DSpark(args.model)
    rss("drafter loaded")
    for path in sorted(args.out.glob("taps_*.safetensors")):
        name = path.stem[5:]
        if args.only and name not in args.only.split(","):
            continue
        dpath = args.out / f"draft_{name}.safetensors"
        if dpath.exists():
            print(f"skip {dpath}", flush=True)
            continue
        f = safe_open(str(path), "pt")
        start = int(f.metadata()["start"])
        d = load_file(str(path))
        ids, taps = d["ids"], d["taps"].float()
        T = ids.shape[0]
        t0 = time.time()
        ctx_kv = ds.context_kv(taps, torch.arange(T))
        rss(f"{name} context kv")
        anchors_pos = torch.arange(start, T - 1)
        toks, p1s = [], []
        for s in range(0, anchors_pos.shape[0], args.batch):
            ap = anchors_pos[s : s + args.batch]
            h = ds.block_hidden(ids[ap], ap, ctx_kv)
            fr = torch.stack([ds.f_rnn(taps[int(t) - 1]) for t in ap])
            tk, pp = ds.markov(h[:, 1:], fr, ids[ap], ds.block - 1)
            toks.append(tk)
            p1s.append(pp)
            if s == 0:
                rss(f"{name} first batch")
        save_file({"anchor_pos": anchors_pos, "draft": torch.cat(toks), "p1": torch.cat(p1s).float()}, str(dpath))
        print(f"{name:24s} anchors={anchors_pos.shape[0]:5d}  {time.time() - t0:.0f}s", flush=True)


# --------------------------------------------------------------------------------------
# analysis

# Metal kernel timings on endor (M5 Max), per visit, from the ink-m-moe / ink-m-attn results docs.
# experts: (union, µs) points, NVFP4 and MXFP8; attention: SWA / full at ctx 512, best of fused / pair per row count.
NV_PTS = [(8, 83.0), (29.5, 269.0), (83.2, 760.0)]
MX_PTS = [(8, 131.0), (29.5, 467.0), (83.2, 1290.0)]
ATTN_PTS = [(1, 22.9), (4, 35.8), (16, 97.2)]
N_NV, N_MX, N_VISITS = 44, 4, 48


def interp(pts, x):
    if x <= pts[0][0]:
        (x0, y0), (x1, y1) = pts[0], pts[1]
    else:
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x <= x1:
                break
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def match_len(draft, g, text, t, n):
    """accepted prefix length of draft[0:n] at anchor t (see module doc)."""
    m = 0
    for j in range(1, n + 1):
        if t + j - 1 >= g.shape[0] or int(draft[j - 1]) != int(g[t + j - 1]):
            break
        m = j
        if t + j >= text.shape[0] or int(text[t + j]) != int(g[t + j - 1]):
            break  # the text leaves the target's greedy path: later conditionals are not valid
    return m


def run_analyze(args):
    rows = []
    unions = {"nv": {R: [] for R in range(1, 9)}, "mx": {R: [] for R in range(1, 9)}}
    mx_visits = None
    for dpath in sorted(args.out.glob("draft_*.safetensors")):
        name = dpath.stem[6:]
        d = load_file(str(dpath))
        tp = load_file(str(args.out / f"taps_{name}.safetensors"))
        f = safe_open(str(args.out / f"taps_{name}.safetensors"), "pt")
        kind = f.metadata()["kind"]
        ids, g, eids = tp["ids"], tp["argmax"], tp["expert_ids"].long()
        if mx_visits is None:
            cfg = ir.Cfg.load(args.model)
            mx_visits = [vi for vi, (L, _) in enumerate(cfg.visits()) if L in cfg.mxfp8_layers]
        T = ids.shape[0]
        for a, t in enumerate(d["anchor_pos"].tolist()):
            m = match_len(d["draft"][a], g, ids, t, 7)
            # on the target's greedy path: the anchor itself is the target's token (vLLM W4A4 vs W4A16 near-ties are not)
            on = int(ids[t]) == int(g[t - 1])
            rows.append({"seq": name, "kind": kind if on else kind + "-offpath", "t": t, "m": m, "p1": d["p1"][a].tolist(),
                         "draft": d["draft"][a].tolist()})
        # expert union over R consecutive verify rows (anchor .. anchor + R - 1)
        oh = torch.zeros(eids.shape[0], T, 128, dtype=torch.bool)
        oh.scatter_(2, eids, True)
        start = int(f.metadata()["start"])
        for R in range(1, 9):
            if T - start < R:
                continue
            acc = torch.zeros(eids.shape[0], T - R + 1 - start, 128, dtype=torch.bool)
            for j in range(R):
                acc |= oh[:, start + j : T - R + 1 + j]
            u = acc.sum(-1).float()  # [48, n]
            mxm = torch.zeros(eids.shape[0], dtype=torch.bool)
            mxm[mx_visits] = True
            unions["nv"][R].append(u[~mxm].flatten())
            unions["mx"][R].append(u[mxm].flatten())
    U = {fmt: {R: torch.cat(v).mean().item() for R, v in unions[fmt].items() if v} for fmt in unions}

    def verify_us(R):
        ex = N_NV * interp(NV_PTS, U["nv"][R]) + N_MX * interp(MX_PTS, U["mx"][R])
        at = N_VISITS * interp(ATTN_PTS, R)
        return ex, at

    # plain step (1 row): measured kernels + the dense bytes at the kernels' streaming rate
    dense_us = args.dense_gb * 1e6 / args.gbps
    ex1, at1 = verify_us(1)
    t1 = dense_us + ex1 + at1 + args.fixed_us

    def step_us(R, n_markov, cheap=False):
        ex, at = verify_us(R)
        if cheap:  # perf-strategy §6: fp8 layers, 64k fp8 draft head, Markov bias on the top-k base candidates only
            draft = args.cheap_draft_gb * 1e6 / args.gbps
        else:
            draft = (args.draft_layers_gb + args.draft_head_gb) * 1e6 / args.gbps + n_markov * args.draft_fold_gb * 1e6 / args.gbps
        return dense_us + ex + at + args.fixed_us + draft + args.draft_fixed_us

    kinds = sorted({r["kind"] for r in rows})
    for r in rows:
        r["base"] = r["kind"].removesuffix("-offpath")
    out = {"U": U, "t1_us": t1, "tables": {}}
    lines = []
    lines.append(f"anchors: " + ", ".join(f"{k} {sum(1 for r in rows if r['kind'] == k)}" for k in kinds))
    lines.append("")
    lines.append("Expert union U(R) per visit over R consecutive verify rows (real routing, teacher-forced text):")
    lines.append("| R | " + " | ".join(str(R) for R in range(1, 9)) + " |")
    lines.append("|---|" + "---:|" * 8)
    for fmt in ("nv", "mx"):
        lines.append(f"| {fmt} | " + " | ".join(f"{U[fmt][R]:.1f}" for R in range(1, 9)) + " |")
    lines.append(f"| indep. routing 128(1-(120/128)^R) | " + " | ".join(f"{128 * (1 - (120 / 128) ** R):.1f}" for R in range(1, 9)) + " |")
    lines.append("")
    lines.append(f"Cost model: plain step {t1 / 1000:.2f} ms ({1e6 / t1:.0f} tok/s modelled); verify(R) ms: "
                 + ", ".join(f"R={R} {step_us(R, 0) / 1000 - (args.draft_layers_gb + args.draft_head_gb) * 1e3 / args.gbps - args.draft_fixed_us / 1000:.2f}" for R in range(1, 9)))
    lines.append("")
    sels = kinds + [b + "+offpath" for b in sorted({r["base"] for r in rows})] + ["all-onpath", "all"]
    for sel in sels:
        if sel == "all":
            rs = rows
        elif sel == "all-onpath":
            rs = [r for r in rows if not r["kind"].endswith("-offpath")]
        elif sel.endswith("+offpath"):
            rs = [r for r in rows if r["base"] == sel.removesuffix("+offpath")]
        else:
            rs = [r for r in rows if r["kind"] == sel]
        n = len(rs)
        if not n:
            continue
        lines.append(f"### {sel} ({n} anchors)")
        # acceptance by position: P(m >= j) and conditional P(m >= j | m >= j-1)
        pj = [sum(1 for r in rs if r["m"] >= j) / n for j in range(1, 8)]
        cj = [pj[0]] + [pj[j] / pj[j - 1] if pj[j - 1] else 0 for j in range(1, 7)]
        lines.append("| draft position j | " + " | ".join(str(j) for j in range(1, 8)) + " |")
        lines.append("|---|" + "---:|" * 7)
        lines.append("| P(accepted >= j) | " + " | ".join(f"{x:.3f}" for x in pj) + " |")
        lines.append("| P(j accepted given j-1) | " + " | ".join(f"{x:.3f}" for x in cj) + " |")
        lines.append("")
        lines.append("| policy | mean k | tokens / step | rows | modelled ms / step (native drafter) | tok/s | speedup | speedup, cheap drafter |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        lines.append(f"| no spec | 0 | 1.000 | 1 | {t1 / 1000:.2f} | {1e6 / t1:.1f} | 1.00 | 1.00 |")
        res = {}

        def pol(label, kf):
            ks = [kf(r) for r in rs]
            acc = [min(r["m"], k) for r, k in zip(rs, ks)]
            toks = sum(1 + a for a in acc) / n
            # drafter always runs (k = 0 still pays it, unless the gate is decided before drafting)
            us = sum(step_us(k + 1, max(k, 1)) for k in ks) / n
            usc = sum(step_us(k + 1, max(k, 1), True) for k in ks) / n
            lines.append(f"| {label} | {sum(ks) / n:.2f} | {toks:.3f} | {sum(k + 1 for k in ks) / n:.2f} | {us / 1000:.2f} | "
                         f"{toks * 1e6 / us:.1f} | {toks * t1 / us:.2f} | {toks * t1 / usc:.2f} |")
            res[label] = {"mean_k": sum(ks) / n, "tokens_per_step": toks, "ms": us / 1000, "speedup": toks * t1 / us,
                          "speedup_cheap": toks * t1 / usc}

        for k in range(1, 8):
            pol(f"fixed k={k}", lambda r, k=k: k)
        for kmax in (2, 3, 4, 7):
            for tau in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
                def gate(r, tau=tau, kmax=kmax):
                    k = 0
                    while k < kmax and r["p1"][k] >= tau:
                        k += 1
                    return k
                pol(f"gate p1>={tau} kmax={kmax}", gate)
        for kmax in (3, 7):
            for tau in (0.5, 0.7, 0.8):
                def cgate(r, tau=tau, kmax=kmax):
                    k, c = 0, 1.0
                    while k < kmax and c * r["p1"][k] >= tau:
                        c *= r["p1"][k]
                        k += 1
                    return k
                pol(f"cum prod>={tau} kmax={kmax}", cgate)
        # calibration: draft position 1 p1 bucket vs accept
        lines.append("")
        lines.append("Calibration of position-1 confidence (p1 bucket: n, accept rate):")
        bks = [0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.01]
        cal = []
        for lo, hi in zip(bks, bks[1:]):
            sub = [r for r in rs if lo <= r["p1"][0] < hi]
            if sub:
                cal.append(f"[{lo:.2f},{min(hi, 1):.2f}) {len(sub)}: {sum(1 for r in sub if r['m'] >= 1) / len(sub):.2f}")
        lines.append("; ".join(cal))
        lines.append("")
        out["tables"][sel] = {"p_ge": pj, "cond": cj, "policies": res}
    lines.append("Per sequence: anchors, P(accepted >= 1), mean accepted of 7, off-path anchors")
    for sq in sorted({r["seq"] for r in rows}):
        rs = [r for r in rows if r["seq"] == sq]
        off = sum(1 for r in rs if r["kind"].endswith("-offpath"))
        lines.append(f"- {sq}: {len(rs)}, {sum(1 for r in rs if r['m'] >= 1) / len(rs):.3f}, {sum(r['m'] for r in rs) / len(rs):.2f}, {off}")
    txt = "\n".join(lines)
    print(txt)
    (args.out / "analysis.md").write_text(txt + "\n")
    (args.out / "analysis.json").write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["taps", "draft", "analyze"])
    ap.add_argument("--model", type=Path, default=ir.SNAPSHOT)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--golden", type=Path, default=Path("/opt/code/intarsiaml/golden"))
    ap.add_argument("--set", choices=["greedy", "prose", "all"], default="all")
    ap.add_argument("--only", default=None)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--expert-cache-gb", type=float, default=6.0)
    # cost model knobs (endor M5 Max)
    ap.add_argument("--gbps", type=float, default=480.0, help="dense / drafter streaming rate")
    ap.add_argument("--dense-gb", type=float, default=3.65, help="dense bytes per step (attention proj + unembed + ...)")
    ap.add_argument("--fixed-us", type=float, default=1500.0, help="per-step launch / mix / conv overhead not in the kernels")
    ap.add_argument("--draft-layers-gb", type=float, default=0.524)
    ap.add_argument("--draft-head-gb", type=float, default=0.922, help="base logits through the target unembed")
    ap.add_argument("--draft-fold-gb", type=float, default=0.102, help="Markov fold [V, 256] per sequential step")
    ap.add_argument("--cheap-draft-gb", type=float, default=0.5)
    ap.add_argument("--draft-fixed-us", type=float, default=300.0)
    args = ap.parse_args()
    {"taps": run_taps, "draft": run_draft, "analyze": run_analyze}[args.mode](args)


if __name__ == "__main__":
    main()
