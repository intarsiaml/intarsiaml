#!/usr/bin/env python3
"""A tiny DSpine parity fixture from Tencent/DSpine's own MLX reference (`dspine_mlx`, Apache-2.0, imported and run, not copied).

    /opt/code/TensorFold/.venv/bin/python -I tools/dspine/dspine_mlx_fixture.py --repo <DSpine repo> <out dir>

Writes <out dir>/config.json + model.safetensors (a random 3-layer DSpine draft, bf16, in the checkpoint layout, with the four
I64 version scalars, an all-zero `out.weight` and every `selected_prefix_*` key) and <out dir>/golden.safetensors: for four cases
the context features, block ids, every layer's bridge output, the final hidden rows, the last write's stash, the pass-1
logits, candidates, drafts (and the plain argmax drafts), and the vocabulary tables.

The reference raises on any activation dtype but bf16 (`bridge.py`), so everything runs in bf16 on the MLX CPU device. Its
config validator refuses sliding windows, so MLX gets a stripped config (`full_attention`); the committed config.json keeps the
`sliding_attention` / `sliding_window = 8` the engine parses. The window is reproduced in the reference by an explicit boolean
mask in `scaled_dot_product_attention` (`qpos - kpos < W`, bottom-right aligned), the FlashAttention `window_left = W - 1`
convention the SGLang serving path uses; the reference code itself is not modified.

The seed is searched until every decision the tail takes is far from a tie (so the bitwise gates on candidates and drafts cannot flip on
the noise the hidden-row gate allows)."""
import argparse, json, os, sys, tempfile
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("out")
ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--tries", type=int, default=400)
args = ap.parse_args()
os.environ["DSPINE_TOPK"] = "full"  # the two-stage top-k needs V / 64 >= K; the selected set is the same
sys.path.insert(0, str(Path(args.repo) / "dspine_mlx"))
import numpy as np
import mlx.core as mx
import mlx.nn as nn

mx.set_default_device(mx.cpu)
from dspine_mlx import model as dm
from dspine_mlx import bridge as db
from dspine_mlx import propose as dp
from dspine_mlx.target import HeadAccess

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
fixture = Path(__file__).resolve().parents[2] / "crates/intarsia-models/tests/fixtures/tiny_qwen3_5"
H, V, NH, NKV, HD, DI, L, C, R, K, BLOCK, W, MASK = 64, 96, 4, 2, 16, 48, 3, 16, 8, 4, 6, 8, 95
TAPS = [1, 3, 5]
THETA = 1_000_000.0
EPS = 1e-6
TOL = 2e-3  # the margin unit below (the design's hidden-row bound; the engine test's HIDDEN_TOL is 3e-2, the measured noise)
# NOTE on margins: a seed is accepted when every tail and top-K decision is at least 5 * TOL = 1 % (relative) from a tie. Two correct
# bf16 pipelines differ by more than that per row (engine vs MLX 1.3-2.0 %, MLX CPU vs MLX GPU 1.0-1.3 %), so the engine's end-to-end
# test gates candidates and drafts on an agreement rate; only the tail on the reference's own inputs is bitwise.
base_cfg = {
    "architectures": ["DFlashDraftModel"], "attention_bias": False, "block_size": BLOCK, "hidden_size": H,
    "num_hidden_layers": L, "num_attention_heads": NH, "num_key_value_heads": NKV, "head_dim": HD, "intermediate_size": DI,
    "vocab_size": V, "draft_vocab_size": V, "rms_norm_eps": EPS, "rope_parameters": {"rope_theta": THETA, "rope_type": "default"},
    "rope_scaling": None, "rope_theta": THETA, "max_position_embeddings": 4096, "model_type": "qwen3", "is_causal": True,
    "num_target_layers": 6, "tie_word_embeddings": False, "hidden_act": "silu", "dtype": "bfloat16",
    "code_bridge_second_pass": {"first_exact": L - 1, "head_topk": K, "mode": "serial_table"},
    "dflash_config": {
        "draft_block_causal": True, "mask_token_id": MASK, "target_layer_ids": TAPS,
        "selected_prefix_code_bridge": {"code_size": C, "eps": EPS, "kappa": 10.0, "rank": R},
        "selected_prefix_decode_mode": "serial", "selected_prefix_feature_layers": list(range(1, L + 1)),
        "selected_prefix_feature_write_mode": "independent", "selected_prefix_feature_writes": True,
        "selected_prefix_last_write_refine": {"head_topk": K, "kind": "exact_sender", "version": 1, "weight": 1.0},
        "selected_prefix_terminal": False, "context_fusion_checkpointing": True, "decoder_checkpointing": True,
    },
}
cfg = dict(base_cfg)
cfg.update({"layer_types": ["sliding_attention"] * L, "sliding_window": W, "use_sliding_window": True})
stripped = dict(base_cfg)
stripped.update({"layer_types": ["full_attention"] * L})
(out / "config.json").write_text(json.dumps(cfg, indent=1))


def build_weights(seed):
    rng = np.random.default_rng(seed)
    w = {}

    def put(name, shape, kind):
        if kind == "norm":
            a = 1.0 + 0.1 * rng.standard_normal(shape)
        elif kind == "codebook":
            a = rng.standard_normal(shape)
            a = a / np.linalg.norm(a, axis=-1, keepdims=True)
        else:
            a = kind * rng.standard_normal(shape)
        w[name] = mx.array(a.astype(np.float32)).astype(mx.bfloat16)

    for l in range(L):
        p = f"layers.{l}"
        put(f"{p}.input_layernorm.weight", (H,), "norm"); put(f"{p}.post_attention_layernorm.weight", (H,), "norm")
        put(f"{p}.self_attn.q_proj.weight", (NH * HD, H), 0.15); put(f"{p}.self_attn.k_proj.weight", (NKV * HD, H), 0.15)
        put(f"{p}.self_attn.v_proj.weight", (NKV * HD, H), 0.15); put(f"{p}.self_attn.o_proj.weight", (H, NH * HD), 0.15)
        put(f"{p}.self_attn.q_norm.weight", (HD,), "norm"); put(f"{p}.self_attn.k_norm.weight", (HD,), "norm")
        put(f"{p}.mlp.gate_proj.weight", (DI, H), 0.15); put(f"{p}.mlp.up_proj.weight", (DI, H), 0.15)
        put(f"{p}.mlp.down_proj.weight", (H, DI), 0.1)
    put("fc.weight", (H, len(TAPS) * H), 0.1); put("hidden_norm.weight", (H,), "norm"); put("norm.weight", (H,), "norm")
    b = "selected_prefix_bridge"
    put(f"{b}.codebook", (V, C), "codebook")
    put(f"{b}.read_weight", (C, H), 0.15); put(f"{b}.early_read_weights", (L - 1, C, H), 0.15)
    put(f"{b}.feature_write_weights", (L, H, R), 0.25)
    put(f"{b}.gate.weight", (R, C), 0.3); put(f"{b}.gate.bias", (R,), 0.5)
    put(f"{b}.predecessor_gate.weight", (R, C), 0.3); put(f"{b}.prev.weight", (R, C), 0.3)
    w[f"{b}.out.weight"] = mx.zeros((C, R), dtype=mx.bfloat16)
    for name in ("draft_block_causal_version", "last_write_refine_version", f"{b}.independent_write_version", f"{b}.terminal_disabled_version"):
        w[name] = mx.array(1, dtype=mx.int64)
    return w


tgt = mx.load(str(fixture / "model.safetensors"))
embed = nn.Embedding(V, H)
embed.weight = tgt["model.embed_tokens.weight"].astype(mx.bfloat16)
lm = nn.Linear(H, V, bias=False)
lm.weight = tgt["lm_head.weight"].astype(mx.bfloat16)
head = HeadAccess(lm)


class Target:  # `bind` wants an `embed_tokens`
    embed_tokens = embed


# the window oracle: an explicit boolean mask, bottom-right aligned over the concatenated keys
_sdpa = mx.fast.scaled_dot_product_attention


def sdpa_window(q, k, v, scale, mask=None):
    sq, sk = q.shape[2], k.shape[2]
    qpos = sk - sq + mx.arange(sq)[:, None]
    kpos = mx.arange(sk)[None, :]
    allowed = (kpos <= qpos) & (qpos - kpos < W)
    return _sdpa(q, k, v, scale=scale, mask=allowed)


# collect every bridge call's (normed, out)
_orig_call = db.DSpineCodeBridge.__call__
_log = []


def logged(self, *a, **kw):
    r = _orig_call(self, *a, **kw)
    _log.append((r[0], r[1], a[0]))
    return r


def f32(x):
    return x.astype(mx.float32)


def bf_np(x):
    return np.asarray(f32(mx.array(np.asarray(x, dtype=np.float32)).astype(mx.bfloat16)))


CASES = [  # name, ingest chunks, block rows
    ("a", [2], 6), ("b", [3], 6), ("c", [9, 4], 6), ("d", [9, 4], 3),
]


def run_case(model, tail, seed, name, chunks, T):
    rng = np.random.default_rng(seed * 1000 + ord({"d": "c"}.get(name, name)))  # case d re-runs case c's inputs with a shorter block
    P = sum(chunks)
    feat = mx.array(rng.standard_normal((P, len(TAPS) * H)).astype(np.float32)).astype(mx.bfloat16)
    anchor = int(rng.integers(0, MASK))
    ids = mx.array([anchor] + [MASK] * (T - 1), dtype=mx.int32)
    model.set_block_size(T)
    caches = model.make_cache()
    done = 0
    for n in chunks:
        model.append_context(feat[done:done + n][None], caches)
        done += n
    _log.clear()
    hidden, stash = model(None, ids[None], caches)
    rows = hidden[0, 1:]
    logits = head.matmul(rows)
    drafts, cand = tail.greedy(rows, stash, anchor)
    mx.eval(hidden, stash["raw"], stash["rms"], stash["packed"], logits, drafts, cand, *[a for p in _log for a in p])
    return dict(feat=feat, ids=ids, anchor=anchor, hidden=hidden[0], stash=stash, logits=logits, drafts=drafts, cand=cand, log=list(_log), T=T)


def margins_ok(res, tg, tsv):
    """Every decision of the tail is at least 5 * TOL of the row's largest magnitude away from a tie."""
    lg = np.asarray(f32(res["logits"]))
    cand = np.asarray(res["cand"])
    worst = np.inf
    for r in range(lg.shape[0]):
        s = np.sort(lg[r])[::-1]
        worst = min(worst, (s[K - 1] - s[K]) / (np.abs(lg[r]).max() + 1e-9))
    # chain: top-2 score gap for the chosen sender
    raw = np.asarray(f32(res["stash"]["raw"]), dtype=np.float64)[1:]
    rms = np.asarray(f32(res["stash"]["rms"]), dtype=np.float64)[1:]
    q = np.asarray(f32(res["stash"]["packed"]), dtype=np.float64)[1:, :R]
    nw = np.asarray(f32(model.norm.weight), dtype=np.float64)
    gb = np.asarray(f32(model.selected_prefix_bridge.gate.bias), dtype=np.float64)
    headw = np.asarray(f32(lm.weight), dtype=np.float64)
    tgn = np.asarray(f32(tg), dtype=np.float64)
    tsvn = np.asarray(f32(tsv), dtype=np.float64)
    sender = res["anchor"]
    drafts = np.asarray(res["drafts"])
    for i in range(cand.shape[0]):
        m = 1 / (1 + np.exp(-(q[i] + tsvn[sender, :R] + gb))) * tsvn[sender, R:]
        sc = np.array([(raw[i] * nw) @ headw[c] + rms[i] * (m @ tgn[c]) for c in cand[i]])
        s = np.sort(sc)[::-1]
        worst = min(worst, (s[0] - s[1]) / (np.abs(sc).max() + 1e-9))
        sender = int(drafts[i])
    return worst >= 5 * TOL, worst


seed0 = args.seed if args.seed is not None else 1
for seed in range(seed0, seed0 + args.tries):
    w = build_weights(seed)
    mx.save_safetensors(str(out / "model.safetensors"), w)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "config.json").write_text(json.dumps(stripped, indent=1))
        mx.save_safetensors(str(td / "model.safetensors"), w)
        model, dcfg = dm.load_draft(td)
    model.bind(Target)
    tg, tsv = db.build_serial_tables(
        head_rows=head.rows_range, vocab_size=V, norm_w=model.norm.weight,
        write_w=model.selected_prefix_bridge.feature_write_weights[-1],
        pv_w=model.selected_prefix_bridge.packed_weight[R:], codebook=model.selected_prefix_bridge.codebook, chunk=32)
    tail = dp.RefinementTail(model, head, tg, tsv)
    db.DSpineCodeBridge.__call__ = logged
    mx.fast.scaled_dot_product_attention = sdpa_window
    try:
        res = {name: run_case(model, tail, seed, name, chunks, T) for name, chunks, T in CASES}
    finally:
        db.DSpineCodeBridge.__call__ = _orig_call
        mx.fast.scaled_dot_product_attention = _sdpa
    ok, worst = True, np.inf
    for name, r in res.items():
        o, wst = margins_ok(r, tg, tsv)
        ok &= o
        worst = min(worst, wst)
    differs = any(np.any(np.asarray(r["drafts"]) != np.asarray(mx.argmax(f32(r["logits"]), axis=-1))) for r in res.values())
    # the served shorter block is a prefix of the full block (case d against case c)
    pre = float(np.abs(np.asarray(f32(res["d"]["hidden"])) - np.asarray(f32(res["c"]["hidden"]))[:3]).max())
    if seed % 500 == 0 or (ok and differs):
        print(f"seed {seed}: margin {worst:.4f} (need {5 * TOL:.4f}), tail differs from argmax: {differs}, d-vs-c prefix {pre:.2e}")
    if ok and differs and pre < 3e-2:
        break
else:
    sys.exit("no seed with safe margins")

gold = {"tg": f32(tg), "tsv": f32(tsv)}
for name, r in res.items():
    gold[f"{name}.feat"] = f32(r["feat"])
    gold[f"{name}.ids"] = r["ids"]
    gold[f"{name}.hidden"] = f32(r["hidden"])
    gold[f"{name}.raw"] = f32(r["stash"]["raw"])
    gold[f"{name}.rms"] = f32(r["stash"]["rms"])
    gold[f"{name}.packed"] = f32(r["stash"]["packed"])
    gold[f"{name}.logits"] = f32(r["logits"])
    gold[f"{name}.cand"] = r["cand"].astype(mx.int32)
    gold[f"{name}.drafts"] = r["drafts"].astype(mx.int32)
    gold[f"{name}.argmax"] = mx.argmax(f32(r["logits"]), axis=-1).astype(mx.int32)
    gold[f"{name}.chunks"] = mx.array([c for c in CASES if c[0] == name][0][1], dtype=mx.int32)
    for l, (normed, o, pre) in enumerate(r["log"]):
        gold[f"{name}.layer{l}.out"] = f32(o[0])
        gold[f"{name}.layer{l}.normed"] = f32(normed[0])
        gold[f"{name}.layer{l}.raw"] = f32(pre[0])
mx.save_safetensors(str(out / "golden.safetensors"), gold)
print("wrote", out, "seed", seed, {k: tuple(v.shape) for k, v in list(gold.items())[:6]}, "...")
