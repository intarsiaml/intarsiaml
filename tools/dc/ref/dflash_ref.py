#!/usr/bin/env python
"""Independent MLX reference for DFlash (parallel block drafter) and the DCOCO corrector.

Target Qwen3-8B (HF bf16) + draft TD-Labs/Qwen3-8B-DCOCO (DFlash layout, 5 layers, taps 0/8/16/24/32).
Greedy speculative decoding (block = [anchor] + 15 mask rows, 15 drafts), plus a fixture dump.

  python dflash_ref.py --ckpt DIR --target DIR --prompts prompts_dc.jsonl --max-new 192 \
      --mode {dflash,dcoco} [--variant SPEC ...] --out run.json [--fixture-dir D --fixture-ids a,b,c]

MLX bf16 on the Apple GPU is the numeric reference for the target and the draft transformer; the
DCOCO corrector (dcoco_ref.py) runs in true f32 on the CPU stream.
Layer taps: feature j = OUTPUT of target decoder layer target_layer_ids[j] (0-based), no final norm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx_lm import load as mlx_load
from mlx_lm.models.cache import KVCache, make_prompt_cache

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dcoco_ref as D  # noqa: E402
from zlab_dflash_mlx import DFlashConfig, DFlashDecoderLayer, _build_rope  # noqa: E402

# Block layouts: (rows in the query block incl. the anchor row, first row used as a draft, n drafts).
# "r15": [anchor]+14 masks, drafts = rows 0..14 (row r predicts the token at anchor+r+1: standard
#        shifted next-token convention; found empirically to be what this checkpoint was trained for).
# "r16s": [anchor]+15 masks, drafts = rows 0..14 (row 15 computed and ignored).
# "zlab": [anchor]+15 masks, drafts = rows 1..15 (z-lab's DFlash convention: mask row at position p
#        predicts the token at p). Kept for comparison; it gives tau ~1 on this checkpoint.
LAYOUTS = {"r15": (15, 0, 15), "r16s": (16, 0, 15), "zlab": (16, 1, 15)}
EOS_IDS = {151645, 151643}   # <|im_end|>, <|endoftext|>
TOPK = 16


# ----------------------------------------------------------------------------------------------
# draft model (adapter for the DCOCO checkpoint's config format; layers are the z-lab ones)
# ----------------------------------------------------------------------------------------------
def draft_config(ckpt: Path) -> DFlashConfig:
    cfg = json.loads((ckpt / "config.json").read_text())
    df = cfg["dflash_config"]
    rope = cfg.get("rope_parameters") or {}
    rope_scaling = None if rope.get("rope_type", "default") == "default" else rope
    n = cfg["num_hidden_layers"]
    return DFlashConfig(
        hidden_size=cfg["hidden_size"], num_hidden_layers=n,
        num_attention_heads=cfg["num_attention_heads"], num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"], intermediate_size=cfg["intermediate_size"], vocab_size=cfg["vocab_size"],
        rms_norm_eps=cfg["rms_norm_eps"], rope_theta=cfg.get("rope_theta", rope.get("rope_theta", 10000.0)),
        max_position_embeddings=cfg["max_position_embeddings"],
        block_size=int(df["block_size"]),
        target_layer_ids=tuple(df["target_layer_ids"]), num_target_layers=36,
        mask_token_id=df["mask_token_id"], rope_scaling=rope_scaling,
        layer_types=tuple(cfg.get("layer_types") or ["full_attention"] * n),
        sliding_window=cfg.get("sliding_window"),
        is_causal=bool(df["causal"]) if "causal" in df else None,
    )


class Draft(nn.Module):
    def __init__(self, cfg: DFlashConfig):
        super().__init__()
        self.config = cfg
        self.fc = nn.Linear(len(cfg.target_layer_ids) * cfg.hidden_size, cfg.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.layers = [DFlashDecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)]
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.rope = _build_rope(cfg.head_dim, cfg.rope_theta, cfg.max_position_embeddings, cfg.rope_scaling)

    def new_cache(self):
        return [KVCache() for _ in self.layers]

    def block(self, embed, lm_head, tokens, feat, caches, collect=False, start=0, n=15):
        """tokens [1,R] (anchor + masks), feat [1,S,20480] target features of the NEW context rows.
        Draft rows are start..start+n-1. Returns dict: logits [1,n,V] (bf16), h_post [n,d], h_pre [n,d],
        (+ h_ctx, per_layer when collect)."""
        h = embed(tokens)
        h_ctx = self.hidden_norm(self.fc(feat))
        per_layer = []
        for layer, c in zip(self.layers, caches):
            h = layer(h, h_ctx, self.rope, c)
            if collect:
                per_layer.append(h)
        hn = self.norm(h)
        sl = slice(start, start + n)
        out = {"logits": lm_head(hn[:, sl]), "h_post": hn[0, sl], "h_pre": h[0, sl]}
        if collect:
            out.update(h_ctx=h_ctx[0], per_layer=[x[0] for x in per_layer])
        return out


def load_draft(ckpt: Path):
    cfg = draft_config(ckpt)
    tensors = mx.load(str(ckpt / "model.safetensors"))
    model = Draft(cfg)
    skip = ("embed_tokens.", "lm_head.")        # byte-identical to the target's; bound from the target
    weights = [(k, v) for k, v in tensors.items() if not k.startswith("dcoco.") and not k.startswith(skip)]
    model.load_weights(weights, strict=True)
    model.eval()
    mx.eval(model.parameters())
    return model, tensors


# ----------------------------------------------------------------------------------------------
# target with feature taps
# ----------------------------------------------------------------------------------------------
class _LayerHook:
    def __init__(self, layer, idx, storage):
        self._layer, self._idx, self._storage = layer, idx, storage

    def __call__(self, *args, **kwargs):
        out = self._layer(*args, **kwargs)
        self._storage[self._idx] = out[0] if isinstance(out, tuple) else out
        return out

    def __getattr__(self, name):
        return getattr(self._layer, name)


def patch_target(model, layer_ids):
    model._hidden_states = [None] * len(layer_ids)
    for i, lid in enumerate(layer_ids):
        model.model.layers[lid] = _LayerHook(model.model.layers[lid], i, model._hidden_states)


def feats(model):
    return mx.concatenate(model._hidden_states, axis=-1)


# ----------------------------------------------------------------------------------------------
# generation
# ----------------------------------------------------------------------------------------------
def encode_prompt(tok, messages):
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok.encode(text, add_special_tokens=False)


def plain_greedy(model, prompt_ids, max_new):
    cache = make_prompt_cache(model)
    logits = model(mx.array(prompt_ids)[None], cache=cache)
    tok = int(mx.argmax(logits[0, -1]).item())
    ids = [tok]
    while len(ids) < max_new and tok not in EOS_IDS:
        logits = model(mx.array([[tok]]), cache=cache)
        tok = int(mx.argmax(logits[0, -1]).item())
        ids.append(tok)
    return ids


def topk_desc(logits_f32: np.ndarray, k: int = TOPK):
    """k largest per row, ordered by (value desc, index asc) - deterministic under bf16 ties."""
    thr = np.partition(logits_f32, -k, axis=-1)[:, -k]
    ids, vals = np.zeros((len(logits_f32), k), np.int64), np.zeros((len(logits_f32), k), np.float32)
    for i, row in enumerate(logits_f32):
        cand = np.nonzero(row >= thr[i])[0]
        order = np.lexsort((cand, -row[cand]))[:k]
        ids[i], vals[i] = cand[order], row[cand[order]]
    return ids, vals


class Proposer:
    """Turns the draft block output into 15 draft tokens (plain argmax or DCOCO-corrected)."""

    def __init__(self, draft, target, tensors, mode, var, layout="r15"):
        self.draft, self.target, self.mode, self.var = draft, target, mode, var
        self.layout = layout
        self.rows, self.start, self.n_draft = LAYOUTS[layout]
        self.embed, self.lm_head = target.model.embed_tokens, target.lm_head
        self.W = D.load_dcoco_weights(tensors) if mode == "dcoco" else None

    def propose(self, anchor, feat, caches, collect=False):
        cfg = self.draft.config
        block = mx.array([[anchor] + [cfg.mask_token_id] * (self.rows - 1)])
        out = self.draft.block(self.embed, self.lm_head, block, feat, caches, collect, self.start, self.n_draft)
        lf = np.array(out["logits"][0].astype(mx.float32))             # [15,V]
        ids, vals = topk_desc(lf)
        res = {"out": out, "top_ids": ids, "top_vals": vals, "argmax": ids[:, 0].copy()}
        if self.mode == "dflash":
            res["drafts"] = res["argmax"]
            return res
        l = vals
        if self.var.logit == "logsoftmax":
            m = lf.max(-1, keepdims=True)
            lse = (m + np.log(np.exp(lf - m).sum(-1, keepdims=True))).astype(np.float32)
            l = vals - lse
        elif self.var.logit == "lmax":
            l = vals - vals[:, :1]
        h = out["h_post" if self.var.hnorm == "post" else "h_pre"].astype(mx.float32)
        cand_emb = self.embed(mx.array(ids)).astype(mx.float32)           # [15,16,d]
        sc = D.dcoco_scores(self.W, h, cand_emb, mx.array(l), self.var)
        S = np.array(sc["S"])
        path = D.select_path(S, l)
        res.update(scores={k: np.array(v) for k, v in sc.items()}, S=S, path=path, l=l,
                   drafts=ids[np.arange(len(path)), path])
        return res


def spec_decode(target, prop: Proposer, prompt_ids, max_new):
    draft = prop.draft
    tcache, dcache = make_prompt_cache(target), draft.new_cache()
    logits = target(mx.array(prompt_ids)[None], cache=tcache)
    feat = feats(target)
    tok = int(mx.argmax(logits[0, -1]).item())
    ids, rounds = [tok], []
    while len(ids) < max_new and ids[-1] not in EOS_IDS:
        anchor, pos = ids[-1], len(prompt_ids) + len(ids) - 1
        res = prop.propose(anchor, feat, dcache)
        drafts = [int(x) for x in res["drafts"]]
        vlog = target(mx.array([[anchor] + drafts]), cache=tcache)
        tgt = [int(x) for x in mx.argmax(vlog[0], axis=-1).tolist()]
        acc = next((i for i in range(len(drafts)) if drafts[i] != tgt[i]), len(drafts))
        new = (drafts[:acc] + [tgt[acc]])[: max_new - len(ids)]
        eos = next((i for i, t in enumerate(new) if t in EOS_IDS), None)
        if eos is not None:
            new = new[: eos + 1]
        ids += new
        rounds.append({"anchor_pos": pos, "anchor_token": anchor, "drafts": drafts, "accepted": acc,
                       "committed": len(new)})
        for c in tcache:
            c.trim(len(drafts) - acc)
        feat = feats(target)[:, : acc + 1]
        mx.eval(feat)
    return ids, rounds


def run_spec(target, tok, draft, tensors, prompts, plain, max_new, mode, var, log=print, layout="r15"):
    prop = Proposer(draft, target, tensors, mode, var, layout)
    per, t0 = [], time.time()
    for p in prompts:
        pids = encode_prompt(tok, p["messages"])
        t1 = time.time()
        ids, rounds = spec_decode(target, prop, pids, max_new)
        ref = plain[p["id"]]["ids"]
        first = next((i for i in range(min(len(ids), len(ref))) if ids[i] != ref[i]), None)
        if first is None and len(ids) != len(ref):
            first = min(len(ids), len(ref))
        accs = [r["accepted"] for r in rounds]
        per.append({"id": p["id"], "kind": p["kind"], "prompt_len": len(pids), "ids": ids, "rounds": rounds,
                    "n_rounds": len(rounds), "tau": float(np.mean(accs) + 1) if accs else None,
                    "tau_committed": (len(ids) - 1) / len(rounds) if rounds else None,
                    "equals_plain_greedy": first is None, "first_diff": first, "sec": round(time.time() - t1, 2)})
        log(f"  [{var.name() if mode == 'dcoco' else 'dflash'}] {p['id']}: rounds={len(rounds)} "
            f"tau={per[-1]['tau']:.3f} equal={first is None} first_diff={first} ({per[-1]['sec']}s)")
    allacc = [r["accepted"] for q in per for r in q["rounds"]]
    agg = {"prompts": len(per), "rounds": len(allacc), "tau_pooled": float(np.mean(allacc) + 1),
           "tau_mean_of_prompts": float(np.mean([q["tau"] for q in per])),
           "tau_committed_pooled": float(sum(len(q["ids"]) - 1 for q in per) / len(allacc)),
           "equal_plain_greedy": sum(q["equals_plain_greedy"] for q in per),
           "sec": round(time.time() - t0, 1)}
    return {"mode": mode, "layout": layout, "variant": var.name() if mode == "dcoco" else "dflash",
            "variant_spec": vars(var) if mode == "dcoco" else None, "aggregate": agg, "prompts": per}


# ----------------------------------------------------------------------------------------------
# fixture dump
# ----------------------------------------------------------------------------------------------
def dump_fixture(target, tok, draft, tensors, prompt, plain_ids, mode, var, out_dir: Path, extra_delay=24,
                 layout="r15"):
    pids = encode_prompt(tok, prompt["messages"])
    P = len(pids) + extra_delay
    toks = pids + plain_ids[: extra_delay + 1]          # prompt + 24 generated + anchor (the 25th generated)
    assert len(plain_ids) >= extra_delay + 1 and len(toks) == P + 1
    cache = make_prompt_cache(target)
    target(mx.array(toks[:P])[None], cache=cache)
    feat = feats(target)                                 # [1,P,20480]
    prop = Proposer(draft, target, tensors, mode, var, layout)
    dcache = draft.new_cache()
    res = prop.propose(toks[P], feat, dcache, collect=True)
    out = res["out"]
    mx.eval(out["per_layer"], out["h_ctx"], out["h_post"])
    f32 = lambda a: np.array(a.astype(mx.float32))
    d = {"tokens": np.array(toks, np.int64), "P": np.int64(P), "feat": f32(feat[0]),
         "anchor_token": np.int64(toks[P]), "h_ctx": f32(out["h_ctx"])}
    for l, layer in enumerate(draft.layers):
        a = layer.self_attn
        hc = out["h_ctx"][None]
        k = a.k_norm(a.k_proj(hc).reshape(1, P, a.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        k = draft.rope(k, offset=0)
        v = a.v_proj(hc).reshape(1, P, a.n_kv_heads, -1).transpose(0, 2, 1, 3)
        d[f"k_ctx_l{l}"] = f32(k[0].transpose(1, 0, 2))      # [P,8,128]
        d[f"v_ctx_l{l}"] = f32(v[0].transpose(1, 0, 2))
        d[f"blk_h_l{l}"] = f32(out["per_layer"][l])          # [rows,4096]
    d.update(h_final=f32(out["h_post"]), h_pre_norm=f32(out["h_pre"]),
             top_ids=res["top_ids"], top_vals=res["top_vals"], argmax_drafts=res["argmax"].astype(np.int64))
    if mode == "dcoco":
        sc = res["scores"]
        d.update(S=res["S"], delta=sc["delta"], path=np.array(res["path"], np.int64),
                 corrected_drafts=res["drafts"].astype(np.int64), b=sc["b"], g=sc["g"], z=sc["z"], s=sc["s"])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"fixture_{prompt['id']}.npz", **d)
    side = {"prompt_id": prompt["id"], "P": P, "prompt_len": len(pids), "mode": mode, "layout": layout,
            "block_rows": prop.rows, "draft_rows": [prop.start, prop.start + prop.n_draft - 1],
            "variant": var.name(), "variant_spec": vars(var), "tap_semantics": "output of target layer i (0-based)",
            "layer_ids": list(draft.config.target_layer_ids),
            "shapes": {k: list(v.shape) for k, v in d.items()},
            "dtypes": {k: str(v.dtype) for k, v in d.items()},
            "notes": "feat/h_ctx/k/v/blk_h/h_final/top_vals stored as float32 casts of bf16 MLX tensors; "
                     "S/delta/b/g/z/s are f32 (CPU sgemm). k_ctx is after k_norm+rope (positions 0..P-1)."}
    (out_dir / f"fixture_{prompt['id']}.json").write_text(json.dumps(side, indent=1))
    return side


# ----------------------------------------------------------------------------------------------
GRID_ONE_AT_A_TIME = ["default", "eps=1e-6", "gelu=tanh", "order=B", "order=C", "hnorm=pre", "logit=logsoftmax"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--max-new", type=int, default=192)
    ap.add_argument("--mode", choices=["dflash", "dcoco"], required=True)
    ap.add_argument("--variant", action="append", default=None,
                    help="dcoco variant spec, e.g. 'eps=1e-6,gelu=tanh' or 'default'; repeatable; "
                         "'grid1' expands to the one-factor-at-a-time grid")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plain-cache", default=None, help="json cache of the plain greedy target ids")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--layout", choices=list(LAYOUTS), default="r15")
    ap.add_argument("--tap-shift", type=int, default=0, help="debug: add to every target_layer_id")
    ap.add_argument("--fixture-dir", default=None)
    ap.add_argument("--fixture-ids", default="")
    ap.add_argument("--skip-spec", action="store_true", help="only plain greedy / fixtures")
    a = ap.parse_args()

    import mlx_lm
    meta = {"mlx": __import__("importlib.metadata").metadata.version("mlx"),
            "mlx_lm": mlx_lm.__version__, "ckpt": a.ckpt, "target": a.target, "max_new": a.max_new,
            "mode": a.mode, "layout": a.layout, "tap_shift": a.tap_shift, "prompts_file": a.prompts,
            "cmd": " ".join(sys.argv)}
    prompts = [json.loads(x) for x in Path(a.prompts).read_text().splitlines() if x.strip()]
    if a.limit:
        prompts = prompts[: a.limit]
    t0 = time.time()
    target, tok = mlx_load(a.target)
    draft, tensors = load_draft(Path(a.ckpt))
    layer_ids = [i + a.tap_shift for i in draft.config.target_layer_ids]
    patch_target(target, layer_ids)
    print(f"loaded in {time.time() - t0:.1f}s; taps={layer_ids}; layout={a.layout}", flush=True)

    plain = {}
    cache_p = Path(a.plain_cache) if a.plain_cache else None
    if cache_p and cache_p.exists():
        plain = json.loads(cache_p.read_text())
    t1 = time.time()
    for p in prompts:
        if p["id"] not in plain or plain[p["id"]].get("max_new") != a.max_new:
            pids = encode_prompt(tok, p["messages"])
            ids = plain_greedy(target, pids, a.max_new)
            plain[p["id"]] = {"ids": ids, "prompt_len": len(pids), "max_new": a.max_new}
            print(f"  plain {p['id']}: prompt_len={len(pids)} new={len(ids)}", flush=True)
    meta["plain_sec"] = round(time.time() - t1, 1)
    if cache_p:
        cache_p.write_text(json.dumps(plain))

    runs = []
    if a.mode == "dflash":
        specs = [("dflash", D.Variant())]
    else:
        names = []
        for v in (a.variant or ["default"]):
            names += GRID_ONE_AT_A_TIME if v == "grid1" else [v]
        specs = [(n, D.Variant.parse(n)) for n in names]
    if not a.skip_spec:
        for _, var in specs:
            runs.append(run_spec(target, tok, draft, tensors, prompts, plain, a.max_new, a.mode, var,
                                 log=lambda s: print(s, flush=True), layout=a.layout))
            ag = runs[-1]["aggregate"]
            print(f"== {runs[-1]['variant']}: tau_pooled={ag['tau_pooled']:.3f} rounds={ag['rounds']} "
                  f"equal={ag['equal_plain_greedy']}/{ag['prompts']} ({ag['sec']}s)", flush=True)
            Path(a.out).write_text(json.dumps({"meta": meta, "plain": plain, "runs": runs}))
    if a.fixture_dir:
        var = specs[0][1]
        for pid in [x for x in a.fixture_ids.split(",") if x]:
            p = next(q for q in prompts if q["id"] == pid)
            side = dump_fixture(target, tok, draft, tensors, p, plain[pid]["ids"], a.mode, var, Path(a.fixture_dir),
                                layout=a.layout)
            print("fixture", pid, "P =", side["P"], flush=True)
    meta["total_sec"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps({"meta": meta, "plain": plain, "runs": runs}))
    print("done", meta["total_sec"], "s", flush=True)


if __name__ == "__main__":
    main()
