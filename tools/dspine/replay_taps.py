#!/usr/bin/env python3
"""Tap-replay test: replay the target taps the intarsiaml engine fed its DSpine draft through Tencent's MLX reference (`dspine_mlx`,
Apache-2.0, imported and run, not copied) and compare drafts and acceptance.

Engine side (a dump per prompt):

    INTARSIA_DSPINE_DUMP=<dump> [INTARSIA_DSPINE_DUMP_LAYERS=1] intarsia generate --backend metal --model <target> \
        --spec dflash --spec-draft <draft> --spec-k 7 --chat --prompt ... --max-tokens N --ignore-eos --min-tokens N --show-ids

writes <dump>/req-0000/{meta.json,taps.bin,passes.bin,layers.bin,index.jsonl} (format: header of
crates/intarsia-core/src/model/dspine/dump.rs) and <dump>/ids-0.json (prompt + generated ids).

Replay side (this script; run it with an isolated venv's python and -I):

    python -I tools/dspine/replay_taps.py --repo <DSpine clone> --draft <draft dir> --target <target dir> --out <out dir> <dump> [<dump> ...]

For every draft pass of every dump it rebuilds the draft's context from the dumped taps (rows at their absolute positions; an ingest at
position p cuts what the cache held beyond p, as the engine's rollback does), runs the reference block (the full block, 16 rows, the
block is causal so the engine's shorter block is a prefix) with the target's bf16 embed_tokens / lm_head loaded straight from the target
safetensors, then compares with the engine's pass:
  * per-row draft agreement and the histogram of the first diverging row,
  * candidate-set agreement of the head's top-16 and the argmax,
  * for both drafts the accepted length against the target's real greedy tokens (ids[p0+1 ..], capped at k): tokens per pass = 1 + accepted,
  * with the dump's layers (INTARSIA_DSPINE_DUMP_LAYERS=1): the rel-L2 per layer of out / normed / raw and of the last stash, to find the
    first diverging stage,
  * a noise floor (--noise cpu): the same reference pass on the MLX CPU device, the other bf16 pipeline of the same model, so "ours vs
    reference" can be read against "reference GPU vs reference CPU".
"""
import argparse, json, os, sys, time
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("dumps", nargs="+", help="dump directories (each holds req-*/ and ids-*.json)")
ap.add_argument("--repo", required=True, help="clone of github.com/Tencent/DSpine")
ap.add_argument("--draft", required=True, help="draft checkpoint dir (config.json + model.safetensors)")
ap.add_argument("--target", required=True, help="target checkpoint dir (lm_head / embed_tokens read from its safetensors)")
ap.add_argument("--out", required=True, help="output dir (per-dump json, run.log)")
ap.add_argument("--k", type=int, default=7, help="drafts per pass the engine ran (acceptance is capped at k)")
ap.add_argument("--limit-passes", type=int, default=0, help="replay only the first N passes of each request (0 = all)")
ap.add_argument("--noise", choices=["none", "cpu"], default="none", help="also run the reference on this device for a noise floor")
ap.add_argument("--noise-passes", type=int, default=24, help="passes (evenly spread) the noise floor runs on")
ap.add_argument("--no-incremental", action="store_true", help="rebuild the context cache from scratch every pass (slow, a cross-check)")
ap.add_argument("--save-layers", action="store_true", help="save the reference's per-layer rows (float32, first t rows) of every pass to <out>/<name>.layers.npz")
ap.add_argument("--head", choices=["auto", "bf16", "fp8"], default="auto", help="the head the reference's logits and tail read: the checkpoint's bf16, or the engine's recipe default for the 27B "
               "(INTARSIA_LMHEAD_QUANT=fp8: e4m3 with one f32 scale amax/448 per tensor, dequantised to bf16 as the engine's tail reads it). auto = what the dump's "
               "meta.json says the engine ran; any other value than the dump's is refused unless --force-head (a bf16 reference on an fp8-head engine dump disagrees through the head, not the port)")
ap.add_argument("--force-head", action="store_true", help="allow --head to differ from the dump's head (e.g. to measure what the fp8 head costs the draft)")
ap.add_argument("--rows", type=int, default=0, help="block rows the reference runs (0 = the config's block, 16)")
args = ap.parse_args()

os.environ.setdefault("DSPINE_TOPK", "two")
sys.path.insert(0, str(Path(args.repo) / "dspine_mlx"))
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from dspine_mlx import model as dm
from dspine_mlx import bridge as db
from dspine_mlx import propose as dp
from dspine_mlx.target import HeadAccess

out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)
dump_heads = {json.loads(m.read_text()).get("head", "unknown") for d in args.dumps for m in Path(d).glob("req-*/meta.json")}
if len(dump_heads) != 1:
    raise SystemExit(f"the dumps' heads are {sorted(dump_heads)}: replay one head format at a time")
dump_head = dump_heads.pop()
if args.head == "auto":
    if dump_head == "unknown":
        raise SystemExit("dump meta.json has no `head`: pass --head bf16|fp8 --force-head")
    args.head = dump_head
elif args.head != dump_head and not args.force_head:
    raise SystemExit(f"--head {args.head} but the engine ran a {dump_head} head: a mismatched reference disagrees through the head, not the port. Use --head auto, or --force-head")
logf = open(out_dir / "run.log", "a")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    logf.write(s + "\n")
    logf.flush()


def bf_to_f32(u16):
    """bf16 bit patterns (uint16 array) -> float32 numpy."""
    return (u16.astype(np.uint32) << 16).view(np.float32)


def mx_to_np(x):
    return np.array(x.astype(mx.float32))


def to_mx_bf16(raw, dtype):
    if dtype == "bf16":
        return mx.array(raw.view(np.uint16)).view(mx.bfloat16)
    if dtype == "f16":
        return mx.array(raw.view(np.float16)).astype(mx.bfloat16)
    return mx.array(raw.view(np.float32)).astype(mx.bfloat16)


# ---------------------------------------------------------------------------------------------------------------- draft + target
draft_cfg = json.loads((Path(args.draft) / "config.json").read_text())
W = int(draft_cfg["sliding_window"]) if draft_cfg.get("use_sliding_window") and draft_cfg.get("sliding_window") else 0
mlx_dir = out_dir / "draft-mlx"  # the reference refuses sliding windows: a stripped config beside symlinked weights
mlx_dir.mkdir(exist_ok=True)
stripped = dict(draft_cfg)
stripped.update({"layer_types": ["full_attention"] * int(draft_cfg["num_hidden_layers"]), "use_sliding_window": False, "sliding_window": None})
(mlx_dir / "config.json").write_text(json.dumps(stripped, indent=1))
for f in Path(args.draft).glob("model*.safetensors"):
    link = mlx_dir / f.name
    if not link.exists():
        link.symlink_to(f.resolve())
t0 = time.time()
model, cfg = dm.load_draft(mlx_dir)
H, V, BLOCK = cfg.hidden_size, cfg.vocab_size, cfg.block_size
R, K = cfg.bridge.rank, cfg.head_topk
NT = len(cfg.target_layer_ids)
T_REF = args.rows or BLOCK


def find_tensor(names):
    tdir = Path(args.target)
    idx = tdir / "model.safetensors.index.json"
    if idx.exists():
        wm = json.loads(idx.read_text())["weight_map"]
        for n in names:
            if n in wm:
                return mx.load(str(tdir / wm[n]))[n]
    else:
        for f in sorted(tdir.glob("*.safetensors")):
            d = mx.load(str(f))
            for n in names:
                if n in d:
                    return d[n]
    raise SystemExit(f"none of {names} in {tdir}")


embed_w = find_tensor(["model.language_model.embed_tokens.weight", "model.embed_tokens.weight", "language_model.model.embed_tokens.weight"]).astype(mx.bfloat16)
head_w = find_tensor(["lm_head.weight", "language_model.lm_head.weight", "model.language_model.lm_head.weight"]).astype(mx.bfloat16)
assert tuple(embed_w.shape) == (V, H) and tuple(head_w.shape) == (V, H), (embed_w.shape, head_w.shape, V, H)
if args.head == "fp8":
    amax = max(float(mx.max(mx.abs(head_w[i:i + 16384].astype(mx.float32))).item()) for i in range(0, V, 16384))
    ws = amax / 448.0 if amax > 0 else 1.0
    head_w = mx.concatenate([(mx.from_fp8(mx.to_fp8(head_w[i:i + 16384].astype(mx.float32) / ws), dtype=mx.float32) * ws).astype(mx.bfloat16) for i in range(0, V, 16384)], axis=0)
    mx.eval(head_w)
embed = nn.Embedding(V, H)
embed.weight = embed_w
lm = nn.Linear(H, V, bias=False)
lm.weight = head_w
head = HeadAccess(lm)


class Target:  # `bind` wants an `embed_tokens`
    embed_tokens = embed


model.bind(Target)
bridge = model.selected_prefix_bridge
tg, tsv = db.build_serial_tables(
    head_rows=head.rows_range, vocab_size=V, norm_w=model.norm.weight, write_w=bridge.feature_write_weights[-1],
    pv_w=bridge.packed_weight[R:], codebook=bridge.codebook, chunk=8192)
tail = dp.RefinementTail(model, head, tg, tsv)
say(f"reference loaded in {time.time() - t0:.1f} s: head={args.head} H={H} V={V} block={BLOCK} taps={cfg.target_layer_ids} K={K} R={R} window={W or 'none'} device={mx.default_device()}")

# a sliding window the reference does not implement: an explicit boolean mask in sdpa (qpos - kpos < W, bottom-right aligned), the
# FlashAttention window_left = W - 1 convention the engine and the serving path use. Only changes the result past W keys.
_sdpa = mx.fast.scaled_dot_product_attention


def sdpa_window(q, k, v, scale, mask=None):
    sq, sk = q.shape[2], k.shape[2]
    qpos = sk - sq + mx.arange(sq)[:, None]
    kpos = mx.arange(sk)[None, :]
    return _sdpa(q, k, v, scale=scale, mask=(kpos <= qpos) & (qpos - kpos < W))


# per-layer rows: (normed, out, raw) of every bridge call
_orig_call = db.DSpineCodeBridge.__call__
_log = []


def logged(self, *a, **kw):
    r = _orig_call(self, *a, **kw)
    _log.append((r[0], r[1], a[0]))
    return r


db.DSpineCodeBridge.__call__ = logged
if W:
    mx.fast.scaled_dot_product_attention = sdpa_window


def run_block(caches, ctx_hidden, anchor, T, want_layers):
    """One reference pass. `caches` already hold the context (ctx_hidden None) or ctx_hidden [1,S,NT*H] is given with fresh caches."""
    ids = mx.array([[anchor] + [cfg.mask_token_id] * (T - 1)], dtype=mx.int32)
    model.set_block_size(T)
    _log.clear()
    if ctx_hidden is not None:
        model.append_context(ctx_hidden, caches)
    hidden, stash = model(None, ids, caches)
    rows = hidden[0, 1:]
    logits = head.matmul(rows)  # bf16 [T-1, V]
    drafts, cand = tail.greedy(rows, stash, anchor)
    lg = mx.take_along_axis(logits, cand.astype(mx.int32), axis=-1)
    ev = [drafts, cand, lg, hidden[0], stash["raw"], stash["rms"], stash["packed"]]
    if want_layers:
        ev += [a for p in _log for a in p]
    mx.eval(*ev)
    res = dict(drafts=np.array(drafts), cand=np.array(cand), logit=mx_to_np(lg), hidden=mx_to_np(hidden[0]),
               raw=mx_to_np(stash["raw"]), rms=mx_to_np(stash["rms"]), packed=mx_to_np(stash["packed"]))
    if want_layers:
        res["layers"] = [dict(normed=mx_to_np(n[0]), out=mx_to_np(o[0]), raw=mx_to_np(r[0])) for n, o, r in _log]
    return res


def rel_l2(a, b):
    d = np.linalg.norm((a - b).astype(np.float64))
    return float(d / max(np.linalg.norm(b.astype(np.float64)), 1e-30))


# ---------------------------------------------------------------------------------------------------------------- one dump
def load_request(dump, rq_dir, n):
    meta = json.loads((rq_dir / "meta.json").read_text())
    assert meta["hidden"] == H and len(meta["taps"]) == NT, (meta, H, NT)
    assert list(meta["taps"]) == list(cfg.target_layer_ids), f"dump taps {meta['taps']} != draft's target_layer_ids {cfg.target_layer_ids}"
    ev = [json.loads(l) for l in (rq_dir / "index.jsonl").read_text().splitlines() if l.strip()]
    ids_f = dump / f"ids-{n}.json"
    ids = json.loads(ids_f.read_text()) if ids_f.exists() else None
    return meta, ev, ids


def process(dump, rq_dir, n, name):
    meta, ev, idsj = load_request(dump, rq_dir, n)
    dtype = meta["dtype"]
    esz = {"bf16": 2, "f16": 2, "f32": 4}[dtype]
    row_bytes = NT * H * esz
    tapsf = rq_dir / "taps.bin"
    nrows = tapsf.stat().st_size // row_bytes
    taps = np.memmap(tapsf, dtype=np.uint8, mode="r", shape=(nrows, row_bytes)) if nrows else None
    passes_f = np.fromfile(rq_dir / "passes.bin", dtype=np.uint8) if (rq_dir / "passes.bin").exists() else np.zeros(0, np.uint8)
    layers_f = np.memmap(rq_dir / "layers.bin", dtype=np.uint8, mode="r") if (rq_dir / "layers.bin").exists() and (rq_dir / "layers.bin").stat().st_size else None
    # committed ids: the passes' new_tokens (prompt first) up to the last anchor, then the run's ids
    toks = []
    for e in ev:
        if e["ev"] == "pass":
            assert e["tok_from"] <= len(toks)
            toks[e["tok_from"]:] = e["new_tokens"]
    if idsj is not None:
        full = list(idsj["prompt"]) + list(idsj["ids"])
        assert full[:len(toks)] == toks, "ids-N.json disagrees with the dump's committed ids"
    else:
        full = toks
        say(f"  {name}: no ids-{n}.json, the target's tokens end at the last pass's anchor (later passes lose rows)")
    # replay the ingests: position -> row of taps.bin
    rowmap = np.full(max(len(full), 1) + 64, -1, dtype=np.int64)
    ctx_len = 0
    passes = []
    for e in ev:
        if e["ev"] == "ingest":
            p0, t, off = e["pos0"], e["t"], e["off"]
            if p0 + t > len(rowmap):
                rowmap = np.concatenate([rowmap, np.full(p0 + t - len(rowmap) + 64, -1, dtype=np.int64)])
            rowmap[p0:p0 + t] = off // row_bytes + np.arange(t)
            ctx_len = p0 + t
        else:
            p0 = e["p0"]
            assert ctx_len >= p0, f"pass at {p0} but the context holds {ctx_len} rows"
            ctx_len = p0
            passes.append((e, rowmap[:p0].copy()))
    if args.limit_passes:
        passes = passes[:args.limit_passes]
    say(f"  {name}: {len(passes)} passes, {nrows} tap rows, {len(full)} ids (prompt {len(idsj['prompt']) if idsj else '?'})")

    caches, cached = model.make_cache(), np.zeros(0, dtype=np.int64)
    noise_set = set()
    if args.noise != "none" and passes:
        noise_set = set(np.unique(np.linspace(0, len(passes) - 1, min(args.noise_passes, len(passes))).astype(int)).tolist())
    rec = []
    layer_save = {}
    for pi, (e, rm) in enumerate(passes):
        p0, anchor, t = e["p0"], e["anchor"], e["t"]
        assert np.all(rm >= 0), "a context position has no tap row"

        def ctx_rows(rows_idx):
            return to_mx_bf16(np.ascontiguousarray(taps[rows_idx]).reshape(len(rows_idx), NT * H * esz).view({"bf16": np.uint16, "f16": np.float16, "f32": np.float32}[dtype]), dtype)[None]

        want_layers = layers_f is not None and e.get("layers_off") is not None or args.save_layers
        if args.no_incremental:
            caches = model.make_cache()
            res = run_block(caches, ctx_rows(rm) if p0 else None, anchor, T_REF, want_layers)
        else:
            common = 0
            m = min(len(cached), len(rm))
            if m:
                neq = np.nonzero(cached[:m] != rm[:m])[0]
                common = int(neq[0]) if len(neq) else m
            for c in caches:
                c.offset = common
            if p0 > common:
                model.append_context(ctx_rows(rm[common:]), caches)
            cached = rm.copy()
            res = run_block(caches, None, anchor, T_REF, want_layers)
        # engine side of the pass
        k = e["t"] - 1
        base = e["cand_off"]
        ci = passes_f[base:base + k * K * 4].view(np.int32).reshape(k, K)
        cl = passes_f[base + k * K * 4:base + k * K * 8].view(np.float32).reshape(k, K)
        ours = np.array(e["drafts"], dtype=np.int64)
        ref = res["drafts"].astype(np.int64)
        kk = min(k, args.k)
        want_toks = full[p0 + 1:p0 + 1 + kk]
        known = len(want_toks) == kk and (idsj is not None or p0 + 1 + kk <= len(full))

        def acc(d):
            n_ = 0
            for i in range(min(kk, len(want_toks))):
                if int(d[i]) != want_toks[i]:
                    break
                n_ += 1
            return n_

        r = dict(p0=p0, anchor=anchor, t=t, known=bool(known), acc_ours=acc(ours), acc_ref=acc(ref), ours=ours[:k].tolist(), ref=ref[:k].tolist(),
                 target=want_toks, row_agree=[int(ours[i] == ref[i]) for i in range(k)],
                 first_div=next((i + 1 for i in range(k) if ours[i] != ref[i]), 0))
        # candidates and the head's logits (bf16 keys): set agreement per row and argmax agreement
        cs, am = [], []
        for i in range(k):
            a, b = set(ci[i].tolist()), set(res["cand"][i].tolist())
            cs.append(len(a & b) / K)
            ta = int(ci[i][np.argmax(cl[i])])
            tb = int(res["cand"][i][np.argmax(res["logit"][i])])
            am.append(int(ta == tb))
        r["cand_overlap"], r["argmax_agree"] = cs, am
        # per-layer rows and the last stash (rms / q), when the dump has them
        if layers_f is not None and e.get("layers_off") is not None:
            lo, nl = e["layers_off"], meta["layers"]
            seg = layers_f[lo:lo + nl * 3 * t * H * 2 + t * 4 + t * 3 * R * 2]
            lay = seg[:nl * 3 * t * H * 2].view(np.uint16).reshape(nl, 3, t, H)
            rms_o = seg[nl * 3 * t * H * 2:nl * 3 * t * H * 2 + t * 4].view(np.float32)
            pk_o = bf_to_f32(seg[nl * 3 * t * H * 2 + t * 4:].view(np.uint16).reshape(t, 3 * R))
            ls = []
            for l in range(nl):
                o_out, o_norm, o_raw = (bf_to_f32(lay[l, j]) for j in range(3))
                rl = res["layers"][l]
                ls.append(dict(out=rel_l2(o_out[1:], rl["out"][1:t]), normed=rel_l2(o_norm[1:], rl["normed"][1:t]), raw=rel_l2(o_raw[1:], rl["raw"][1:t]),
                               out_row=[rel_l2(o_out[i], rl["out"][i]) for i in range(1, t)]))
            r["layers"] = ls
            r["stash"] = dict(rms=rel_l2(rms_o[1:], res["rms"][1:t]), packed_q=rel_l2(pk_o[1:, :R], res["packed"][1:t, :R]))
        if args.save_layers and "layers" in res:
            for l, rl in enumerate(res["layers"]):
                for key in ("out", "normed", "raw"):
                    layer_save[f"p{pi}.l{l}.{key}"] = rl[key][:t]
        if pi in noise_set:
            nc = model.make_cache()
            with mx.stream(mx.cpu):
                nres = run_block(nc, ctx_rows(rm) if p0 else None, anchor, T_REF, True)
            nd = nres["drafts"].astype(np.int64)
            r["noise"] = dict(drafts=nd[:k].tolist(), agree=[int(nd[i] == ref[i]) for i in range(k)], acc=acc(nd),
                              first_div=next((i + 1 for i in range(k) if nd[i] != ref[i]), 0),
                              ours_vs_cpu=[int(nd[i] == ours[i]) for i in range(k)],
                              layer_l2=[rel_l2(nres["layers"][l]["out"][1:], res["layers"][l]["out"][1:]) for l in range(len(nres["layers"]))] if "layers" in res else None)
        rec.append(r)
        if (pi + 1) % 10 == 0 or pi + 1 == len(passes):
            say(f"    {name}: pass {pi + 1}/{len(passes)} p0={p0} ours acc {r['acc_ours']} ref acc {r['acc_ref']} first_div {r['first_div']}")
    if args.save_layers and layer_save:
        np.savez(out_dir / f"{name}.layers.npz", **layer_save)
    return rec


def summarize(name, rec):
    s = {"name": name, "passes": len(rec), "head": args.head, "engine_head": dump_head}
    kn = [r for r in rec if r["known"]]
    s["passes_scored"] = len(kn)
    if kn:
        s["tokens_per_pass_ours"] = 1 + float(np.mean([r["acc_ours"] for r in kn]))
        s["tokens_per_pass_ref"] = 1 + float(np.mean([r["acc_ref"] for r in kn]))
    rows = [(i, a) for r in rec for i, a in enumerate(r["row_agree"])]
    s["row_agreement"] = float(np.mean([a for _, a in rows])) if rows else None
    s["row_agreement_by_row"] = {str(i + 1): float(np.mean([r["row_agree"][i] for r in rec if len(r["row_agree"]) > i])) for i in range(max((len(r["row_agree"]) for r in rec), default=0))}
    s["first_div_hist"] = dict(sorted(Counter(r["first_div"] for r in rec).items()))  # 0 = the whole chain agrees
    s["pass_identical"] = float(np.mean([r["first_div"] == 0 for r in rec])) if rec else None
    s["cand_overlap"] = float(np.mean([c for r in rec for c in r["cand_overlap"]])) if rec else None
    s["argmax_agree"] = float(np.mean([c for r in rec for c in r["argmax_agree"]])) if rec else None
    if kn:
        s["acc_len_hist_ours"] = dict(sorted(Counter(r["acc_ours"] for r in kn).items()))
        s["acc_len_hist_ref"] = dict(sorted(Counter(r["acc_ref"] for r in kn).items()))
    if rec and "layers" in rec[0]:
        nl = len(rec[0]["layers"])
        s["layer_rel_l2"] = {key: [{"median": float(np.median([r["layers"][l][key] for r in rec])), "max": float(np.max([r["layers"][l][key] for r in rec]))} for l in range(nl)] for key in ("out", "normed", "raw")}
        s["stash_rel_l2"] = {key: {"median": float(np.median([r["stash"][key] for r in rec])), "max": float(np.max([r["stash"][key] for r in rec]))} for key in ("rms", "packed_q")}
    nz = [r["noise"] for r in rec if "noise" in r]
    if nz:
        s["noise"] = dict(passes=len(nz), row_agreement_ref_gpu_vs_cpu=float(np.mean([a for n in nz for a in n["agree"]])),
                          pass_identical_ref_gpu_vs_cpu=float(np.mean([n["first_div"] == 0 for n in nz])),
                          row_agreement_ours_vs_cpu=float(np.mean([a for n in nz for a in n["ours_vs_cpu"]])),
                          first_div_hist_gpu_vs_cpu=dict(sorted(Counter(n["first_div"] for n in nz).items())))
        sub = [r for r in rec if "noise" in r]
        s["noise"]["row_agreement_ours_vs_ref_same_passes"] = float(np.mean([a for r in sub for a in r["row_agree"]]))
        s["noise"]["pass_identical_ours_vs_ref_same_passes"] = float(np.mean([r["first_div"] == 0 for r in sub]))
        kn2 = [r for r in sub if r["known"]]
        if kn2:
            s["noise"]["tokens_per_pass_same_passes"] = dict(ours=1 + float(np.mean([r["acc_ours"] for r in kn2])), ref_gpu=1 + float(np.mean([r["acc_ref"] for r in kn2])),
                                                           ref_cpu=1 + float(np.mean([r["noise"]["acc"] for r in kn2])))
        if nz[0]["layer_l2"]:
            nl = len(nz[0]["layer_l2"])
            s["noise"]["layer_out_rel_l2_gpu_vs_cpu_median"] = [float(np.median([n["layer_l2"][l] for n in nz])) for l in range(nl)]
    return s


all_rec = []
for dump in map(Path, args.dumps):
    for rq_dir in sorted(dump.glob("req-*")):
        n = int(rq_dir.name.split("-")[1])
        name = f"{dump.parent.name if dump.name == 'dump' else dump.name}.{rq_dir.name}"  # <out>/<prompt>/dump/ -> <prompt>.req-0000
        say(f"== {name}")
        t0 = time.time()
        rec = process(dump, rq_dir, n, name)
        s = summarize(name, rec)
        s["seconds"] = time.time() - t0
        (out_dir / f"{name}.json").write_text(json.dumps({"summary": s, "passes": rec}))
        say(json.dumps(s, indent=1))
        all_rec.append((name, s, rec))

# overall
if len(all_rec) > 1:
    rec = [r for _, _, rr in all_rec for r in rr]
    s = summarize("ALL", rec)
    (out_dir / "ALL.json").write_text(json.dumps({"summary": s}))
    say("== ALL\n" + json.dumps(s, indent=1))
say("\nname | passes (scored) | tok/pass ours | tok/pass ref | row agree | pass identical | cand overlap | argmax agree")
for name, s, _ in all_rec + ([("ALL", s, None)] if len(all_rec) > 1 else []):
    f = lambda x: "-" if x is None else f"{x:.3f}"
    say(f"{name} | {s['passes']} ({s['passes_scored']}) | {f(s.get('tokens_per_pass_ours'))} | {f(s.get('tokens_per_pass_ref'))} | {f(s['row_agreement'])} | {f(s['pass_identical'])} | {f(s['cand_overlap'])} | {f(s['argmax_agree'])}")
