"""Paired teacher-forced comparison of block layouts r15 ([anchor]+14 masks) vs r16s ([anchor]+15 masks), both with
drafts = rows 0..14, on the same anchors of the plain greedy continuation (plain DFlash argmax drafts).
  python layout_tf.py --ckpt DIR --target DIR --prompts FILE --plain plain192.json [--step 2]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dflash_ref as F

ap = argparse.ArgumentParser()
for k in ("ckpt", "target", "prompts", "plain"): ap.add_argument("--" + k, required=True)
ap.add_argument("--step", type=int, default=2)
a = ap.parse_args()
target, tok = load(a.target); draft, _ = F.load_draft(Path(a.ckpt)); F.patch_target(target, list(draft.config.target_layer_ids))
plain = json.load(open(a.plain)); prompts = [json.loads(l) for l in open(a.prompts)]
embed, lm_head = target.model.embed_tokens, target.lm_head
acc = {"r15": [], "r16s": []}; dl = []; same_arg = []
for p in prompts:
    pids = F.encode_prompt(tok, p["messages"]); seq = pids + plain[p["id"]]["ids"]
    cache = make_prompt_cache(target); target(mx.array(seq)[None], cache=cache); feat = F.feats(target)
    for pos in range(len(pids), len(seq) - 16, a.step):
        outs = {}
        for name, rows in (("r15", 15), ("r16s", 16)):
            blk = mx.array([[seq[pos]] + [151669] * (rows - 1)])
            o = draft.block(embed, lm_head, blk, feat[:, :pos], draft.new_cache(), False, 0, 15)
            outs[name] = np.array(o["logits"][0].astype(mx.float32))
        true = np.array(seq[pos + 1:pos + 16])
        for n, lg in outs.items():
            ok = lg.argmax(-1) == true; acc[n].append(int(np.argmin(ok)) if not ok.all() else 15)
        dl.append(np.abs(outs["r15"] - outs["r16s"]).max()); same_arg.append((outs["r15"].argmax(-1) == outs["r16s"].argmax(-1)).mean())
d = np.array(acc["r16s"]) - np.array(acc["r15"])
print("anchors", len(d))
for n in acc: print(f"{n}: mean accepted {np.mean(acc[n]):.3f}")
print(f"paired diff r16s - r15: mean {d.mean():+.3f} +- {d.std(ddof=1)/np.sqrt(len(d)):.3f} (s.e.); max|dlogit| rows0-14 mean {np.mean(dl):.3f}; argmax agreement {np.mean(same_arg):.3f}")
