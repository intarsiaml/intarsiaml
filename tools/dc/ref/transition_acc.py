"""Teacher-forced transition accuracy of the DCOCO corrector (fast, offline variant discriminator).

For every anchor position of the plain greedy continuation (stride --step), run the draft block with the
true greedy context, take the top-16 candidates per slot, and measure for each transition p -> p+1 with
BOTH true tokens inside the top-16: is argmax_b S_p(k_true_p, b) == k_true_{p+1}? The baseline is
argmax_b l_{p+1}(b). Needs the plain greedy cache written by dflash_ref.py (--plain-cache).

  python transition_acc.py --ckpt DIR --target DIR --prompts FILE --plain plain192.json [--step 3]
"""
import argparse, sys, json, time, numpy as np, mlx.core as mx
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dflash_ref as F, dcoco_ref as D
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True); ap.add_argument('--target', required=True)
ap.add_argument('--prompts', required=True); ap.add_argument('--plain', required=True)
ap.add_argument('--step', type=int, default=3)
args = ap.parse_args()
target, tok = load(args.target)
draft, tensors = F.load_draft(Path(args.ckpt))
F.patch_target(target, list(draft.config.target_layer_ids))
plain = json.load(open(args.plain))
prompts = [json.loads(l) for l in open(args.prompts)]
STEP = args.step
W = D.load_dcoco_weights(tensors)
V = {'argmax': None, 'A': D.Variant(), 'A_tanh_eps6': D.Variant(eps=1e-6, gelu='tanh'), 'B': D.Variant(order='B'), 'C': D.Variant(order='C'),
     'pre': D.Variant(hnorm='pre'), 'lsm': D.Variant(logit='logsoftmax'), 'lmax': D.Variant(logit='lmax')}
# extra experimental modifications done by monkey patching weights / inputs
# swap the [b;g] input order of W_fuse to [g;b]
W_swap = dict(W); Wf = W['cand_ctx_proj.weight']; W_swap['cand_ctx_proj.weight'] = mx.concatenate([Wf[:, 128:], Wf[:, :128]], axis=1)   # [g;b] order
W0 = W['trans_mlp.0.weight']
embed, lm_head = target.model.embed_tokens, target.lm_head
res = {k: [0, 0] for k in list(V) + ['fuse_gb', 'noglue']}; n_trans = 0
t0 = time.time()
for p in prompts:
    pids = F.encode_prompt(tok, p['messages']); seq = pids + plain[p['id']]['ids']
    cache = make_prompt_cache(target); target(mx.array(seq)[None], cache=cache); feat = F.feats(target)
    for a in range(len(pids), len(seq) - 16, STEP):
        dc = draft.new_cache()
        blk = mx.array([[seq[a]] + [151669] * 14])
        out = draft.block(embed, lm_head, blk, feat[:, :a], dc, False, 0, 15)
        lf = np.array(out['logits'][0].astype(mx.float32)); ids, vals = F.topk_desc(lf)
        true = np.array(seq[a + 1:a + 16])
        pos = [np.nonzero(ids[i] == true[i])[0] for i in range(15)]
        cemb = embed(mx.array(ids)).astype(mx.float32)
        lse = lf.max(-1, keepdims=True); lse = lse + np.log(np.exp(lf - lse).sum(-1, keepdims=True))
        Ss = {}
        for name, var in list(V.items()) + [('fuse_gb', D.Variant())]:
            if var is None: continue
            h = out['h_post' if var.hnorm == 'post' else 'h_pre'].astype(mx.float32)
            l = vals - lse if var.logit == 'logsoftmax' else (vals - vals[:, :1] if var.logit == 'lmax' else vals)
            Ww = W_swap if name == 'fuse_gb' else W
            Ss[name] = np.array(D.dcoco_scores(Ww, h, cemb, mx.array(l), var)['S'])
        for pidx in range(14):
            if len(pos[pidx]) == 0 or len(pos[pidx + 1]) == 0: continue
            ka, kb = pos[pidx][0], pos[pidx + 1][0]
            n_trans += 1
            res['argmax'][0] += int(np.argmax(vals[pidx + 1]) == kb)
            for name in Ss:
                res[name][0] += int(np.argmax(Ss[name][pidx, ka]) == kb)
print('transitions', n_trans, 'sec', round(time.time() - t0))
for k, v in res.items():
    if k != 'noglue': print(f'{k:12s} cond. top1 acc of slot p+1 given true slot p: {v[0]/n_trans:.4f}')
