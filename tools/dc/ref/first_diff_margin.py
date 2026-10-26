"""For every prompt whose speculative output differs from plain greedy, report the single-step target logit gap
between the plain-greedy token and the speculative token at the first differing position (a near-tie of the
order of one bf16 ulp points to batched-vs-single-step numerics, not to a draft/verify logic error).

  python first_diff_margin.py --target DIR --prompts FILE --run dflash_run.json [--run-index 0]
"""
import argparse, json, sys
from pathlib import Path
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dflash_ref as F

ap = argparse.ArgumentParser()
ap.add_argument("--target", required=True); ap.add_argument("--prompts", required=True)
ap.add_argument("--run", required=True); ap.add_argument("--run-index", type=int, default=0)
ap.add_argument("--out", default=None)
a = ap.parse_args()
target, tok = load(a.target)
prompts = {json.loads(l)["id"]: json.loads(l) for l in open(a.prompts)}
run = json.load(open(a.run)); plain = run["plain"]; res = []
for q in run["runs"][a.run_index]["prompts"]:
    if q["equals_plain_greedy"]:
        continue
    f = q["first_diff"]; pids = F.encode_prompt(tok, prompts[q["id"]]["messages"]); ref = plain[q["id"]]["ids"]
    if f >= len(q["ids"]) or f >= len(ref):
        res.append({"id": q["id"], "first_diff": f, "note": "length differs only"}); continue
    cache = make_prompt_cache(target)
    lg = target(mx.array(pids)[None], cache=cache)[0, -1]
    for t in ref[:f]:
        lg = target(mx.array([[t]]), cache=cache)[0, -1]
    lg = lg.astype(mx.float32)
    top2 = mx.sort(lg)[-2:].tolist()
    res.append({"id": q["id"], "first_diff": f, "plain_token": ref[f], "spec_token": q["ids"][f],
                "plain_logit": lg[ref[f]].item(), "spec_logit": lg[q["ids"][f]].item(),
                "gap_single_step": lg[ref[f]].item() - lg[q["ids"][f]].item(), "top2": top2})
    print(res[-1], flush=True)
if a.out:
    Path(a.out).write_text(json.dumps(res, indent=1))
