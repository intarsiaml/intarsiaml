"""KL and top-1 agreement between two intarsia-golden-v1 files (full-vocabulary fp16 log-probs).

A golden may also be a self-golden written by `intarsia accuracy --write-self-golden F --self-topk <vocab>`.
Usage (inside an image with torch + safetensors, e.g. dsv41-serve:v4):
    python3 -I golden_kl.py <reference golden> <other golden> [--json]

Same maths as `intarsia accuracy`'s full-vocabulary comparison (docs/accuracy.md, crates/intarsia-cli/src/cmd/accuracy.rs
`compare_pos`): both rows are fp16 log-probs read as f64 and renormalised (log-softmax over the stored values), and
KL(reference || other) = sum_v p_ref(v) * (log p_ref(v) - log p_other(v)). Top-1 agreement: argmax of the other row equals
the reference's argmax (a tie is counted when the other's argmax has a reference log-prob >= the reference's best). Positions
are pooled over all prompts (row t of prompt i predicts token t+1); the ids of the two files must be identical. A width
mismatch (vocab padding) is cut to the common prefix; the mass shift that causes is reported.
"""
import argparse
import json
import sys

import torch
from safetensors import safe_open


def load(path):
    out = {}
    with safe_open(path, "pt") as f:
        meta = f.metadata()
        for k in f.keys():
            out[k] = f.get_tensor(k)
    # an `intarsia accuracy --write-self-golden F --self-topk <vocab>` file (intarsia-golden-topk-v1 with every id kept)
    # is expanded to full rows, so intarsia's own distribution can be compared like a golden
    for k in [k for k in out if k.startswith("topk_ids_")]:
        i = k.split("_")[2]
        ids, lp = out.pop(k).long(), out.pop(f"topk_logprobs_{i}").float()
        vocab = int(meta["vocab"]) if meta and "vocab" in meta else int(ids.max()) + 1
        if ids.shape[1] != vocab or (ids < 0).any():
            sys.exit(f"{path}: top-{ids.shape[1]} file does not keep the whole vocabulary of {vocab}; use --self-topk {vocab}")
        full = torch.full((ids.shape[0], vocab), -1e30)
        out[f"logprobs_{i}"] = full.scatter_(1, ids, lp)
    return out, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("other")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r, rm = load(a.ref)
    o, om = load(a.other)
    idx = sorted(int(k.split("_")[1]) for k in r if k.startswith("ids_"))
    oidx = sorted(int(k.split("_")[1]) for k in o if k.startswith("ids_"))
    if idx != oidx:
        sys.exit(f"prompt sets differ: {idx} vs {oidx}")
    kls, top1, tie, pad_mass = [], [], [], 0.0
    per = []
    for i in idx:
        if not torch.equal(r[f"ids_{i}"], o[f"ids_{i}"]):
            sys.exit(f"token ids of prompt {i} differ")
        lr, lo = r[f"logprobs_{i}"].double(), o[f"logprobs_{i}"].double()
        if lr.shape[0] != lo.shape[0]:
            sys.exit(f"prompt {i}: {lr.shape[0]} vs {lo.shape[0]} rows")
        w = min(lr.shape[1], lo.shape[1])
        if lr.shape[1] != lo.shape[1]:
            big = lr if lr.shape[1] > w else lo
            pad_mass = max(pad_mass, float((torch.logsumexp(big, -1) - torch.logsumexp(big[:, :w], -1)).abs().max()))
        lr, lo = lr[:, :w], lo[:, :w]
        lr = lr - torch.logsumexp(lr, -1, keepdim=True)
        lo = lo - torch.logsumexp(lo, -1, keepdim=True)
        kl = (lr.exp() * (lr - lo)).nan_to_num(0.0).sum(-1).clamp_min(0.0)
        ar, ao = lr.argmax(-1), lo.argmax(-1)
        t1 = ar == ao
        tie_ = (~t1) & (lr.gather(1, ao[:, None])[:, 0] >= lr.gather(1, ar[:, None])[:, 0])
        kls.append(kl)
        top1.append(t1)
        tie.append(tie_)
        per.append((i, kl.numel(), float(kl.mean()), float(t1.float().mean())))
    kl = torch.cat(kls)
    t1 = torch.cat(top1)
    ti = torch.cat(tie)
    res = {
        "reference": a.ref,
        "other": a.other,
        "reference_model": (rm or {}).get("model"),
        "other_model": (om or {}).get("model"),
        "positions": kl.numel(),
        "kl_mean": float(kl.mean()),
        "kl_p99": float(torch.quantile(kl, 0.99)),
        "kl_max": float(kl.max()),
        "top1": float(t1.float().mean()),
        "top1_ties": int(ti.sum()),
        "top1_excl_ties": float((t1 | ti).float().mean()),
        "width_mismatch_max_abs_lse_shift": pad_mass,
        "per_prompt": [{"prompt": i, "n": n, "kl_mean": m, "top1": t} for i, n, m, t in per],
    }
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        print(f"{res['positions']} positions  KL(ref||other) mean {res['kl_mean']:.3e}  p99 {res['kl_p99']:.3e}  "
              f"max {res['kl_max']:.3e}  top-1 {100 * res['top1']:.2f}% (ties {res['top1_ties']}, excl. ties "
              f"{100 * res['top1_excl_ties']:.2f}%)")
        for p in res["per_prompt"]:
            print(f"  prompt {p['prompt']}: n={p['n']} KL mean {p['kl_mean']:.3e} top-1 {100 * p['top1']:.2f}%")


main()
