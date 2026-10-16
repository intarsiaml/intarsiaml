#!/usr/bin/env python3
"""Cross-compare top-k logprob dumps (vLLM golden, intarsia --write-self-golden dumps of any backend) per position.

    uv run --project tools python scripts/acc35b/xcompare.py NAME=FILE [NAME=FILE ...] [--mode decode|prefill]
        [--pairs A:B,...] [--prompt P] [--positions] [--top 10]

For every ordered pair (reference A, candidate B) prints the mean / max 21-way KL(A || B) per prompt and overall, the
top-1 agreement and the largest positions, with the same 21-way definition as `intarsia accuracy` (reference top-20 ids
plus one tail bucket on both sides; the candidate's mass on an id outside its own top-k is taken as its k-th
probability, an upper bound; this is counted and reported). With --positions, prints for the worst positions the top-3
of every dump, so a near-tie in the reference is visible.
"""
import argparse, json, math, sys
import numpy as np
from safetensors import safe_open

TAIL_K = 20


def load(path):
    f = safe_open(path, "np")
    meta = f.metadata() or {}
    keys = set(f.keys())
    names = None
    if "prompts" in meta:
        try:
            names = [p["name"] for p in json.loads(meta["prompts"])]
        except Exception:
            names = None
    out = {}
    i = 0
    while f"ids_{i}" in keys:
        ids = f.get_tensor(f"ids_{i}")
        d = {"ids": ids, "pre": {}, "dec": {}}
        if f"topk_ids_{i}" in keys:
            ti, tl = f.get_tensor(f"topk_ids_{i}"), f.get_tensor(f"topk_logprobs_{i}")
            for p in range(ti.shape[0]):
                if ti[p, 0] >= 0:
                    d["pre"][p] = (ti[p], tl[p])
        if f"dec_pos_{i}" in keys:
            dp = f.get_tensor(f"dec_pos_{i}")
            di, dl = f.get_tensor(f"dec_topk_ids_{i}"), f.get_tensor(f"dec_topk_logprobs_{i}")
            di = di.reshape(len(dp), -1); dl = dl.reshape(len(dp), -1)
            for j, p in enumerate(dp):
                d["dec"][int(p)] = (di[j], dl[j])
        name = names[i] if names and i < len(names) else f"p{i}"
        out[name] = d
        i += 1
    return out, meta


def row(d, mode, p):
    """Rows of a dump for a mode; a vLLM golden has only prompt rows, and its decode-mode reference is the same row."""
    if mode == "decode":
        if p in d["dec"]:
            return d["dec"][p]
        return d["pre"].get(p)
    return d["pre"].get(p)


def kl21(ref, cand):
    rid, rlp = ref
    cid, clp = cand
    m = rid >= 0
    rid, rlp = rid[m][:TAIL_K], rlp[m][:TAIL_K].astype(np.float64)
    cm = cid >= 0
    cmap = dict(zip(cid[cm].tolist(), clp[cm].astype(np.float64).tolist()))
    floor = float(clp[cm].min())
    pr = np.exp(rlp)
    miss = 0
    pc = []
    for t in rid.tolist():
        if t in cmap:
            pc.append(math.exp(cmap[t]))
        else:
            pc.append(math.exp(floor)); miss += 1
    pc = np.array(pc)
    tr = max(1.0 - pr.sum(), 1e-12)
    tc = max(1.0 - pc.sum(), 1e-12)
    pc = np.maximum(pc, 1e-30)
    kl = float(np.sum(pr * (rlp - np.log(pc))) + tr * (math.log(tr) - math.log(tc)))
    top1 = int(cid[cm][np.argmax(clp[cm])]) == int(rid[np.argmax(rlp)])
    return max(kl, 0.0), top1, miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--mode", default="decode")
    ap.add_argument("--pairs", default="")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--positions", action="store_true")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    D = {}
    for s in a.dumps:
        n, path = s.split("=", 1)
        D[n] = load(path)[0]
    names = list(D)
    pairs = [tuple(p.split(":")) for p in a.pairs.split(",") if p] or [(x, y) for x in names for y in names if x != y]
    result = {}
    for ra, cb in pairs:
        A, B = D[ra], D[cb]
        allk, flips, misses, worst = [], 0, 0, []
        per = {}
        for pname in A:
            if a.prompt and pname != a.prompt:
                continue
            if pname not in B:
                continue
            if not np.array_equal(A[pname]["ids"], B[pname]["ids"]):
                print(f"WARN {pname}: ids differ between {ra} and {cb}", file=sys.stderr); continue
            src = A[pname]["dec"] if a.mode == "decode" and A[pname]["dec"] else A[pname]["pre"]
            posl = sorted(B[pname]["dec"]) if a.mode == "decode" and B[pname]["dec"] else sorted(src)
            ks = []
            for p in posl:
                r, c = row(A[pname], a.mode, p), row(B[pname], a.mode, p)
                if r is None or c is None:
                    continue
                k, t1, mi = kl21(r, c)
                ks.append(k); misses += mi; flips += (not t1)
                worst.append((k, pname, p, t1))
            if ks:
                per[pname] = (len(ks), float(np.mean(ks)), float(np.max(ks)))
                allk += ks
        if not allk:
            continue
        worst.sort(reverse=True)
        result[f"{ra}:{cb}"] = {"mean": float(np.mean(allk)), "max": float(np.max(allk)), "n": len(allk), "flips": flips,
                                "per_prompt": per, "worst": [(w[1], w[2], w[0]) for w in worst[: a.top]]}
        print(f"== {a.mode}  KL21({ra} || {cb})  n={len(allk)}  mean={np.mean(allk):.4e}  max={np.max(allk):.4f}  "
              f"top-1 disagree={flips}  cand-missing-ids={misses}")
        for pname, (n, mn, mx) in per.items():
            print(f"   {pname:16s} n={n:3d} mean={mn:.4e} max={mx:.4f}")
        print("   worst: " + ", ".join(f"{w[1]}@{w[2]}={w[0]:.3f}{'' if w[3] else '*'}" for w in worst[: a.top]))
    if a.positions:
        seen = set()
        for key, r in result.items():
            for pname, p, _ in r["worst"][:3]:
                if (pname, p) in seen:
                    continue
                seen.add((pname, p))
                print(f"-- {pname} pos {p}: top-3 per dump (token:prob)")
                for n in names:
                    rr = row(D[n][pname], a.mode, p) if pname in D[n] else None
                    if rr is None:
                        continue
                    ids, lp = rr
                    o = np.argsort(-lp)[:3]
                    print(f"   {n:10s} " + "  ".join(f"{int(ids[j])}:{math.exp(lp[j]):.4f}" for j in o))
    if a.json:
        json.dump(result, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
