#!/usr/bin/env python3
"""style7: compare first-token top-20 logprob files (scripts/style7/firsttok.py output).
usage (run with deepeval's venv for `tokenizers`): analyze.py --tok tokenizer.json --ref A.json --cmp B.json [--cmp C.json ...] [--replies label=result.json ...] [--json OUT]
Per file: top-1 token, mass on digit-first tokens (number-only style) from the top-20, first-token entropy proxy.
Pairs: top-1 agreement, top-1 style agreement, mean KL(ref||cmp) on the shared top-20 support (each side renormalised on the union, missing = floor),
mean |dlogit margin|, and margins on questions where the engines' first-token styles differ."""
import argparse, json, math, re, sys
from tokenizers import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--tok", required=True); ap.add_argument("--ref", required=True); ap.add_argument("--cmp", action="append", default=[])
ap.add_argument("--replies", nargs="*", default=[]); ap.add_argument("--json")
a = ap.parse_args()
tok = Tokenizer.from_file(a.tok)
cache = {}
def isdigit(t):
    if t not in cache:
        s = tok.decode([t]).strip(); cache[t] = bool(s) and bool(re.fullmatch(r"[\d$,.-]+", s)) and any(c.isdigit() for c in s)
    return cache[t]
def load(p): return {r["id"]: r for r in json.load(open(p))}
def stats(r):
    top = r["top"]; ps = [math.exp(l) for _, l in top]
    return {"top1": top[0][0], "m12": top[0][1] - top[1][1], "pd": sum(p for (t, _), p in zip(top, ps) if isdigit(t)), "d1": isdigit(top[0][0]), "mass": sum(ps)}
def kl(p, q):  # p, q: dict tok -> logprob (top-20). Union support; missing mass of each lumped into one "other" bucket, missing tokens floored at the smaller tail
    U = set(p) | set(q)
    def dist(d):
        rest = max(1e-9, 1 - sum(math.exp(v) for v in d.values())); per = rest / max(1, 150000)
        return {t: (math.exp(d[t]) if t in d else per) for t in U}, rest
    P, rp = dist(p); Q, rq = dist(q)
    return sum(P[t] * math.log(P[t] / Q[t]) for t in U) + (rp * math.log(rp / rq))
ref = load(a.ref); out = {"files": {}, "pairs": {}}
sty = {}
for lab_p in a.replies:
    lab, p = lab_p.split("=", 1); d = json.load(open(p))["benchmarks"][0]["replies"]
    sty[lab] = {r["id"]: bool(re.match(r"^\s*\$?-?[\d,]+(\.\d+)?\s*%?\.?\s*$", r["reply"])) for r in d}   # True = number-only
def summarize(name, R):
    S = {i: stats(r) for i, r in R.items()}
    n = len(S); o = {"n": n, "digit_top1": sum(s["d1"] for s in S.values()), "mean_p_digit": round(sum(s["pd"] for s in S.values()) / n, 4),
                     "mean_top20_mass": round(sum(s["mass"] for s in S.values()) / n, 4)}
    for lab, st in sty.items():
        ids = [i for i in S if i in st]
        o[f"agree_with_reply_style[{lab}]"] = f"{sum(S[i]['d1'] == st[i] for i in ids)}/{len(ids)}"
    out["files"][name] = o; return S
SR = summarize(a.ref, ref)
for c in a.cmp:
    C = load(c); SC = summarize(c, C); ids = sorted(set(ref) & set(C))
    t1 = sum(SR[i]["top1"] == SC[i]["top1"] for i in ids); sa = sum(SR[i]["d1"] == SC[i]["d1"] for i in ids)
    kls = [kl(dict(map(tuple, ref[i]["top"])), dict(map(tuple, C[i]["top"]))) for i in ids]
    dif = [i for i in ids if SR[i]["d1"] != SC[i]["d1"]]
    dpd = [SC[i]["pd"] - SR[i]["pd"] for i in ids]
    out["pairs"][f"{a.ref} vs {c}"] = {"n": len(ids), "top1_agree": t1, "style_agree": sa, "style_differ_ids": len(dif),
        "mean_KL_ref_to_cmp": round(sum(kls) / len(kls), 5), "median_KL": round(sorted(kls)[len(kls)//2], 5),
        "mean_dp_digit(cmp-ref)": round(sum(dpd) / len(dpd), 4), "mean_abs_dp_digit": round(sum(map(abs, dpd)) / len(dpd), 4),
        "style_differ_mean_margin_ref": round(sum(SR[i]["m12"] for i in dif) / max(1, len(dif)), 3), "style_differ_mean_margin_cmp": round(sum(SC[i]["m12"] for i in dif) / max(1, len(dif)), 3),
        "style_differ_margin_lt0.5_ref": sum(SR[i]["m12"] < 0.5 for i in dif), "style_differ_margin_lt0.5_cmp": sum(SC[i]["m12"] < 0.5 for i in dif),
        "style_differ_ref_digit": sum(SR[i]["d1"] for i in dif), "style_differ_cmp_digit": sum(SC[i]["d1"] for i in dif)}
print(json.dumps(out, indent=1))
if a.json: json.dump(out, open(a.json, "w"), indent=1)
