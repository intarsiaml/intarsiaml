"""Paired off/on analysis for a localeval result pair. usage: analyze_pair.py gsm|long OFF_RESULT.json[,more] ON_RESULT.json[,more]
(comma lists are concatenated in order; pairs are matched by item id/name; items present in only one arm are dropped and counted.)"""
import json, math, sys
from math import comb
from tokenizers import Tokenizer
tk = Tokenizer.from_file("/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4/tokenizer.json")
kind = sys.argv[1]

def load(paths):
    d = {}
    for p in paths.split(","):
        r = json.load(open(p))
        if kind == "gsm":
            for x in r["benchmarks"][0]["replies"]:
                d[x["id"]] = dict(text=x["reply"] or "", ok=bool(x["flexible"]), trunc=bool(x["truncated"]), ntok=x["completion_tokens"])
        else:
            for c in r["datasets"][0]["cases"]:
                sp = c.get("speed") or {}
                exp = " ".join((c["expected_output"] or "").lower().split())
                ok = exp in " ".join((c["actual_output"] or "").lower().split())
                d[c["name"]] = dict(text=c["actual_output"] or "", ok=ok, trunc=(sp.get("completion_tokens") or 0) >= 128, ntok=sp.get("completion_tokens"), err=c.get("error"), ttft=sp.get("ttft_s"), dec=sp.get("decode_tok_s"), pt=sp.get("prompt_tokens"))
    return d

o, n = load(sys.argv[2]), load(sys.argv[3])
ids = sorted(set(o) & set(n))
a = [i for i in ids if o[i]["ok"] and not n[i]["ok"]]
c = [i for i in ids if n[i]["ok"] and not o[i]["ok"]]
N = len(ids); k = min(len(a), len(c)); m = len(a) + len(c)
p = 1.0 if m == 0 else min(1.0, 2 * sum(comb(m, j) for j in range(k + 1)) / 2 ** m)
d = (len(c) - len(a)) / N
se = math.sqrt(max(m - (len(c) - len(a)) ** 2 / N, 0)) / N
div = []
for i in ids:
    if o[i]["text"] != n[i]["text"]:
        x, y = tk.encode(o[i]["text"], add_special_tokens=False).ids, tk.encode(n[i]["text"], add_special_tokens=False).ids
        j = next((t for t in range(min(len(x), len(y))) if x[t] != y[t]), min(len(x), len(y)))
        div.append((i, j))
import statistics as st
res = dict(n_pairs=N, only_off=len(set(o) - set(n)), only_on=len(set(n) - set(o)),
    off_score=sum(o[i]["ok"] for i in ids) / N, on_score=sum(n[i]["ok"] for i in ids) / N,
    off_right_on_wrong=len(a), on_right_off_wrong=len(c), mcnemar_p=p, diff=d, ci95=[d - 1.96 * se, d + 1.96 * se],
    trunc_off=sum(o[i]["trunc"] for i in ids), trunc_on=sum(n[i]["trunc"] for i in ids),
    items_text_differs=len(div), first_diverge_token_median=(st.median(j for _, j in div) if div else None),
    first_diverge_token_min=(min(j for _, j in div) if div else None), first_diverge_token_max=(max(j for _, j in div) if div else None),
    discordant_off_right_on_wrong=a, discordant_on_right_off_wrong=c, diverge=div)
if kind == "long":
    for tag, dd in (("off", o), ("on", n)):
        t = [dd[i]["ttft"] for i in ids if dd[i].get("ttft")]; v = [dd[i]["dec"] for i in ids if dd[i].get("dec")]
        res[f"ttft_median_{tag}"] = st.median(t) if t else None; res[f"decode_tok_s_median_{tag}"] = st.median(v) if v else None
        res[f"ttft_first5_{tag}"] = t[:5]; res[f"prompt_tokens_median_{tag}"] = st.median([dd[i]["pt"] for i in ids if dd[i].get("pt")] or [0])
print(json.dumps(res, indent=1))
