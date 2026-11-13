#!/usr/bin/env python3
"""vllm-null: vLLM-vs-vLLM thinking-length spread (the null) against intarsiaml (Bc), on the same GSM8K ids.
usage: null.py [--first N=85] label=result.json ... --ref Bc  (the run compared to every vLLM run; others are vLLM runs, first is V1)
Length = localeval `completion_tokens` (reasoning + answer, as in the soak and think-len docs). Bootstrap: paired, 20000 resamples, seed 1."""
import json, math, random, statistics as st, sys

def load(p):
    return {r["id"]: r for r in json.load(open(p))["benchmarks"][0]["replies"]}
ok = lambda r: r["flexible"] if isinstance(r["flexible"], bool) else str(r["flexible"]).lower() == "true"
def sign_p(a, b):
    n = a + b; k = min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
def boot(x, y, B=20000, seed=1):
    rnd = random.Random(seed); n = len(x); r = []
    for _ in range(B):
        ix = [rnd.randrange(n) for _ in range(n)]
        r.append(sum(x[i] for i in ix) / sum(y[i] for i in ix))
    r.sort(); return r[int(.025 * B)], r[int(.975 * B)]
def gm(x, y): return math.exp(st.mean(math.log(a / b) for a, b in zip(x, y)))

argv = sys.argv[1:]; first = 85; ref = "Bc"; runs = {}; i = 0
while i < len(argv):
    if argv[i] == "--first": first = int(argv[i + 1]); i += 2
    elif argv[i] == "--ref": ref = argv[i + 1]; i += 2
    else: k, p = argv[i].split("=", 1); runs[k] = load(p); i += 1
ids = sorted(runs[ref])[:first]
L = {k: [int(r[i]["completion_tokens"]) for i in ids] for k, r in runs.items()}
V = [k for k in runs if k != ref]
L["Vmean"] = [st.mean(L[k][j] for k in V) for j in range(len(ids))]
P = print
P(f"n={len(ids)} questions ({ids[0]}..{ids[-1]}), length = completion_tokens")
P("\n== per run: mean / median / max / >2500 / >4000 tokens, accuracy, truncated, finish_reason stop")
for k in V + [ref]:
    rs = [runs[k][i] for i in ids]
    P(f"  {k}: mean {st.mean(L[k]):.1f} median {st.median(L[k]):.0f} max {max(L[k])} >2500 {sum(x>2500 for x in L[k])} >4000 {sum(x>4000 for x in L[k])} "
      f"acc {sum(ok(r) for r in rs)}/{len(rs)} truncated {sum(str(r.get('truncated')).lower()=='true' for r in rs)} stop {sum(r['finish_reason']=='stop' for r in rs)}")
P(f"  Vmean (per-question mean of {','.join(V)}): mean {st.mean(L['Vmean']):.1f}")

def pair(a, b, name):
    x, y = L[a], L[b]; lo, hi = boot(x, y)
    up = sum(p > q for p, q in zip(x, y)); dn = sum(p < q for p, q in zip(x, y))
    P(f"  {name:>14}: ratio of means {sum(x)/sum(y):.3f} (boot 95% CI {lo:.3f} to {hi:.3f}); geo-mean ratio {gm(x,y):.3f}; first longer on {up}, shorter on {dn}; sign p {sign_p(up,dn):.3f}")
    return sum(x) / sum(y)
P("\n== the null: vLLM against vLLM (first run named is the numerator)")
nulls = [pair(V[j], V[i], f"{V[j]} / {V[i]}") for i in range(len(V)) for j in range(i + 1, len(V))]
P(f"  null ratios of means: {', '.join(f'{r:.3f}' for r in nulls)}; vLLM run means: {', '.join(f'{st.mean(L[k]):.0f}' for k in V)} (spread of run means {max(st.mean(L[k]) for k in V)/min(st.mean(L[k]) for k in V):.3f}x)")
P(f"\n== {ref} against vLLM")
for k in V + ["Vmean"]: pair(ref, k, f"{ref} / {k}")
P(f"  {ref} against the vLLM runs pooled (sum of {ref} over sum of all vLLM lengths): {sum(L[ref])*len(V)/sum(sum(L[k]) for k in V):.3f}")

P("\n== outliers: the 5 largest per-question excesses of ref over V1 (the think-len set), and what V2/V3/Vmean do on them")
d1 = sorted(range(len(ids)), key=lambda j: L[ref][j] - L[V[0]][j], reverse=True)
tot = sum(L[ref][j] - L[V[0]][j] for j in range(len(ids)))
P("  question: " + " ".join(f"{k:>7}" for k in V + ['Vmean', ref]) + "   excess over V1 | over Vmean")
for j in d1[:5]:
    P(f"  {ids[j]}: " + " ".join(f"{L[k][j]:7.0f}" for k in V + ['Vmean', ref]) + f"   {L[ref][j]-L[V[0]][j]:6d} | {L[ref][j]-L['Vmean'][j]:7.0f}")
top5 = sum(L[ref][j] - L[V[0]][j] for j in d1[:5])
P(f"  top 5 carry {top5} of {tot} total excess over V1 ({top5/tot:.0%})")
dm = sorted(range(len(ids)), key=lambda j: L[ref][j] - L["Vmean"][j], reverse=True)
totm = sum(L[ref][j] - L["Vmean"][j] for j in range(len(ids)))
ov = [ids[j] for j in d1[:5] if j in dm[:5]]
P(f"  top 5 excess over Vmean: {[ids[j] for j in dm[:5]]} carry {sum(L[ref][j]-L['Vmean'][j] for j in dm[:5]):.0f} of {totm:.0f}; overlap with the V1 top 5: {len(ov)} {ov}")
for k in V[1:]:
    dk = sorted(range(len(ids)), key=lambda j: L[ref][j] - L[k][j], reverse=True)
    P(f"  top 5 excess over {k}: {[ids[j] for j in dk[:5]]}; overlap with the V1 top 5: {len(set(dk[:5]) & set(d1[:5]))}")
P(f"  of the V1 top 5, ref is still longer than V2/V3/Vmean on: " + ", ".join(f"{k} {sum(L[ref][j]>L[k][j] for j in d1[:5])}/5" for k in V[1:] + ['Vmean']))
P(f"  how the V1 top 5 are also 'long' for vLLM itself: V1 length rank among its 85: {[sorted(L[V[0]],reverse=True).index(L[V[0]][j])+1 for j in d1[:5]]}")
x = [L[ref][j] for j in range(len(ids)) if j not in d1[:5]]; y = [L['Vmean'][j] for j in range(len(ids)) if j not in d1[:5]]
P(f"  ref / Vmean without the V1 top 5: {sum(x)/sum(y):.3f}")

P("\n== accuracy, paired on the same ids (right-only-in-a, right-only-in-b, McNemar exact p)")
names = V + [ref]
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        ra = [ok(runs[a][q]) for q in ids]; rb = [ok(runs[b][q]) for q in ids]
        P(f"  {a} vs {b}: {sum(ra)} vs {sum(rb)} of {len(ids)}; {sum(p and not q for p,q in zip(ra,rb))} vs {sum(q and not p for p,q in zip(ra,rb))}; p {sign_p(sum(p and not q for p,q in zip(ra,rb)), sum(q and not p for p,q in zip(ra,rb))):.3f}")
wrong = {k: [q for q in ids if not ok(runs[k][q])] for k in names}
P("  wrong ids: " + "; ".join(f"{k} {v}" for k, v in wrong.items()))
P("\n== final answer text identical to V1 / bare-number answers / worked answers (reply)")
for k in names:
    P(f"  {k}: identical to V1 {sum(runs[k][q]['reply'].strip()==runs[V[0]][q]['reply'].strip() for q in ids)}; bare number {sum(str(runs[k][q].get('bare_number')).lower()=='true' for q in ids)}")
