#!/usr/bin/env python3
"""think-len: compare the vLLM probe with the intarsia probe(s), position by position.
usage: analyze.py <raw dir> <tokenizer.json> [intarsia arm names, default bf16,fp8]   (writes <raw dir>/analysis.txt)"""
import json, math, re, statistics as st, sys
from tokenizers import Tokenizer

raw, tokp = sys.argv[1].rstrip("/"), sys.argv[2]
arms = (sys.argv[3] if len(sys.argv) > 3 else "bf16,fp8").split(",")
tok = Tokenizer.from_file(tokp)
dec = lambda i: tok.decode([int(i)], skip_special_tokens=False)
import gzip, os
load = lambda f: json.load(gzip.open(f"{raw}/{f}.gz") if os.path.exists(f"{raw}/{f}.gz") else open(f"{raw}/{f}"))
E_ = {"vllm": {(r["id"], r["kind"]): r for r in load("probe-vllm.json")}}
for a in arms:
    try:
        E_[a] = {(r["id"], r["kind"]): r for r in load(f"probe-vllm-b.json" if a == "vllm-b" else f"probe-intarsia-{a}.json")}
    except (FileNotFoundError, OSError):
        pass
out = []
P = lambda *a: out.append(" ".join(str(x) for x in a))


def fill_ids(D):
    """intarsia's completions reply has no token ids unless `return_token_ids` is set; with greedy decoding the sampled id is the
    top-1 id, and `chosen_lp == top-1 logprob` is checked here (the reply's token_logprobs is the sampled token's own)."""
    for r in D.values():
        if r["ids"] is None:
            r["ids"] = [int(max(tp, key=tp.get)) for tp in r["tops"]]
            r["lp_mismatch"] = sum(abs(c - max(tp.values())) > 1e-6 for c, tp in zip(r["chosen_lp"], r["tops"]))


for _D in E_.values():
    fill_ids(_D)


def probs(row):  # token id (str) -> prob
    return {k: math.exp(v) for k, v in row.items()}


def top1(row):
    return max(row, key=row.get)


def is_bare(i):  # a token that starts a bare numeric answer
    t = dec(i).strip()
    return bool(re.match(r"^[\$\d]", t)) if t else False


P("greedy check: chosen token is the engine's top-1 at every generated position")
for n, D in E_.items():
    bad = tot = 0
    for r in D.values():
        for t, tp in zip(r["ids"], r["tops"]):
            tot += 1
            bad += str(t) != top1(tp)
        bad += r.get("lp_mismatch", 0)
    P(f"  {n}: {bad} of {tot} positions where the sampled id is not top-1 (intarsia: sampled logprob != top-1 logprob)")

for kind, label, ids in (("off", "thinking OFF, first answer token (200 prompts)", None), ("tf", "thinking ON, first token after the vLLM trace + </think> (85)", None)):
    P(f"\n== {label}: P(bare-number start) = summed prob of top-20 tokens that start with a digit or $")
    base = E_["vllm"]
    for n, D in E_.items():
        if n == "vllm":
            continue
        keys = sorted(k for k in base if k[1] == kind and k in D)
        if not keys:
            continue
        pv = [sum(p for i, p in probs(base[k]["tops"][0]).items() if is_bare(i)) for k in keys]
        pi = [sum(p for i, p in probs(D[k]["tops"][0]).items() if is_bare(i)) for k in keys]
        d = [b - a for a, b in zip(pv, pi)]
        lo = lambda p: math.log(max(p, 1e-6) / max(1 - p, 1e-6))
        dl = [lo(b) - lo(a) for a, b in zip(pv, pi)]
        flips = sum(top1(base[k]["tops"][0]) != top1(D[k]["tops"][0]) for k in keys)
        bare_v = sum(is_bare(top1(base[k]["tops"][0])) for k in keys)
        bare_i = sum(is_bare(top1(D[k]["tops"][0])) for k in keys)
        # per-prompt KL(vllm || intarsia) over the top-20 of vllm, with the rest bucketed
        kls = []
        for k in keys:
            a, b = probs(base[k]["tops"][0]), probs(D[k]["tops"][0])
            ks = list(a)
            pa = [a[i] for i in ks] + [max(1 - sum(a.values()), 1e-9)]
            pb = [b.get(i, 1e-9) for i in ks] + [max(1 - sum(b.get(i, 0) for i in ks), 1e-9)]
            kls.append(sum(x * math.log(x / y) for x, y in zip(pa, pb) if x > 0))
        P(f"  vLLM vs {n}: n={len(keys)}  mean P(bare) vLLM {st.mean(pv):.3f}  intarsia {st.mean(pi):.3f}  mean diff {st.mean(d):+.3f} (se {st.stdev(d)/math.sqrt(len(d)):.3f})")
        P(f"    mean log-odds shift {st.mean(dl):+.3f} (se {st.stdev(dl)/math.sqrt(len(dl)):.3f}); argmax is bare: vLLM {bare_v}, intarsia {bare_i} of {len(keys)}; argmax token differs on {flips}; mean top-20 KL {st.mean(kls):.4f} (median {st.median(kls):.4f}, max {max(kls):.3f})")
        P(f"    prompts where intarsia has the higher P(bare): {sum(x>0 for x in d)}, lower: {sum(x<0 for x in d)}")

P("\n== thinking ON, first 96 greedy tokens: where do the engines diverge, and what was the margin there?")
base = E_["vllm"]
for n, D in E_.items():
    if n == "vllm":
        continue
    keys = sorted(k for k in base if k[1] == "on" and k in D)
    if not keys:
        continue
    cps, margins, ex = [], [], 0
    for k in keys:
        a, b = base[k]["ids"], D[k]["ids"]
        j = 0
        while j < min(len(a), len(b)) and a[j] == b[j]:
            j += 1
        cps.append(j)
        if j < min(len(a), len(b)):
            rv, ri = base[k]["tops"][j], D[k]["tops"][j]
            m = lambda row: sorted(row.values(), reverse=True)
            margins.append((m(rv)[0] - m(rv)[1], m(ri)[0] - m(ri)[1]))
    full = sum(c >= 96 for c in cps)
    P(f"  vLLM vs {n}: identical 96 tokens on {full} of {len(keys)}; first-divergence token index quantiles (10/25/50/75/90%): {[sorted(cps)[int(q*len(cps))] for q in (.1,.25,.5,.75,.9)]}")
    if margins:
        P(f"    top-1 minus top-2 logprob gap at the divergence: vLLM median {st.median(x for x,_ in margins):.3f}, intarsia median {st.median(y for _,y in margins):.3f}; gap < 0.5 nats on both engines in {sum(1 for x,y in margins if x<0.5 and y<0.5)} of {len(margins)}")

# the "a few" / "three" decision: the token after "The user provides"
P("\n== thinking ON: the 'The user provides [a few | three] examples' decision")
ids_a, ids_t = tok.encode(" a").ids[-1], tok.encode(" three").ids[0]
def slot(r):
    text = ""
    for j, t in enumerate(r["ids"]):
        if text.endswith("The user provides"):
            return j
        text += dec(t)
    return None
for n, D in E_.items():
    c = {"a": 0, "three": 0, "other": 0}
    ps = []
    for k in sorted(k for k in D if k[1] == "on"):
        r = D[k]; j = slot(r)
        if j is None:
            c["other"] += 1; continue
        t = dec(r["ids"][j]).strip()
        c["three" if t == "three" else "a" if t == "a" else "other"] += 1
    P(f"  {n}: first-trace choice (own greedy path) {c}")
# teacher-forced: both engines see the SAME prefix only up to the slot, so compare the row at the first divergence when it is the slot, else skip
for n, D in E_.items():
    if n == "vllm":
        continue
    sh = []
    for k in sorted(k for k in D if k[1] == "on" and k in base):
        a, b = base[k], D[k]
        j = 0
        while j < min(len(a["ids"]), len(b["ids"])) and a["ids"][j] == b["ids"][j]:
            j += 1
        if j >= min(len(a["ids"]), len(b["ids"])):
            continue
        ta, tb = a["tops"][j], b["tops"][j]
        dtxt = {dec(a["ids"][j]).strip(), dec(b["ids"][j]).strip()}
        if dtxt == {"a", "three"}:
            ia, it = [i for i in ta if dec(i).strip() == "a"], [i for i in ta if dec(i).strip() == "three"]
            ib, itb = [i for i in tb if dec(i).strip() == "a"], [i for i in tb if dec(i).strip() == "three"]
            if ia and it and ib and itb:
                sh.append((ta[it[0]] - ta[ia[0]], tb[itb[0]] - tb[ib[0]]))
    if sh:
        P(f"  {n}: at the 'a'/'three' divergence slot (same context on both engines), log-odds(three vs a): vLLM mean {st.mean(x for x,_ in sh):+.3f}, intarsia mean {st.mean(y for _,y in sh):+.3f}, n={len(sh)}; paired diff {st.mean(y-x for x,y in sh):+.3f} (se {st.stdev([y-x for x,y in sh])/math.sqrt(len(sh)):.3f})")
# the soak's own traces (full runs) against the probes: is an engine reproducible against itself, probe vs soak?
P("\n== thinking ON: first-96-token text of each probe against the soak run's trace of the same question (same engine = reproducibility)")
soak = f"{raw}/../../2026-11-13-cutover-soak/raw"
def soak_trace(run):
    d = json.load(open(f"{soak}/{run}/result.json"))["benchmarks"][0]
    return {x["id"]: x["reasoning_content"] for x in d["replies"]}
SV, SB = soak_trace("20261113-100540-adhoc-soak-V-think-gsm8k"), soak_trace("20261113-105601-adhoc-soak-Bc-gsm8k")
def cp(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n
for n, D in E_.items():
    for sn, S in (("soak vLLM", SV), ("soak Bc (intarsia bf16, store on)", SB)):
        ks = [k for k in D if k[1] == "on" and k[0] in S]
        if not ks:
            continue
        same = sum(cp(D[k]["text"], S[k[0]]) >= len(D[k]["text"]) - 1 for k in ks)
        P(f"  probe {n} vs {sn}: first 96 tokens identical on {same} of {len(ks)}")
try:
    TB = {(r["id"], r["kind"]): r for r in load("probe-vllm-tf-bc.json")}
    am = sum(is_bare(top1(r["tops"][0])) for r in TB.values())
    pb = st.mean(sum(p for i, p in probs(r["tops"][0]).items() if is_bare(i)) for r in TB.values())
    P(f"\n== thinking ON, first token after Bc's own traces + </think>, vLLM only: argmax is a bare number on {am} of {len(TB)}, mean P(bare) {pb:.3f} (Bc's actual replies: 61 of 85 bare; vLLM on its own traces: 74 of 85, 0.871)")
except (FileNotFoundError, OSError):
    pass
open(f"{raw}/analysis.txt", "w").write("\n".join(out) + "\n")
print("\n".join(out))
