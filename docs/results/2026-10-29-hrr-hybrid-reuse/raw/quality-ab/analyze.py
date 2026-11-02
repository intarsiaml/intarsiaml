#!/usr/bin/env python3
import json, sys, math
def cuts(P, chunk=4096):
    out = []; pos = 0
    while pos < P:
        rem = P - pos
        if rem <= chunk:
            if rem > 1: out.append(P - 1)
            break
        pos += chunk; out.append(pos)
    return out
def reqstats(rows, name):
    n = len(rows); res = [r for r in rows if r["cached"]]
    cold = [r for r in res if r["cached"] in cuts(r["prompt_tokens"])]
    mean_lost = sum(r["prompt_tokens"] - (r["cached"] or 0) for r in rows) / n
    print(f"  {name}: {n} requests, {len(res)} resumed (cached>0), {len(cold)} of those at a cold cut of their own prompt, mean uncached tokens {mean_lost:.0f}")
def cmp(a, b, label):
    A = {r["id"]: r for r in a}; B = {r["id"]: r for r in b}; ids = [i for i in A if i in B]; n = len(ids)
    ca = sum(A[i]["ok"] for i in ids); cb = sum(B[i]["ok"] for i in ids)
    diff = [i for i in ids if A[i]["pred"] != B[i]["pred"]]
    ra = sum(A[i]["ok"] for i in diff); rb = sum(B[i]["ok"] for i in diff)
    bb = sum(1 for i in ids if A[i]["ok"] and not B[i]["ok"]); cc = sum(1 for i in ids if B[i]["ok"] and not A[i]["ok"])
    se = math.sqrt(max(bb + cc - (bb - cc) ** 2 / n, 0)) / n
    sa = math.sqrt(ca / n * (1 - ca / n) / n); sb = math.sqrt(cb / n * (1 - cb / n) / n)
    print(f"{label}: n={n}  A(default) {ca/n:.4f} (SE {sa:.4f})  B(exact) {cb/n:.4f} (SE {sb:.4f})  A-B {(ca-cb)/n:+.4f} paired SE {se:.4f}")
    print(f"  answers differing {len(diff)}: A right {ra}, B right {rb}, neither {len(diff)-ra-rb}")
    reqstats(a, "arm A"); reqstats(b, "arm B")
if __name__ == "__main__":
    for k in sys.argv[1:]:
        cmp(json.load(open(f".scratch/qa/mt-A-{k}.json")), json.load(open(f".scratch/qa/mt-B-{k}.json")), f"multi-turn {k}")
