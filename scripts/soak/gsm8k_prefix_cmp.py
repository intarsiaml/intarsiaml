#!/usr/bin/env python3
"""restore-acc: compare two gsm8k_prefix.py arms. usage: gsm8k_prefix_cmp.py A.jsonl B.jsonl [--label-a W --label-b C]
Reports identical replies / token-id sequences, every non-identical reply with its question id and first diverging token index
(and the two tokens), and the paired accuracy (exact two-sided McNemar on the discordant questions)."""
import argparse
import json
from math import comb


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--json")
    x = ap.parse_args()
    A = {r["id"]: r for r in map(json.loads, open(x.a))}
    B = {r["id"]: r for r in map(json.loads, open(x.b))}
    ids = sorted(set(A) & set(B))
    same_text = same_ids = 0
    diffs = []
    for i in ids:
        a, b = A[i], B[i]
        ta, tb = a.get("token_ids"), b.get("token_ids")
        st = (a.get("reply") == b.get("reply"))
        si = (ta is not None and ta == tb)
        same_text += st
        same_ids += si
        if not (st and si):
            k = None
            if ta is not None and tb is not None:
                k = next((j for j, (p, q) in enumerate(zip(ta, tb)) if p != q), min(len(ta), len(tb)))
            diffs.append({"id": i, "first_diverging_token_index": k, "tok_a": ta[k] if k is not None and k < len(ta) else None,
                          "tok_b": tb[k] if k is not None and k < len(tb) else None, "len_a": len(ta or []), "len_b": len(tb or []),
                          "a_ok": a["ok"], "b_ok": b["ok"]})
    ca = {i: A[i]["correct"] for i in ids}
    cb = {i: B[i]["correct"] for i in ids}
    only_a = sum(ca[i] and not cb[i] for i in ids)
    only_b = sum(cb[i] and not ca[i] for i in ids)
    res = {"n": len(ids), "a": x.label_a, "b": x.label_b, "identical_text": same_text, "identical_token_ids": same_ids,
           "acc_a": sum(ca.values()) / len(ids), "acc_b": sum(cb.values()) / len(ids), f"right_only_{x.label_a}": only_a,
           f"right_only_{x.label_b}": only_b, "mcnemar_p": round(mcnemar(only_a, only_b), 4), "non_identical": diffs}
    print(json.dumps({k: v for k, v in res.items() if k != "non_identical"}))
    for d in diffs:
        print("  non-identical:", json.dumps(d))
    if x.json:
        json.dump(res, open(x.json, "w"), indent=1)


if __name__ == "__main__":
    main()
