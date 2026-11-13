#!/usr/bin/env python3
"""ab35b run 2: accuracy and exact-reply equality across localeval gsm8k result.json files.
usage: compare_replies.py label=path/to/result.json ... [--pairs a:b,a:c] [--json OUT]
Equality is on the reply text, question by question (same id). Ids are the benchmark's own, so the runs must be the
same question set (no nonce, no shuffling)."""
import json, sys, itertools

def load(p):
    d = json.load(open(p))
    b = d["benchmarks"][0]
    return {r["id"]: r for r in b["replies"]}, b

def main():
    argv = sys.argv[1:]
    pairs = out = None
    runs = {}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--pairs": pairs = [p.split(":") for p in argv[i + 1].split(",")]; i += 2; continue
        if a == "--json": out = argv[i + 1]; i += 2; continue
        k, p = a.split("=", 1); runs[k] = load(p); i += 1
    res = {"accuracy": {}, "pairs": {}}
    for k, (rep, b) in runs.items():
        n = len(rep); s = sum(1 for r in rep.values() if r["flexible"])
        bare = [r for r in rep.values() if r.get("bare_number")]
        res["accuracy"][k] = {"n": n, "correct": s, "acc": round(s / n, 4), "overall_score": b.get("overall_score"),
                              "bare": len(bare), "bare_correct": sum(1 for r in bare if r["flexible"]),
                              "worked": n - len(bare), "worked_correct": s - sum(1 for r in bare if r["flexible"]),
                              "mean_completion_tokens": round(sum(r["completion_tokens"] for r in rep.values()) / n, 1),
                              "truncated": sum(1 for r in rep.values() if r.get("truncated"))}
    pr = pairs or [list(p) for p in itertools.combinations(runs, 2)]
    for a, b in pr:
        ra, rb = runs[a][0], runs[b][0]
        ids = sorted(set(ra) & set(rb))
        same = [i for i in ids if ra[i]["reply"] == rb[i]["reply"]]
        diff = [i for i in ids if ra[i]["reply"] != rb[i]["reply"]]
        sc = sum(1 for i in ids if ra[i]["flexible"] == rb[i]["flexible"])
        def first_diff(i):
            x, y = ra[i]["reply"], rb[i]["reply"]
            n = next((j for j in range(min(len(x), len(y))) if x[j] != y[j]), min(len(x), len(y)))
            return {"id": i, "first_diff_char": n, "len_a": len(x), "len_b": len(y), "correct_a": ra[i]["flexible"], "correct_b": rb[i]["flexible"]}
        res["pairs"][f"{a} vs {b}"] = {"common": len(ids), "identical": len(same), "different": len(diff), "same_correctness": sc,
                                        "diffs": [first_diff(i) for i in diff]}
    print(json.dumps(res, indent=1))
    if out: json.dump(res, open(out, "w"), indent=1)

main()
