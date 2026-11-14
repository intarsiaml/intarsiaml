#!/usr/bin/env python3
"""soak lane: paired GSM8K verdicts on the same question ids (localeval result.json files).
usage: paired_gsm8k.py [--first N] label=path/to/result.json ... --pairs a:b,c:d [--json OUT]
Per pair, on the ids both runs answered (optionally only the first N ids, sorted): accuracy of each, discordant counts
(a right b wrong, a wrong b right), exact two-sided binomial (McNemar) p on the discordant pairs, identical reply text count,
plus truncation, mean completion tokens and `thinking_seen` / `reasoning_content` counts per run."""
import json, math, sys

def load(p):
    b = json.load(open(p))["benchmarks"][0]
    return {r["id"]: r for r in b["replies"]}

def ok(r):
    v = r["flexible"]
    return v if isinstance(v, bool) else str(v).lower() == "true"

def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)

def main():
    argv = sys.argv[1:]; runs = {}; pairs = []; first = None; out = None; i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--pairs": pairs = [p.split(":") for p in argv[i + 1].split(",")]; i += 2; continue
        if a == "--first": first = int(argv[i + 1]); i += 2; continue
        if a == "--json": out = argv[i + 1]; i += 2; continue
        k, p = a.split("=", 1); runs[k] = load(p); i += 1
    res = {"runs": {}, "pairs": {}}
    for k, rep in runs.items():
        ids = sorted(rep)[:first] if first else sorted(rep)
        rs = [rep[x] for x in ids]
        res["runs"][k] = {"n": len(rs), "correct": sum(ok(r) for r in rs), "acc": round(sum(ok(r) for r in rs) / len(rs), 4),
                          "truncated": sum(1 for r in rs if str(r.get("truncated")).lower() == "true"),
                          "mean_completion_tokens": round(sum(int(r["completion_tokens"]) for r in rs) / len(rs), 1),
                          "with_reasoning_content": sum(1 for r in rs if (r.get("reasoning_content") or "").strip()),
                          "thinking_seen": sum(1 for r in rs if str(r.get("thinking_seen")).lower() == "true")}
    for a, b in pairs:
        ids = sorted(set(runs[a]) & set(runs[b]))
        if first: ids = ids[:first]
        ra, rb = runs[a], runs[b]
        ab = sum(1 for x in ids if ok(ra[x]) and not ok(rb[x])); ba = sum(1 for x in ids if not ok(ra[x]) and ok(rb[x]))
        res["pairs"][f"{a} vs {b}"] = {"common": len(ids), "acc_a": round(sum(ok(ra[x]) for x in ids) / len(ids), 4),
                                       "acc_b": round(sum(ok(rb[x]) for x in ids) / len(ids), 4), "a_right_b_wrong": ab,
                                       "a_wrong_b_right": ba, "mcnemar_p": round(mcnemar(ab, ba), 4),
                                       "identical_reply_text": sum(1 for x in ids if ra[x]["reply"] == rb[x]["reply"])}
    print(json.dumps(res, indent=1))
    if out: json.dump(res, open(out, "w"), indent=1)

main()
