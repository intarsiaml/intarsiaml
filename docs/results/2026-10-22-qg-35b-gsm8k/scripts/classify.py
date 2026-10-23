"""Classify GSM8K replies into the lane-qg buckets (run: cd /opt/code/deepeval && uv run python <this> a.jsonl [b.jsonl]).

Rows need: i, expected, text, finish, ct, score (deepeval exact match). Buckets:
  ok-strict          exact match passed
  fmt-extraction     reply is wrong under exact match but its extracted last number is the gold answer
                     (worked solution + number; or direct number with a comma/format quirk)
  wrong-answer       a final number was produced and it is not the gold answer (split: direct vs cot)
  truncated          finish_reason == length (max_tokens hit)
  degenerate         repeated n-gram loop, or empty reply
"""
import json, re, sys, random
from collections import Counter
from localeval.gsm8k import flexible_match

def cls(r):
    t = r["text"] or ""
    if r.get("err"): return "error"
    if r["finish"] == "length": return "truncated"
    if not t.strip() or re.search(r"(.{12,}?)\1{4,}", t, re.S): return "degenerate"
    if r["score"] == 1: return "ok-strict"
    if flexible_match(r["expected"], t): return "fmt-extraction"
    return "wrong-answer (cot)" if r["ct"] > 20 else "wrong-answer (direct)"

if __name__ == "__main__":
    for path in sys.argv[1:]:
        rows = [json.loads(l) for l in open(path) if l.strip()]
        c = Counter(cls(r) for r in rows)
        n = len(rows)
        print(f"## {path}  n={n}  strict={sum(r['score'] for r in rows)/n:.4f}  flexible={sum(flexible_match(r['expected'], r['text']) for r in rows)/n:.4f}")
        for k, v in sorted(c.items(), key=lambda kv: -kv[1]): print(f"  {k:24s} {v:5d} {v/n:6.1%}")
        long_ = sum(1 for r in rows if r["ct"] > 20)
        print(f"  replies with >20 completion tokens (worked solution): {long_}  | with <=20 (bare number): {n-long_}")
        direct = [r for r in rows if r["ct"] <= 20]; cot = [r for r in rows if r["ct"] > 20]
        print(f"  flexible acc on bare-number replies {sum(flexible_match(r['expected'], r['text']) for r in direct)}/{len(direct)}; on worked-solution replies {sum(flexible_match(r['expected'], r['text']) for r in cot)}/{len(cot)}")
        random.seed(20261022)
        fails = [r for r in rows if r["score"] == 0]
        print("  | q | gold | ct | finish | bucket | reply tail |\n  |---:|---:|---:|---|---|---|")
        for r in sorted(random.sample(fails, 20), key=lambda r: r["i"]):
            tail = r["text"][-50:].replace("\n", " ").replace("|", "/")
            print(f"  | {r['i']} | {r['expected']} | {r['ct']} | {r['finish']} | {cls(r)} | `{tail}` |")
