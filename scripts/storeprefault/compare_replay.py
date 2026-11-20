#!/usr/bin/env python3
"""storeprefault gate (b): compare the reply texts of replay2.py runs (same plan, same trace, greedy) per (conv, turn).
usage: compare_replay.py A.jsonl B.jsonl [C.jsonl ...]   -> prints, for every file against the first, how many requests have an identical reply,
the first differing (conv, turn) and the first differing character. Rows that failed (ok false) count as different."""
import json
import sys


def load(p):
    return {(r["conv"], r["turn"]): r for r in map(json.loads, open(p)) if r.get("seed") is not None}


base_path = sys.argv[1]
base = load(base_path)
print(f"base {base_path}: {len(base)} requests, {sum(1 for r in base.values() if r['ok'])} ok")
for p in sys.argv[2:]:
    other = load(p)
    same = diff = 0
    first = None
    for k in sorted(base):
        a, b = base[k], other.get(k)
        if b is not None and a["ok"] and b["ok"] and a.get("reply") == b.get("reply") and a.get("completion_tokens") == b.get("completion_tokens"):
            same += 1
        else:
            diff += 1
            if first is None:
                ra, rb = (a.get("reply") or ""), ((b or {}).get("reply") or "")
                at = next((i for i, (x, y) in enumerate(zip(ra, rb)) if x != y), min(len(ra), len(rb)))
                first = (k, at, ra[max(0, at - 20):at + 20], rb[max(0, at - 20):at + 20])
    print(f"{p}: identical {same} of {len(base)}, different {diff}" + (f"; first difference conv/turn {first[0]} at char {first[1]}: {first[2]!r} vs {first[3]!r}" if first else ""))
