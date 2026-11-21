#!/usr/bin/env python3
"""drill: compare two verify.py outputs: byte-identical replies (raw streamed text) per request. usage: cmp.py BASE.json OTHER.json"""
import json, sys
a = json.load(open(sys.argv[1]))["rows"]; b = json.load(open(sys.argv[2]))["rows"]
n = min(len(a), len(b)); same = sum(1 for x, y in zip(a, b) if x["ok"] and y["ok"] and x["raw"] == y["raw"])
diff = [x["i"] for x, y in zip(a, b) if not (x["ok"] and y["ok"] and x["raw"] == y["raw"])]
print(f"identical {same}/{n} (base ok {sum(x['ok'] for x in a)}, other ok {sum(y['ok'] for y in b)}); different request indexes: {diff}")
for i in diff[:3]:
    x, y = a[i], b[i]; k = next((j for j in range(min(len(x['raw']), len(y['raw']))) if x['raw'][j] != y['raw'][j]), None)
    print(f"  req {i}: first differing char {k}; base {x['raw'][:80]!r} other {y['raw'][:80]!r}")
ct = [(x["cached_tokens"], y["cached_tokens"]) for x, y in zip(a, b)]
print(f"cached_tokens per request identical: {all(p == q for p, q in ct)} (first request {ct[0]}; sum base {sum(p for p,_ in ct)} other {sum(q for _,q in ct)})")
sys.exit(0 if same == n and len(a) == len(b) else 1)
