#!/usr/bin/env python3
"""storettft gate (b): compare ids_check.py outputs. usage: compare_ids.py BASE-ids.json OTHER-ids.json [...]
Per file against the first: how many (size, kind) replies are identical in text and token count, and which are not."""
import json
import sys


def load(p):
    return {(r["size"], r["kind"]): r for r in json.load(open(p))}


base = load(sys.argv[1])
print(f"base {sys.argv[1]}: {len(base)} replies, {sum(1 for r in base.values() if r['ok'])} ok; cached tokens on the 'again' sends: "
      + ", ".join(f"{k[0]}:{r['cached_tokens']}" for k, r in sorted(base.items()) if k[1] == "again"))
for p in sys.argv[2:]:
    o = load(p)
    bad = [k for k in sorted(base) if not (k in o and base[k]["ok"] and o[k]["ok"] and base[k]["text"] == o[k]["text"] and base[k]["completion_tokens"] == o[k]["completion_tokens"])]
    print(f"{p}: identical {len(base) - len(bad)} of {len(base)}" + (f"; different: {bad}" if bad else "") + "; cached tokens on the 'again' sends: "
          + ", ".join(f"{k[0]}:{r['cached_tokens']}" for k, r in sorted(o.items()) if k[1] == "again"))
