#!/usr/bin/env python3
"""Compare `plan show --json` of two trees: plan_diff.py <dir> [a=fusion] [b=base] reads <dir>/<a>-<m>.json and
<dir>/<b>-<m>.json for m in 35b, 8b (and cuda-35b / cuda-8b when present) and prints, per model: launches per decode step
(sum of calls_per_decode_step), scratch bytes, the (op, kernel) rows only one side has, and whether the rest is identical."""
import json
import os
import sys

d = sys.argv[1]
a, b = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ("fusion", "base")


def load(path):
    try:
        s = open(path).read()
        return json.loads(s[s.index("{"):])
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def rows(p):
    return {(o["op"], o["kernel"]): o.get("calls_per_decode_step", 0) for o in p.get("ops", [])}


for m in ("35b", "8b"):
    pa, pb = load(os.path.join(d, f"{a}-{m}.json")), load(os.path.join(d, f"{b}-{m}.json"))
    if "error" in pa or "error" in pb:
        print(f"{m}: unreadable ({pa.get('error')} / {pb.get('error')})")
        continue
    ra, rb = rows(pa), rows(pb)
    la, lb = sum(ra.values()), sum(rb.values())
    print(f"== {m}: launches per decode step {b} {lb} -> {a} {la} ({la - lb:+d}); scratch {pb.get('scratch_bytes')} -> {pa.get('scratch_bytes')}")
    for k in sorted(set(ra) | set(rb)):
        if ra.get(k) != rb.get(k):
            print(f"   {k[0]:28s} {k[1]:40s} {b} {rb.get(k, '-'):>4}  {a} {ra.get(k, '-'):>4}")
    same = {k: v for k, v in ra.items() if rb.get(k) == v}
    print(f"   {len(same)} (op, kernel) rows identical, {len(set(ra) ^ set(rb))} present on one side only")
    for key in ("exactness", "act_dtype", "kv_dtype", "prefill_chunk"):
        if pa.get(key) != pb.get(key):
            print(f"   {key}: {pb.get(key)} -> {pa.get(key)}")
