#!/usr/bin/env python3
"""kvflip: compare the ids files of several windows token for token (first file is the reference): python3 ids_compare.py a1-ids.json b1-ids.json ..."""
import json, sys
ref = json.load(open(sys.argv[1])); ok = True
for f in sys.argv[2:]:
    cur = json.load(open(f))
    ok &= len(ref) == len(cur)
    for a, b in zip(ref, cur):
        same = a["token_ids"] == b["token_ids"] and a["text"] == b["text"] and a["prompt_tokens"] == b["prompt_tokens"] and bool(a["token_ids"])
        ok &= same
        d = next((i for i, (x, y) in enumerate(zip(a["token_ids"] or [], b["token_ids"] or [])) if x != y), None)
        print(f"{sys.argv[1]} vs {f}: size {a['size']} {a['mode']} prompt {a['prompt_tokens']}/{b['prompt_tokens']} out {a['completion_tokens']}/{b['completion_tokens']} identical={same}" + (f" first divergence at {d}" if d is not None else ""))
print("ALL IDENTICAL" if ok else "DIFFERENCES"); sys.exit(0 if ok else 1)
