#!/usr/bin/env python3
"""kvonce: compare the seeded replies of several windows byte for byte (first file is the reference): python3 compare_seeded.py w8-seeded.json w9-seeded.json ..."""
import json, sys
ref = json.load(open(sys.argv[1])); ok = True
for f in sys.argv[2:]:
    cur = json.load(open(f))
    for a, b in zip(ref, cur):
        same = a["text"] == b["text"] and a["completion_tokens"] == b["completion_tokens"] and a["prompt_tokens"] == b["prompt_tokens"]
        ok &= same
        print(f"{f}: size {a['size']} seed {a['seed']} prompt {a['prompt_tokens']}/{b['prompt_tokens']} out {a['completion_tokens']}/{b['completion_tokens']} identical={same}")
    ok &= len(ref) == len(cur)
print("ALL IDENTICAL" if ok else "DIFFERENCES"); sys.exit(0 if ok else 1)
