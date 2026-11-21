#!/usr/bin/env python3
"""samp2s: compare seeded2.py replies byte for byte, keyed by (cell, size, seed); first file is the reference.
  python3 compare_seeded2.py a1-seeded2.json a2-seeded2.json ..."""
import json, sys
load = lambda f: {(r["cell"], r["size"], r["seed"]): r for r in json.load(open(f))}
ref = load(sys.argv[1]); ok = True
print(f"reference {sys.argv[1]}: " + ", ".join(f"{k[0]}/{k[1]}/s{k[2]}={v['completion_tokens']}t" for k, v in ref.items()))
for f in sys.argv[2:]:
    cur = load(f); same_n = 0
    for k, a in ref.items():
        b = cur.get(k)
        same = bool(b) and not a["error"] and not b["error"] and a["text"] == b["text"] and a["completion_tokens"] == b["completion_tokens"] and a["prompt_tokens"] == b["prompt_tokens"]
        ok &= same; same_n += same
        print(f"{f}: {k[0]} size {k[1]} seed {k[2]} prompt {a['prompt_tokens']}/{b and b['prompt_tokens']} out {a['completion_tokens']}/{b and b['completion_tokens']} identical={same}")
    print(f"{f}: {same_n} of {len(ref)} identical")
print("ALL IDENTICAL" if ok else "DIFFERENCES"); sys.exit(0 if ok else 1)
