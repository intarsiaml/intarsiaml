#!/usr/bin/env python3
"""Compare two run-full.sh `longp` outputs: same text, same tokens, same top-5 logprobs bit for bit.
Usage: cmp_longp.py A_DIR B_DIR   (each holds longp-*.json)"""
import json, sys, glob, os
ok = True
for fa in sorted(glob.glob(os.path.join(sys.argv[1], "longp-*.json"))):
    fb = os.path.join(sys.argv[2], os.path.basename(fa))
    if not os.path.exists(fb):
        print("missing", fb); ok = False; continue
    a, b = (json.load(open(f))["choices"][0] for f in (fa, fb))
    la, lb = a["logprobs"], b["logprobs"]
    same_text = a["text"] == b["text"]
    same_tok = la["tokens"] == lb["tokens"]
    same_lp = la["token_logprobs"] == lb["token_logprobs"] and la["top_logprobs"] == lb["top_logprobs"]
    n = sum(1 for x, y in zip(la["token_logprobs"], lb["token_logprobs"]) if x == y)
    worst = max((abs(x - y) for x, y in zip(la["token_logprobs"], lb["token_logprobs"])), default=0.0)
    print(f"{os.path.basename(fa)}: text {'same' if same_text else 'DIFF'}, tokens {'same' if same_tok else 'DIFF'}, logprobs {'bit-identical' if same_lp else 'DIFF'} "
          f"({n}/{len(la['token_logprobs'])} equal, max abs diff {worst:.3g}); first token {la['tokens'][0]!r} {la['token_logprobs'][0]:.6f} vs {lb['tokens'][0]!r} {lb['token_logprobs'][0]:.6f}")
    ok &= same_text and same_tok and same_lp
sys.exit(0 if ok else 1)
