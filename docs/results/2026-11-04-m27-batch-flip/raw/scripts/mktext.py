#!/usr/bin/env python3
"""Prompt texts keyed by token count: the q prompts of ident-rt.sh by their known length, the length probes by the exp3.py search."""
import json
from tokenizers import Tokenizer
tok = Tokenizer.from_file("/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4/tokenizer.json")
text = open("docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-16384.txt").read()
out = {}
lens = [150, 400, 900, 1800, 3000, 5000, 8000, 11000, 15700, 22000]
want = [48, 108, 233, 465, 772, 1272, 2055, 2861, 4099, 5770]
for k, (n, w) in enumerate(zip(lens, want)):
    p = f"Request {k}, {n} characters of context follows.\n" + text[:n]
    got = len(tok.encode(p, add_special_tokens=False).ids); assert got == w, (k, got, w); out[str(w)] = p
for tgt in [258, 259, 260, 262, 265, 266, 273, 320, 516, 1028, 1088, 1412, 1540]:
    head = f"Length probe {tgt}.\n"; n = int(tgt * 3.9)
    for _ in range(40):
        pl = len(tok.encode(head + text[:n], add_special_tokens=False).ids)
        if pl == tgt: break
        n += (tgt - pl) if abs(tgt - pl) < 4 else int((tgt - pl) * 3.5)
    assert pl == tgt, tgt; out[str(tgt)] = head + text[:n]
json.dump(out, open(".scratch/logs/gates/prompts-text.json", "w")); print(len(out), "prompts")
