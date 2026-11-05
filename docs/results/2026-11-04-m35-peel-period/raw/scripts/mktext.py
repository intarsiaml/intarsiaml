#!/usr/bin/env python3
"""Prompt texts keyed by exact token count for a model's tokenizer. usage: mktext.py MODEL_DIR OUT.json LEN,LEN,..."""
import json, sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1] + "/tokenizer.json")
text = open("docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-16384.txt").read()
out = {}
for tgt in [int(x) for x in sys.argv[3].split(",")]:
    head = f"Length probe {tgt}.\n"; n = int(tgt * 3.9)
    for _ in range(40):
        pl = len(tok.encode(head + text[:n], add_special_tokens=False).ids)
        if pl == tgt: break
        n += (tgt - pl) if abs(tgt - pl) < 4 else int((tgt - pl) * 3.5)
    assert pl == tgt, tgt
    out[str(tgt)] = head + text[:n]
json.dump(out, open(sys.argv[2], "w")); print(len(out), "prompts")
