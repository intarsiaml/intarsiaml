#!/usr/bin/env python3
"""Rebuild the exact length-probe prompt text exp3.py used (same search, token counts from tokenizer.json) and save text and ids."""
import json, sys
from tokenizers import Tokenizer
tok = Tokenizer.from_file("/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4/tokenizer.json")
text = open("docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-16384.txt").read()
out = {}
for tgt in [int(x) for x in sys.argv[1].split(",")]:
    head = f"Length probe {tgt}.\n"; n = int(tgt * 3.9)
    for _ in range(40):
        pl = len(tok.encode(head + text[:n], add_special_tokens=False).ids)
        if pl == tgt: break
        n += (tgt - pl) if abs(tgt - pl) < 4 else int((tgt - pl) * 3.5)
    out[str(tgt)] = {"n": n, "plen": pl, "ok": pl == tgt, "ids": tok.encode(head + text[:n], add_special_tokens=False).ids}
    print(tgt, n, pl, flush=True)
json.dump(out, open(".scratch/logs/oracle/prompts.json", "w"))
