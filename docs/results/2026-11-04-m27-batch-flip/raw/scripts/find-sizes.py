#!/usr/bin/env python3
"""Offline: which localeval --sizes values land on an exposed prompt length. Same filler as localeval (seed = size), offset fitted on the scout's reported prompt lengths."""
import sys, statistics as st
sys.path.insert(0, "/opt/code/deepeval/src")
from localeval.speed import filler, INSTRUCTION
from tokenizers import Tokenizer
tok = Tokenizer.from_file("/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4/tokenizer.json")
P, B = 22.1, 53
ln = lambda s: len(tok.encode(filler(max(0, round((s - B) / P)), seed=s) + INSTRUCTION, add_special_tokens=False).ids)
obs = {4100: 4079, 4110: 4098, 4120: 4093, 4130: 4115, 16390: 16656, 16400: 16659, 16410: 16660, 16420: 16686}
off = {s: p - ln(s) for s, p in obs.items()}; print("offsets", off)
o = st.median(off.values())
for lo, hi in ((3800, 4400), (16200, 16900)):
    c = []
    for s in range(lo, hi):
        p = ln(s) + o
        if 3 <= p % 256 <= 8 and p % 2048 >= 256: c.append((s, int(p), int(p) % 256))
    print(lo, hi, len(c), c[:16])
