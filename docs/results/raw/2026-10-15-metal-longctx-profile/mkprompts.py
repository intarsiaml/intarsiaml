#!/usr/bin/env python3
"""Synthetic prompts (same sentence pattern as docs/results/raw/2026-10-14-metal-phase-b/b2a/prompt-16k.txt) with an exact
Qwen3 token count. usage: mkprompts.py <outdir> <tokens>...   (tokenizers from the mlx-lm uv tool env)"""
import sys, random
from tokenizers import Tokenizer
tok = Tokenizer.from_file("/Users/josef/models/Qwen3-8B/tokenizer.json")
subj = ["A patient gardener", "The old lighthouse keeper", "The harbor master", "A curious fox"]
verb = ["measured", "counted", "watched", "repaired", "remembered"]
obj = ["a row of copper kettles", "a stack of faded maps", "the slow tide at dawn", "the grey waves rolling in", "every window along the hill"]
rng = random.Random(7)
def sent(i): return f"{rng.choice(subj)} {rng.choice(verb)} {rng.choice(obj)} on day {i}."
out = sys.argv[1]
text, i = [], 0
targets = sorted(int(x) for x in sys.argv[2:])
def n(t): return len(tok.encode(" ".join(t), add_special_tokens=False).ids)
for T in targets:
    while n(text) < T + 200:
        text.append(sent(i)); i += 1
    lo, hi = 1, len(text)
    while lo < hi:  # fewest sentences with >= T tokens
        mid = (lo + hi) // 2
        if n(text[:mid]) >= T: hi = mid
        else: lo = mid + 1
    s = " ".join(text[:lo])
    ids = tok.encode(s, add_special_tokens=False).ids
    # trim the tail words until exactly T tokens
    words = s.split(" ")
    while len(ids) > T:
        words.pop(); s = " ".join(words); ids = tok.encode(s, add_special_tokens=False).ids
    open(f"{out}/prompt-{T}.txt", "w").write(s)
    print(T, len(ids), len(s))
