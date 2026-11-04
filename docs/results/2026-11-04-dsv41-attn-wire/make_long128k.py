#!/usr/bin/env python3
"""Build the ~126K-token prompt of the attn-wire 128K comparison: long64k.txt (65K tokens, from the attn-fusion lane), then the markdown
files under docs/ in sorted path order (skipping this lane's own and long ones that repeat), until the DSv4.1 tokenizer counts
TARGET tokens; cut at a line break. No question at the end: the first window's prompt ended in one and both binaries answered with end-of-sequence at once (1 token, empty text), a vacuous comparison. Usage: make_long128k.py TOKENIZER OUT [TARGET]  (run from the repo root)"""
import sys, glob, hashlib
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1]); out = sys.argv[2]; target = int(sys.argv[3]) if len(sys.argv) > 3 else 126000
n = lambda s: len(tok.encode(s, add_special_tokens=False).ids)
text = open("docs/results/2026-11-03-dsv41-attn-fusion/prompts/long64k.txt").read().rstrip("\n") + "\n\n"
have = n(text)
for f in sorted(glob.glob("docs/**/*.md", recursive=True)):
    if "attn-wire" in f: continue
    s = open(f).read()
    k = n(s)
    if have + k > target:
        lines = s.splitlines(keepends=True); acc = ""
        for l in lines:
            if have + n(acc + l) > target: break
            acc += l
        text += acc; break
    text += s + "\n\n"; have += k
open(out, "w").write(text)
print(f"{out}: {len(text)} bytes, {n(text)} tokens, sha256 {hashlib.sha256(text.encode()).hexdigest()[:16]}")
