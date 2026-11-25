"""Output-check prompts: dir short (4 x 4300 chars, ~1.2k tokens, as 48b) and dir long (4 x 8600 chars, ~2.5k tokens: two 2048 chunks).
usage: mkprompts2.py SRC OUTDIR"""
import os
import sys

t = open(sys.argv[1]).read()
n = len(t) // 4
os.makedirs(f"{sys.argv[2]}/short", exist_ok=True)
os.makedirs(f"{sys.argv[2]}/long", exist_ok=True)
for i in range(4):
    open(f"{sys.argv[2]}/short/p{i}.txt", "w").write(t[i * n:i * n + 4300])
    open(f"{sys.argv[2]}/long/p{i}.txt", "w").write(t[i * 1900:i * 1900 + 8600])
