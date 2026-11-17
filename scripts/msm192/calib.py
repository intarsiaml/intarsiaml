#!/usr/bin/env python3
"""Lane msm192: token counts of the load.py slices (chars at offset + 997*i) with the 27B tokenizer: pick CHARS giving 140 / 170 / 190 / 100.
usage: uv run --with tokenizers calib.py <tokenizer.json> <prompt-16384.txt> <offset> <requests> chars..."""
import sys
from tokenizers import Tokenizer
tk = Tokenizer.from_file(sys.argv[1]); text = open(sys.argv[2]).read(); off0 = int(sys.argv[3]); nreq = int(sys.argv[4])
for ch in map(int, sys.argv[5:]):
    n = [len(tk.encode(text[(off0 + 997 * i) % max(1, len(text) - ch):][:ch]).ids) for i in range(nreq)]
    print(ch, n, sum(n) / len(n))
