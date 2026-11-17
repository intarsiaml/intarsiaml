#!/usr/bin/env python3
"""Lane msm192: prompts of exactly N tokens (27B tokenizer) cut from slices of the long text, 8 per N, distinct offsets (no prefix-cache hit).
usage: uv run --with tokenizers gen_prompts.py <tokenizer.json> <prompt-16384.txt> <out.json> N..."""
import json, sys
from tokenizers import Tokenizer
tk = Tokenizer.from_file(sys.argv[1]); text = open(sys.argv[2]).read(); out = {}
for n in map(int, sys.argv[4:]):
    ps = []
    for i in range(8 if n < 1000 else 1):
        off = (3001 + 4093 * i + 311 * n) % (len(text) - n * 6)
        sl = text[off: off + n * 6]
        e = tk.encode(sl)
        p = sl[: e.offsets[n - 1][1]]
        for _ in range(8):
            c = len(tk.encode(p).ids)
            if c == n: break
            p = sl[: e.offsets[n - 1 - (c - n)][1]] if c > n else sl[: e.offsets[min(len(e.ids) - 1, n - 1 + (n - c))][1]]
        assert len(tk.encode(p).ids) == n, (n, i, len(tk.encode(p).ids))
        ps.append(p)
    out[str(n)] = ps
json.dump(out, open(sys.argv[3], "w"))
print({k: [len(tk.encode(p).ids) for p in v] for k, v in out.items()})
