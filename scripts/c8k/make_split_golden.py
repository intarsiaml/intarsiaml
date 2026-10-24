#!/usr/bin/env python3
"""Lane c8k: derive a reuse-only golden from the 35B long golden whose 12k prompts carry an explicit `reuse_split`.

The prefix-reuse gate (`accuracy --mode reuse`) reads only the prompts (names, ids, optional per-prompt `reuse_split`) from a golden. At
prefill chunk 8192 the long golden's 12k prompts get the default split n-2 (their `reuse_split` is absent, so n_prompt/2 = 6000 is not
above the chunk and the gate falls back to 1.5 x chunk, bounded by n-2), which leaves turn 2 two tokens of new text. A split inside the
prompt above 8192 gives turn 2 ~2.5k tokens of new text after the restored prefix. Only the metadata changes; the arrays and the prompt
ids are the long golden's, byte for byte.

usage: make_split_golden.py LONG_GOLDEN.safetensors OUT.safetensors SPLIT [PROMPT ...]   (default prompts: long_docs_12k long_code_12k)
"""
import json, struct, sys

src, out, split = sys.argv[1], sys.argv[2], int(sys.argv[3])
names = sys.argv[4:] or ["long_docs_12k", "long_code_12k"]
with open(src, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    data = f.read()
meta = hdr["__metadata__"]
prompts = json.loads(meta["prompts"])
hit = 0
for p in prompts:
    if p["name"] in names:
        assert p["n_tokens"] > split + 2, p["name"]
        p["reuse_split"] = split
        hit += 1
assert hit == len(names), (hit, names)
meta["prompts"] = json.dumps(prompts)
meta["note"] = (meta.get("note", "") + f" | c8k: reuse-only derivative, reuse_split={split} for {','.join(names)}; arrays and ids unchanged").strip(" |")
h = json.dumps(hdr, separators=(",", ":")).encode()
h += b" " * ((8 - len(h) % 8) % 8)
with open(out, "wb") as f:
    f.write(struct.pack("<Q", len(h)))
    f.write(h)
    f.write(data)
print(f"wrote {out}: {hit} prompts get reuse_split={split}")
