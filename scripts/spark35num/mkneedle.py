#!/usr/bin/env python3
"""Lane spark35: needle-in-a-haystack rows for `localeval quality --dataset ... --judge none` (Contains expected), in the format of
/opt/code/deepeval/datasets/m27-hs3-longctx.jsonl (a vault register; ask for one record's code) but sized in real tokens: the m27
rows are 34153 characters (~12k tokens), below the 32k the fleet's contexts need. Deterministic.
  uv run --with tokenizers scripts/spark35num/mkneedle.py <tokenizer.json> <out_dir>
Writes needle-32k.jsonl (12 rows, 32.8k+ prompt tokens) and needle-64k.jsonl (8 rows, 65.5k+ tokens); the asked record sits at depth
5 % .. 95 % (evenly spread); the question comes after the register, as in the m27 set."""
import json, pathlib, random, sys
from tokenizers import Tokenizer
tk = Tokenizer.from_file(sys.argv[1]); out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
A = "amber birch coral dune fern glade harbor iris jade kelp lilac maple onyx pearl quartz raven slate tide umber violet willow yarrow zinc".split()
B = "badger crane dingo egret falcon gecko heron ibis jackal kestrel lynx marten newt otter plover quail robin stoat tern osprey".split()
HEAD = "Vault register. Every record has an owner and a vault code.\n\n"
def rec(i, rng):
    y, m, d = rng.randint(2019, 2025), rng.randint(1, 12), rng.randint(1, 28)
    code = f"{rng.choice(A)}-{rng.choice(B)}-{rng.randint(1000, 9999)}"
    return f"Record R-{i:04d} | owner: {rng.choice(A)}-{rng.choice(B)} | vault code: {code} | filed {y}-{m:02d}-{d:02d}", code
def row(tag, target_tokens, depth, seed):
    rng = random.Random(seed)
    per = len(tk.encode(rec(0, random.Random(1))[0] + "\n").ids)
    n = int(target_tokens / per)
    recs = [rec(i, rng) for i in range(n)]
    ask = min(n - 1, max(0, int(depth * n)))
    text = HEAD + "\n".join(r for r, _ in recs) + f"\n\nQuestion: What is the vault code of record R-{ask:04d}, and who owns it? Answer in one sentence."
    return {"name": f"needle-{tag}-d{int(depth * 100):02d}-R{ask:04d}", "input": text, "expected_output": recs[ask][1]}, len(tk.encode(text).ids)
for tag, tokens, k in (("32k", 33200, 12), ("64k", 66200, 8)):
    rows, lens = [], []
    for j in range(k):
        r, n = row(tag, tokens, 0.05 + 0.90 * j / (k - 1), tokens + j)
        rows.append(r); lens.append(n)
    (out / f"needle-{tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(tag, "rows", k, "tokens", min(lens), max(lens))
