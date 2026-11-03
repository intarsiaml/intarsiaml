"""Generate the m27-hs3 long-context retrieval dataset for `localeval quality --dataset` (judge none, Contains expected).

One shared context of CTX_TOKENS tokens (synthetic records), then 100 rows that differ only in the question at the end.
Each answer is a multi-token code the model must copy from the context, in a one-sentence reply, so decode attention
over the ~16K-token KV decides it. Deterministic (seeded); run with the deepeval venv (needs `tokenizers`):
  cd /opt/code/deepeval && uv run python <this file> /opt/code/deepeval/datasets/m27-hs3-longctx.jsonl
"""
import hashlib, json, random, sys
from tokenizers import Tokenizer

OUT = sys.argv[1]
TOK = "/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4/tokenizer.json"
CTX_TOKENS, N_ROWS, SEED = 16000, 100, 20261103
A = "amber azure birch cedar coral dune ember fern glade harbor iris jade kelp lilac maple north onyx pearl quartz raven slate tidal umber violet willow xenon yarrow zephyr".split()
B = "falcon otter heron badger lynx marten osprey plover quail robin stoat tern vole wren ibis newt".split()
rng = random.Random(SEED)
tk = Tokenizer.from_file(TOK)
n_tok = lambda s: len(tk.encode(s, add_special_tokens=False).ids)

recs, text, i = [], "", 0
head = "Vault register. Every record has an owner and a vault code.\n\n"
while True:
    owner = f"{rng.choice(A)}-{rng.choice(B)}"
    code = f"{rng.choice(A)}-{rng.choice(B)}-{rng.randrange(1000, 10000)}"
    line = f"Record R-{i:04d} | owner: {owner} | vault code: {code} | filed {2020 + rng.randrange(6)}-{1 + rng.randrange(12):02d}-{1 + rng.randrange(28):02d}\n"
    recs.append((i, owner, code))
    text += line
    i += 1
    if i % 25 == 0 and n_tok(head + text) >= CTX_TOKENS:
        break
ctx = head + text
picks = sorted(rng.sample(range(len(recs)), N_ROWS))
rng.shuffle(picks)
rows = []
for k, p in enumerate(picks):
    _, owner, code = recs[p]
    q = f"\nQuestion: What is the vault code of record R-{p:04d}, and who owns it? Answer in one sentence."
    rows.append({"name": f"lc{k:03d}-R{p:04d}", "input": ctx + q, "expected_output": code})
with open(OUT, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"records={len(recs)} context_tokens={n_tok(ctx)} rows={len(rows)} sha256={hashlib.sha256(open(OUT,'rb').read()).hexdigest()}")
