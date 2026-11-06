#!/usr/bin/env python3
"""natural-v1 speed driver on localeval's own client and frozen prompts (sha256-checked), for both arms (suite speed-spec only runs on a spec lane).
usage (cwd /opt/code/deepeval): uv run python nat.py URL OUT.jsonl SIZES REPS [IDS_OUT.json IDS_TOKENS]
One warmup call per size (discarded), then REPS measured calls, each nonce-first, thinking off, 256 forced tokens, temperature 0."""
import json, sys, urllib.request
from localeval.config import Target
from localeval.client import Client
from localeval import protocol
url, out, sizes, reps = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")], int(sys.argv[4])
suite_path = "/opt/code/model-benchmarks/schema/suite.v1.json"
suite = json.load(open(suite_path))
nat = protocol.load_natural_prompts(suite, suite_path, sizes)
t = Target(name="adhoc", url=url, model="intarsia", thinking="off")
c = Client(t)
rows = []
for s in sizes:
    for i in range(reps + 1):
        r = c.chat([{"role": "user", "content": nat[s]}], max_tokens=256, force_tokens=True, nonce=True)
        row = {"size": s, "rep": i, "warmup": i == 0, "ok": r.ok, "error": r.error, "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
               "ttft_s": r.ttft_s, "prefill_tok_s": r.prefill_tok_s, "decode_tok_s": r.decode_tok_s, "peak_decode_tok_s": r.peak_decode_tok_s,
               "cached_tokens": r.cached_tokens}
        rows.append(row)
        print(json.dumps(row), flush=True)
json.dump(rows, open(out, "w"), indent=1)
if len(sys.argv) > 5:  # served greedy ids on the raw natural text (no nonce), /v1/completions
    ids = {}
    for s in sizes:
        b = {"model": "intarsia", "prompt": nat[s], "max_tokens": int(sys.argv[6]), "temperature": 0, "ignore_eos": True, "return_token_ids": True}
        q = urllib.request.Request(url.rstrip("/") + "/completions", json.dumps(b).encode(), {"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(q, timeout=900))
        ids[str(s)] = {"ids": d["choices"][0]["token_ids"], "plen": d["usage"]["prompt_tokens"]}
        print("ids", s, len(ids[str(s)]["ids"]), "plen", ids[str(s)]["plen"], flush=True)
    json.dump(ids, open(sys.argv[5], "w"))
