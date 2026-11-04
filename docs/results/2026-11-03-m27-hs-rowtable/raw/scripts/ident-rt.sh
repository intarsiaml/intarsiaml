#!/bin/zsh
# Greedy token-id identity under batching: serve --max-batch MB --spec off, 12 concurrent greedy /v1/completions requests of different
# lengths (150 to 40000 chars of the long-context prompt, each with a unique first line), return_token_ids, 2 rounds; ids per request and the
# batch counters (ROUNDS env, default 2; /metrics intarsia_batch_decode_rows_total) are saved.
# Inside a fleetq slot: fleetq run --host endor --mem 35 -- zsh ident-rt.sh <outdir> <tag> <bin> <maxbatch> [env...]
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
O=$1; TAG=$2; BIN=$3; MB=$4; mkdir -p $O; O=${O:A}
M=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4; PORT=18474
PR=docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-16384.txt
$BIN serve --backend metal --model $M --served-model-name intarsia --max-seq 12000 --max-batch $MB --spec off --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT > $O/$TAG-server.out 2> $O/$TAG-server.err < /dev/null &
SP=$!
trap 'kill $SP 2>/dev/null; sleep 2; kill -9 $SP 2>/dev/null' EXIT
up=0; for i in {1..240}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 1; done
[ $up = 1 ] || { echo "SERVER FAILED"; exit 1; }
python3 - $PORT $PR $O $TAG <<'PY'
import json, sys, threading, urllib.request
port, pr, out, tag = sys.argv[1:5]
text = open(pr).read()
lens = [150, 400, 900, 1800, 3000, 5000, 8000, 11000, 15700, 22000, 31000, 40000]
prompts = [f"Request {k}, {n} characters of context follows.\n" + text[:n] for k, n in enumerate(lens)]
res = {}
def call(k, rnd):
    body = json.dumps({"model": "intarsia", "prompt": prompts[k], "max_tokens": 200, "temperature": 0, "ignore_eos": True, "return_token_ids": True}).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", body, {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(r, timeout=900))
    c = d["choices"][0]
    res[f"r{rnd}-q{k}"] = {"ids": c.get("token_ids"), "n": d["usage"]["completion_tokens"], "text": c["text"]}
import os
for rnd in range(1, int(os.environ.get('ROUNDS', '2')) + 1):
    ts = [threading.Thread(target=call, args=(k, rnd)) for k in range(len(lens))]
    [t.start() for t in ts]; [t.join() for t in ts]
json.dump(res, open(f"{out}/{tag}-ids.json", "w"))
print(tag, "requests", len(res), "ids lens", sorted({len(v["ids"] or []) for v in res.values()}))
PY
curl -s -m 5 http://127.0.0.1:$PORT/metrics | grep -E "^intarsia_batch_(decode_rows_total|decode_steps_total|iterations_total|mixed_iterations_total|requests_finished_total)" > $O/$TAG-batch-counters.txt
