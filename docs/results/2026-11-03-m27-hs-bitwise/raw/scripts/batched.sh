#!/bin/zsh
# usage: batched.sh OUTDIR TAG BIN  (env passed by the caller) ; serve --max-batch 2 --spec off, two concurrent greedy /v1/completions (short prompt + 4k prompt), twice
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
O=$1; TAG=$2; BIN=$3; mkdir -p $O; O=${O:A}
M=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4; PORT=18472
PR=docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-4096.txt
$BIN serve --backend metal --model $M --served-model-name intarsia --max-seq 8192 --max-batch 2 --spec off --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT > $O/$TAG-server.out 2> $O/$TAG-server.err < /dev/null &
SP=$!
trap 'kill $SP 2>/dev/null; sleep 2; kill -9 $SP 2>/dev/null' EXIT
up=0; for i in {1..240}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 1; done
[ $up = 1 ] || { echo "SERVER FAILED"; exit 1; }
python3 - $PORT $PR $O $TAG <<'PY'
import json, sys, threading, urllib.request
port, pr, out, tag = sys.argv[1:5]
prompts = {"short": "Write a Rust function that parses a CSV line into fields, handling quoted commas, with unit tests.", "4k": open(pr).read()}
def call(name, rnd, res):
    body = json.dumps({"model": "intarsia", "prompt": prompts[name], "max_tokens": 128, "temperature": 0}).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", body, {"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(r, timeout=600))
    res[(rnd, name)] = d["choices"][0]["text"]
res = {}
for rnd in (1, 2):
    ts = [threading.Thread(target=call, args=(n, rnd, res)) for n in prompts]
    [t.start() for t in ts]; [t.join() for t in ts]
json.dump({f"{k[0]}-{k[1]}": v for k, v in res.items()}, open(f"{out}/{tag}-replies.json", "w"), indent=1)
print(tag, {f"{k[0]}-{k[1]}": len(v) for k, v in res.items()})
PY
