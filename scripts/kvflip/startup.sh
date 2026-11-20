#!/usr/bin/env bash
# kvflip: the flip binary on spark1 (tp1, 35B NVFP4, --spec mtp), the knob unset / =0 / =1: the startup line says on / off / on, and one greedy
# request (short prompt, 128 tokens) returns the same ids in all three. vLLM stays up.
#   FLEETQ_OWNER=claude-kvflip fleetq run --host spark1 --quiet gpu,cpu --mem 40 --expect 15 --label "kvflip startup check" -- bash scripts/kvflip/startup.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUT=$REPO/docs/results/2026-11-20-kvflip/raw; W='~/intarsiaml-kvflip/out'; BIN='$HOME/intarsiaml-kvflip/bin/intarsia-flip'
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
PORT=8194
r() { ssh -n -o BatchMode=yes ${HOST:-spark1} "export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
for arm in unset off on; do
  case $arm in unset) E="" ;; off) E="INTARSIA_CUDA_ATTN_KV_ONCE=0" ;; on) E="INTARSIA_CUDA_ATTN_KV_ONCE=1" ;; esac
  SPID=$(r "nohup env $E $BIN serve -m $M35 --backend cuda --max-seq 8192 --mem-budget 30 --profile-every 0 --spec mtp --spec-k 2 --port $PORT --host 0.0.0.0 --served-model-name guidance > $W/startup-$arm.log 2>&1 < /dev/null & echo \$!")
  up=0; for i in $(seq 1 100); do curl -sf "http://${HN:-spark-4e77}:$PORT/health" >/dev/null && { up=1; break; }; sleep 3; done
  [[ $up == 1 ]] || { echo "$arm: did not come up"; r "tail -20 $W/startup-$arm.log"; r "kill $SPID"; continue; }
  echo "== $arm (env '$E'): $(r "grep -a 'cuda attention' $W/startup-$arm.log")"
  curl -s "http://${HN:-spark-4e77}:$PORT/v1/completions" -H 'Content-Type: application/json' -d '{"model":"guidance","prompt":"Explain how a hash map handles collisions, with an example in Python.","max_tokens":128,"temperature":0,"ignore_eos":true,"return_token_ids":true}' > "$OUT/startup-$arm.json"
  python3 -c "import json,sys; d=json.load(open('$OUT/startup-$arm.json'))['choices'][0]; print('$arm ids', len(d['token_ids']), hash(tuple(d['token_ids'])) if False else d['token_ids'][:6])"
  r "kill $SPID; for i in \$(seq 1 30); do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID"
  r "cat $W/startup-$arm.log" > "$OUT/startup-$arm.log"
done
python3 - "$OUT" <<'PY'
import json, sys
ids = {a: json.load(open(f"{sys.argv[1]}/startup-{a}.json"))["choices"][0]["token_ids"] for a in ("unset", "off", "on")}
print("ids identical unset/off/on:", ids["unset"] == ids["off"] == ids["on"], len(ids["unset"]))
PY
