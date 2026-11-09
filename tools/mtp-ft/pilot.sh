#!/usr/bin/env bash
# Pilot of the whole pipeline on one box (spark1, CUDA): regenerate N prompts greedily with our own server (measures regen tok/s),
# dump them with `intarsia mtp-dump`, then the parity gate and the stock head's offline acceptance. Run ON spark1 inside one fleetq slot:
#   fleetq run --host spark1 --mem 45 --expect 60 --label "mtp-ft pilot" -- ssh spark1 'bash ~/mtp-ft/tools/pilot.sh'
# Env: CKPT (model dir), N (prompts, 192), CONC (16), MAXTOK (1024), OUT (~/mtp-ft/data/pilot), BIN (~/mtp-ft/intarsia), PORT (8197),
# LMHEAD (head.py --lm-head: the head the engine drafts with; "auto+fp8" for a bf16 checkpoint under a recipe lm_head_quant = "fp8").
set -uo pipefail
CKPT=${CKPT:-/mnt/models/Qwen3.8-27B-NVFP4-52d1adc5f38a}; N=${N:-192}; CONC=${CONC:-16}; MAXTOK=${MAXTOK:-1024}
W=$HOME/mtp-ft; OUT=${OUT:-$W/data/pilot}; BIN=${BIN:-$W/intarsia}; PORT=${PORT:-8197}
IMG=nvcr.io/nvidia/pytorch:26.07-py3; LMHEAD=${LMHEAD:-auto+fp8}
rel=${OUT#$W/}  # OUT as seen inside the container (/work/...)
mkdir -p "$OUT"; cd "$W/tools"
say() { echo "$(date +%T) $*" | tee -a "$OUT/pilot.log"; }
torch() { docker run --rm --gpus all --network host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v /mnt/models:/mnt/models:ro -v "$W":/work -w /work/tools "$IMG" "$@"; }
memguard() { while sleep 10; do a=$(free -g | awk '/Mem:/ {print $7}'); (( a < 15 )) && { say "memguard: available ${a} GB < 15, stopping server"; kill "$SPID" 2>/dev/null; return; }; done; }

say "serve: $CKPT max-batch $CONC (free GB $(free -g | awk '/Mem:/ {print $7}'))"
"$BIN" serve --backend cuda --model "$CKPT" --spec off --max-batch "$CONC" --max-seq 8192 --mem-budget 40 --host 127.0.0.1 --port "$PORT" \
  --no-supervisor > "$OUT/serve.log" 2>&1 &
SPID=$!; memguard & MG=$!
trap 'kill $SPID $MG 2>/dev/null' EXIT
for i in $(seq 1 120); do curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null && break; kill -0 $SPID 2>/dev/null || { say "serve died"; tail -20 "$OUT/serve.log"; exit 1; }; sleep 5; done
say "serve up"

t0=$(date +%s)
torch python regen.py --ckpt "$CKPT" --prompts /work/data/prompts.jsonl --endpoint "http://127.0.0.1:$PORT" --out-dir "/work/$rel/regen" \
  --max-tokens "$MAXTOK" --concurrency "$CONC" --limit "$N" --exclude $(ls "$W"/eval-cats/*.jsonl | sed "s|^$W|/work|") 2>&1 | tail -5 | tee -a "$OUT/pilot.log"
t1=$(date +%s)
[ -s "$OUT/regen/tokens.jsonl" ] || { say "regen wrote nothing, stopping"; exit 1; }
gen=$(python3 -c "import json;m=json.load(open('$OUT/regen/meta.json'));pl=m.get('prompt_len',m);import sys
t=[json.loads(l) for l in open('$OUT/regen/tokens.jsonl')];print(sum(len(x['tokens'])-pl.get(x['id'],0) for x in t), len(t))")
say "regen: $gen (generated tokens, sequences) in $((t1 - t0)) s -> $(python3 -c "g,n='$gen'.split();print(round(int(g)/max(1,$t1-$t0),1))") tok/s aggregate at concurrency $CONC"
kill $SPID; wait $SPID 2>/dev/null; kill $MG 2>/dev/null

say "mtp-dump"
"$BIN" mtp-dump --backend cuda --model "$CKPT" --max-seq 8192 --input "$OUT/regen/tokens.jsonl" --out "$OUT/dumps" --head-out 2>&1 | tail -3 | tee -a "$OUT/pilot.log"
ls "$OUT"/dumps/*.safetensors >/dev/null 2>&1 || { say "mtp-dump wrote nothing, stopping"; exit 1; }
say "parity"
torch python parity.py --ckpt "$CKPT" --dumps "/work/$rel/dumps" --lm-head "$LMHEAD" --json "/work/$rel/parity.json" 2>&1 | tail -12 | tee -a "$OUT/pilot.log"
say "accept (stock head)"
torch python accept.py --ckpt "$CKPT" --dumps "/work/$rel/dumps" --meta "/work/$rel/regen/meta.json" --lm-head "$LMHEAD" --json "/work/$rel/accept-stock.json" 2>&1 | tail -12 | tee -a "$OUT/pilot.log"
say "done"
