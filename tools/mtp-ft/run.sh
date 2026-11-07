#!/usr/bin/env bash
# Full depth-1 run on one box (spark1, CUDA): regenerate PROMPTS greedily with our own server, dump with `intarsia mtp-dump`,
# train the head at each LR in LRS, then score every tuned head against the stock head on the held-out files, per category.
# Resumable: regen uses --resume, and a step whose output exists is skipped. Run ON spark1 inside one fleetq slot:
#   fleetq run --host spark1 --mem 45 --expect 900 --label "mtp-ft run" -- ssh spark1 'OUT=... bash ~/mtp-ft/tools/run.sh'
# Env: CKPT, PROMPTS (~/mtp-ft/data/prompts-mix.jsonl), OUT (~/mtp-ft/data/run1), CONC (16), MAXTOK (1024), LRS ("2e-5 1e-4"),
# EPOCHS (2), HELDOUT (10, % of files), BIN, PORT, LMHEAD (as pilot.sh).
set -uo pipefail
CKPT=${CKPT:-/mnt/models/Qwen3.8-27B-Quark-AWQ-MXFP4}; CONC=${CONC:-16}; MAXTOK=${MAXTOK:-1024}
W=$HOME/mtp-ft; PROMPTS=${PROMPTS:-$W/data/prompts-mix.jsonl}; OUT=${OUT:-$W/data/run1}; BIN=${BIN:-$W/intarsia}; PORT=${PORT:-8197}
LRS=${LRS:-"2e-5 1e-4"}; EPOCHS=${EPOCHS:-2}; HELDOUT=${HELDOUT:-10}
IMG=nvcr.io/nvidia/pytorch:26.07-py3; LMHEAD=${LMHEAD:-auto+fp8}
rel=${OUT#$W/}
mkdir -p "$OUT"; cd "$W/tools"
say() { echo "$(date '+%F %T') $*" | tee -a "$OUT/run.log"; }
torch() { docker run --rm --gpus all --network host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v /mnt/models:/mnt/models:ro -v "$W":/work -w /work/tools "$IMG" "$@" 2>&1 | grep -vE '^$|^(NOTE|  )|NVIDIA|====|Copyright|governed|found at|Mellanox|detected|Multi-node|terms'; }
memguard() { while sleep 10; do a=$(free -g | awk '/Mem:/ {print $7}'); (( a < 15 )) && { say "memguard: available ${a} GB < 15, stopping server"; kill "$SPID" 2>/dev/null; return; }; done; }
n_prompts=$(wc -l < "$PROMPTS")

if [ "$(wc -l 2>/dev/null < "$OUT/regen/tokens.jsonl" || echo 0)" -lt "$n_prompts" ]; then
  say "serve: $CKPT max-batch $CONC (free GB $(free -g | awk '/Mem:/ {print $7}'))"
  "$BIN" serve --backend cuda --model "$CKPT" --spec off --max-batch "$CONC" --max-seq 8192 --mem-budget 40 --host 127.0.0.1 --port "$PORT" \
    --no-supervisor > "$OUT/serve.log" 2>&1 &
  SPID=$!; memguard & MG=$!
  trap 'kill $SPID $MG 2>/dev/null' EXIT
  for i in $(seq 1 120); do curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null && break; kill -0 $SPID 2>/dev/null || { say "serve died"; tail -20 "$OUT/serve.log"; exit 1; }; sleep 5; done
  say "serve up; regen $n_prompts prompts"
  t0=$(date +%s)
  torch python regen.py --ckpt "$CKPT" --prompts "/work/${PROMPTS#$W/}" --endpoint "http://127.0.0.1:$PORT" --out-dir "/work/$rel/regen" \
    --max-tokens "$MAXTOK" --concurrency "$CONC" --resume --exclude $(ls "$W"/eval-cats/*.jsonl | sed "s|^$W|/work|") | tail -5 | tee -a "$OUT/run.log"
  t1=$(date +%s)
  kill $SPID; wait $SPID 2>/dev/null; kill $MG 2>/dev/null; trap - EXIT
  say "regen: $(wc -l < "$OUT/regen/tokens.jsonl") sequences in $((t1 - t0)) s"
fi
[ -s "$OUT/regen/tokens.jsonl" ] || { say "regen wrote nothing, stopping"; exit 1; }

if ! ls "$OUT"/dumps/*.safetensors >/dev/null 2>&1; then
  say "mtp-dump"
  "$BIN" mtp-dump --backend cuda --model "$CKPT" --max-seq 8192 --input "$OUT/regen/tokens.jsonl" --out "$OUT/dumps" 2>&1 | tail -3 | tee -a "$OUT/run.log"
  ls "$OUT"/dumps/*.safetensors >/dev/null 2>&1 || { say "mtp-dump wrote nothing, stopping"; exit 1; }
  say "dumps: $(ls "$OUT"/dumps/*.safetensors | wc -l) files, $(du -sh "$OUT/dumps" | cut -f1)"
fi

for lr in $LRS; do
  t="$OUT/train-lr$lr"
  [ -s "$t/mtp-tuned.safetensors" ] && continue
  say "train lr $lr, $EPOCHS epochs, held out $HELDOUT%"
  torch python train.py --ckpt "$CKPT" --dumps "/work/$rel/dumps" --meta "/work/$rel/regen/meta.json" --out "/work/${t#$W/}" \
    --lm-head "$LMHEAD" --lr "$lr" --epochs "$EPOCHS" --warmup 10 --eval-every 20 --eval-files 400 --heldout-pct "$HELDOUT" | tail -8 | tee -a "$OUT/run.log"
  [ -s "$t/mtp-tuned.safetensors" ] || { say "train lr $lr wrote no head, stopping"; exit 1; }
done

# Held-out files only (the split is by file id, so it is the same for every LR).
h="$OUT/heldout"; rm -rf "$h"; mkdir -p "$h"
python3 -c "import json,sys;print('\n'.join(json.load(open(sys.argv[1]))['heldout']))" "$OUT/train-lr${LRS%% *}/split.json" \
  | while read -r id; do ln -s "../dumps/$id.safetensors" "$h/$id.safetensors"; done
say "accept on $(ls "$h" | wc -l) held-out files"
for lr in $LRS; do
  say "tuned lr $lr vs stock"
  torch python accept.py --ckpt "$CKPT" --dumps "/work/$rel/heldout" --meta "/work/$rel/regen/meta.json" --lm-head "$LMHEAD" \
    --mtp-file "/work/$rel/train-lr$lr/mtp-tuned.safetensors" --json "/work/$rel/accept-lr$lr.json" | tail -8 | tee -a "$OUT/run.log"
done
say "done"
