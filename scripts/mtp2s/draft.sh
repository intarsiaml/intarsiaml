#!/usr/bin/env bash
# Lane mtp2s task 1 (2026-11-16): is the 5.0 ms draft two serial drafts? One short trace run of main's defaults at 2 streams:
# INTARSIA_CB_TRACE=1 (Emit lines: prepare_us, verify_us) + INTARSIA_SPEC_TRACE=1 ("spec: mtp draft (deferred verify) X ms" per sequence).
# Runs ON the Spark inside a fleetq slot, e.g. under scripts/mtp2s/run-draft.sh. Env: BIN, NAME, ENVPLUS, GEN.
set -uo pipefail
L=$HOME/intarsia-mtp2s; O=$L/out/draft; mkdir -p $O
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
BIN=${BIN:-$L/bin/base}; PORT=8194; NAME=${NAME:-d}; PAT="^$BIN serve"
rm -f $L/out/mem-$NAME.txt
env INTARSIA_CB_TRACE=1 INTARSIA_SPEC_TRACE=1 ${ENVPLUS:-} $BIN serve --backend cuda --model $M35 --model-id m --max-batch 2 --profile-every 0 --mem-budget 40 --max-seq 8192 --no-supervisor --host 127.0.0.1 --port $PORT > $O/serve-$NAME.log 2>&1 < /dev/null &
sleep 3; SP=$(pgrep -f "$PAT" | head -1)
nohup bash $L/memwatch.sh $SP $NAME >/dev/null 2>&1 < /dev/null & WP=$!
trap 'pkill -f "$PAT"; kill $WP 2>/dev/null' EXIT
for i in $(seq 1 100); do pgrep -f "$PAT" >/dev/null || { echo "serve died"; tail -20 $O/serve-$NAME.log; exit 1; }; curl -sf -m 3 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; sleep 3; done
echo "== up $(date +%T) MemAvailable MB $(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)"
python3 $L/src/scripts/mtp2s/gen2.py $PORT 2 128 3 > $O/warm-$NAME.txt 2>&1
python3 $L/src/scripts/mtp2s/gen2.py $PORT 2 ${GEN:-256} 0 > $O/gen-$NAME.txt 2>&1
cat $O/gen-$NAME.txt
echo "min MemAvailable MB: $(sort -k2 -n $L/out/mem-$NAME.txt | head -1)"
pkill -f "$PAT"; sleep 5
grep -a -c "spec: mtp draft (deferred" $O/serve-$NAME.log
