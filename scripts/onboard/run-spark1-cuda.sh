#!/usr/bin/env bash
# Lane `on`: one back-to-back `intarsia onboard` pass on spark1 (CUDA) with the goldens MADE from the vLLM reference server
# (fresh --golden-dir, so the golden-generation path runs, including the 4k golden), vLLM staying up as the reference.
# Run on the Mac INSIDE a fleetq slot (non-exclusive, --mem 40):
#   fleetq run --host spark1 --mem 40 --expect 60 --label "..." -- scripts/onboard/run-spark1-cuda.sh [tree] [run-name]
# Env: CAP_MIN (minutes the onboard may run, default 40), GOLDEN_FROM (default http://localhost:8000/v1 on the box).
set -uo pipefail
: "${FLEETQ_JOB:?run inside a fleetq slot}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TREE=${1:-on}
RUN=${2:-35b-cuda}
HOST=${HOST:-spark1}
CAP_MIN=${CAP_MIN:-40}
GOLDEN_FROM=${GOLDEN_FROM:-http://localhost:8000/v1}   # as the box sees it
REF_URL=${REF_URL:-http://spark-4e77:8000/v1}          # as the Mac sees it
REF_MODEL=${REF_MODEL:-guidance}
RAW=${RAW:-$REPO/docs/results/2026-10-22-on-onboard/raw/cuda}
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH; unset INTARSIA_GOLDEN_DIR;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
mkdir -p "$RAW/$RUN"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$RAW/$RUN/steps.log"; }
OUT='$HOME/onboard-runs/'$RUN
log "preflight"
rsh "ls -d \$HOME/.cache/huggingface/hub/models--* \$HOME/models/* 2>&1 | sed 's|.*/||'; which uv python3; python3 -c 'import numpy, safetensors; print(\"py deps ok\")' 2>&1 | tail -1; df -h \$HOME | tail -1; free -g | sed -n 2p; curl -s -m 10 ${GOLDEN_FROM%/}/models | head -c 160; echo" 2>&1 | tee -a "$RAW/$RUN/steps.log"
if ! rsh "curl -sf -m 10 ${GOLDEN_FROM%/}/models >/dev/null"; then log "ABORT: reference server $GOLDEN_FROM does not answer"; exit 1; fi
T0=$(date +%s)
log "build tree $TREE"
"$REPO/scripts/cuda-decode/build.sh" "$TREE" 2>&1 | tee "$RAW/$RUN/build.log" | tail -8
T1=$(date +%s); log "build_seconds=$((T1 - T0))"
T1=$(date +%s)
rsh "rm -rf $OUT; mkdir -p $OUT/golden-fresh"
# goldens: made here (uv) from the reference server, because the box has no python deps (qualify's own make step would fail there)
log "goldens from ${REF_URL} (on the Mac)"
"$REPO/scripts/onboard/make-goldens.sh" "$RAW/$RUN/goldens" Qwen3.6-35B-A3B-NVFP4 "$REF_URL" "${REF_MODEL}" 2>&1 | tee -a "$RAW/$RUN/steps.log"
T1b=$(date +%s); log "golden_seconds=$((T1b - T1))"
rsync -a "$RAW/$RUN/goldens/" "$HOST:onboard-runs/$RUN/golden-fresh/" --exclude '*.log' --exclude '*.txt'
T1=$T1b
log "onboard start"
rsh "cd \$HOME/intarsia-decfuse/$TREE && date -u +%FT%TZ > $OUT/start.txt && timeout ${CAP_MIN}m target/acc/release/intarsia onboard --model $M35 --backend cuda --golden-from $GOLDEN_FROM --mem-budget 38 --out $OUT/report -- --golden-dir $OUT/golden-fresh --no-unim > $OUT/driver.log 2>&1 < /dev/null; echo exit=\$? >> $OUT/driver.log; date -u +%FT%TZ > $OUT/end.txt; tail -30 $OUT/driver.log" 2>&1 | tee -a "$RAW/$RUN/steps.log"
T2=$(date +%s); log "onboard_wall_seconds=$((T2 - T1))"
rsync -a --exclude '*.safetensors' "$HOST:onboard-runs/$RUN/" "$RAW/$RUN/" 2>&1 | tail -2

du -sh "$RAW/$RUN"
