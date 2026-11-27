#!/usr/bin/env bash
# Lane ink-hip-specflip (coordinator 23:05 check 2): context / KV lines of one drafter vs the other, card $CARD (default 1), binary $BR
# (default main7), after the ABBA of the same window. PART 1: solo loads with --max-seq 1000000 (the context is then the memory-limited
# KV capacity, the engine's own budget formula: 0.5 x measured free memory / bytes per token): spec off, the default (DSpark k5), DFlash2
# (--spec dflash --spec-draft D --spec-k 4). PART 2: the recipe defaults: solo default (DSpark), serve --max-batch 16 default, and serve
# --max-batch 16 --spec dflash (refused by the loader). Out: logs/kv/<name>.err; one summary line each.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-specflip
RAW=$REPO/docs/results/2026-11-26-inkling2-hip-specflip/raw
M=/models/Inkling-2-Mini
D=/models/Inkling-2-Mini-DFlash2
BR=${BR:-main7}; PART=${PART:-1}
B=/work/.scratch/bin/intarsia-$BR
O=$REPO/logs/kv; mkdir -p "$O"
T0=$(date +%s); el() { echo $(( $(date +%s) - T0 )); }
sum() { echo "[kv] $1 t+$(el)s | $(grep -aE '^loaded|^dflash: draft|^--spec dflash|DSpark default|^Error|refuse|batching engine' "$O/$1.err" | tr '\n' ' ' | cut -c1-600)"; }
gen() {  # name, extra args
  local n=$1; shift
  timeout 60 bash "$RAW/run.sh" $B generate -m $M --backend hip --max-tokens 1 --prompt "Hello" "$@" > "$O/$n.out" 2> "$O/$n.err"
  echo "[kv] $n rc $?"; sum "$n"
}
srv() {  # name, extra args: start serve, wait for its loaded line (or exit), then remove the container
  local n=$1; shift
  ( timeout 60 bash "$RAW/run.sh" $B serve -m $M --backend hip --port 18080 "$@" > "$O/$n.out" 2> "$O/$n.err" ) &
  local p=$! i
  for i in $(seq 1 55); do
    grep -aqE '^loaded .*context|^Error|error:' "$O/$n.err" 2>/dev/null && break
    kill -0 $p 2>/dev/null || break
    sleep 1
  done
  docker rm -f ink-hip-specflip-run > /dev/null 2>&1; wait $p 2>/dev/null
  sum "$n"
}
echo "[kv] start $(date +%T) part $PART card ${CARD:-1} $BR"
if [ "$PART" = 1 ]; then
  gen off-big --max-seq 1000000 --spec off
  gen dspark-big --max-seq 1000000
  gen dfl-big --max-seq 1000000 --spec dflash --spec-draft $D --spec-k 4
else
  gen dspark-default
  srv b16-default --max-batch 16
  srv b16-dfl --max-batch 16 --spec dflash --spec-draft $D --spec-k 4
fi
echo "[kv] done $(date +%T) t+$(el)s"
