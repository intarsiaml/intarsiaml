#!/usr/bin/env bash
# gc2: Metal 35B `qualify` on #104 alone (worktree /opt/code/wt-gc, chunk 256) and on #109 (/opt/code/wt-gc2, chunk 2048), each with the
# 4k-golden continue oracles committed on #104. No bench-long / ab / tp / envelope.
#   fleetq run --host endor --mem 42 --expect 40 --quiet gpu,cpu --label "gc2: qualify #104 + #109" -- qualify-both.sh OUTDIR
set -u
: "${FLEETQ_JOB:?run under fleetq}"
OUT=${1:?outdir}; mkdir -p "$OUT"; O=$(cd "$OUT" && pwd)
M=/Users/josef/Models/Qwen3.6-35B-A3B-NVFP4
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$O/job.log"; }
for T in 104:/opt/code/wt-gc 109:/opt/code/wt-gc2; do
  N=${T%%:*}; W=${T#*:}; cd $W
  log "== #$N in $W at $(git rev-parse --short HEAD) ($(git branch --show-current))"
  CARGO_TARGET_DIR=$W-target cargo build --release -p intarsia-cli --features metal 2>&1 | tail -1 | tee -a $O/job.log
  B=$W-target/release/intarsia; export INTARSIA_REPO=$W
  G=$W/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors; OR=$W/golden/oracle/qwen36-35b-a3b-nvfp4-cpu-oracle.safetensors
  t0=$(date +%s)
  $B qualify --backend metal --golden $G --oracle-golden $OR --model-name Qwen3.6-35B-A3B-NVFP4 --mem-budget 40 --stage-timeout-min 25 --model $M \
    --out $O/q-$N --skip envelope --skip bench-long --skip ab --skip tp < /dev/null > $O/q-$N.log 2>&1
  log "qualify #$N exit $? $(( $(date +%s) - t0 )) s"
  grep -E "\] [a-z0-9-]+: (PASS|FAIL|SKIP|ERROR|WARN|MISSING-BASELINE)" $O/q-$N.log | cut -c1-330 | tee -a $O/job.log
  grep -a -hE "^long_docs_4k +(continue|restore|aligned)|continue oracle|oracle rule|vs oracle" $O/q-$N/logs/acc-4k*.out | cut -c1-400 | tee -a $O/job.log
done
log done
