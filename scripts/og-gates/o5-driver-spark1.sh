#!/usr/bin/env bash
# O5, CUDA half: run scripts/og-gates/o5-tails.sh on spark1 (vLLM up, Qwen3-8B BF16, ~26 GB) and bring the dumps back.
#   fleetq run --host spark1 --mem 28 --expect 40 --label "og: ..." -- scripts/og-gates/o5-driver-spark1.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run --host spark1}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT=${OUT:-$REPO/docs/results/raw/2026-10-22-og-gates/o5-cuda}; mkdir -p "$OUT"
HOST=spark1; TREE=${TREE:-og-main}
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/driver.log"; }
BIN="\$HOME/intarsia-decfuse/$TREE/target/acc/release/intarsia"
rsh "test -x $BIN && $BIN --version | head -1; ls -la --time-style=full-iso $BIN; echo ik_matmul_fp8_gemm_lt symbol count: \$(grep -c ik_matmul_fp8_gemm_lt $BIN)" || { log "binary missing ($TREE): build it first"; exit 3; }
gb=$(( $(rsh "awk '/MemAvailable/ {print \$2}' /proc/meminfo") / 1024 / 1024 )); log "MemAvailable ${gb} GB"
(( gb >= 30 )) || { log "ABORT: memory"; exit 4; }
rsh "mkdir -p \$HOME/og-data/scripts \$HOME/og-data/o5-cuda"
scp -q "$REPO/../intarsiaml/golden/qwen3-8b-4k-fp32-top20.safetensors" "$HOST:og-data/" 2>/dev/null \
  || scp -q /opt/code/intarsiaml/golden/qwen3-8b-4k-fp32-top20.safetensors "$HOST:og-data/" || { log "golden copy failed"; exit 5; }
scp -q "$REPO"/scripts/og-gates/o5-tails.sh "$HOST:og-data/scripts/"
rsh "nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv" | tee "$OUT/gpu-before.txt"
rsh "cd \$HOME/og-data && BIN=$BIN bash scripts/o5-tails.sh cuda \$HOME/og-data/o5-cuda ${WHICH:-all}" 2>&1 | tee -a "$OUT/driver.log"
rsync -a "$HOST:og-data/o5-cuda/" "$OUT/"
log "dumps back in $OUT"
