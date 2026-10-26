#!/usr/bin/env bash
# Lane dc, section 13: 35B NVFP4 decode tok/s at 4k/16k/32k, three arms interleaved x ROUNDS, through localeval speed:
#   off       --spec off
#   multirow  --spec mtp, INTARSIA_CUDA_VERIFY_ATTN_PER_ROW=0 (the old multi-row verify attention)
#   perrow    --spec mtp, new default (per-row verify attention)
# Run INSIDE one fleetq lease:
#   fleetq run --host spark1 --expect 70 --mem 40 --quiet gpu,cpu --label "dc: bench35" -- scripts/dc/bench35.sh [rounds]
set -uo pipefail
ROUNDS=${1:-2}
[ -n "${FLEETQ_JOB:-}" ] || { echo "bench35.sh: run under fleetq run" >&2; exit 2; }
H=/home/nvidia/.cache/huggingface/hub
BIN='~/intarsiaml-dc-target/release/intarsia'
M35='$(ls -d '$H'/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)'
PORT=${PORT:-8092}; URL=http://spark-4e77:$PORT/v1
OUT=${OUT:-/opt/code/intarsiaml/.worktrees/dc/docs/results/2026-10-25-dc-dflash-dcoco/raw/bench35}
mkdir -p "$OUT"
PID='~/intarsiaml-dc-out/serve35.pid'; LOG='~/intarsiaml-dc-out/serve35'
stop() { ssh spark1 "if [ -f $PID ]; then p=\$(cat $PID); kill \$p 2>/dev/null; for _ in \$(seq 1 60); do kill -0 \$p 2>/dev/null || break; sleep 1; done; kill -9 \$p 2>/dev/null; rm -f $PID; fi"; }
trap stop EXIT INT TERM
for r in $(seq 1 "$ROUNDS"); do
  for cfg in off multirow perrow; do
    case $cfg in
      off)      SPEC="--spec off"; ENVX="" ;;
      multirow) SPEC="--spec mtp --spec-k 2"; ENVX="INTARSIA_CUDA_VERIFY_ATTN_PER_ROW=0" ;;
      perrow)   SPEC="--spec mtp --spec-k 2"; ENVX="" ;;
    esac
    tag="$cfg-r$r"; echo "== $tag $(date +%T)"
    ssh spark1 "mkdir -p ~/intarsiaml-dc-out; nohup env $ENVX $BIN serve --backend cuda --model $M35 --model-id Qwen3.6-35B-A3B-NVFP4 --port $PORT --host 0.0.0.0 --max-batch 1 --max-seq 36864 --mem-budget 40 $SPEC > $LOG-$tag.log 2>&1 < /dev/null & echo \$! > $PID"
    up=0; for _ in $(seq 150); do curl -sf "http://spark-4e77:$PORT/health" >/dev/null && { up=1; break; }; sleep 2; done
    [ $up = 1 ] || { echo "server $tag did not come up"; ssh spark1 "tail -5 $LOG-$tag.log"; stop; continue; }
    RECIPE="intarsia serve --backend cuda --model Qwen3.6-35B-A3B-NVFP4 --max-batch 1 --max-seq 36864 $SPEC ${ENVX:+env $ENVX} (lane dc, scripts/dc/bench35.sh)"
    (cd /opt/code/deepeval && uv run localeval speed --url "$URL" --model Qwen3.6-35B-A3B-NVFP4 --thinking off --label "dc35-$tag" --recipe "$RECIPE" --sizes 4096,16384,32768 --gen 128 --reps 3 --concurrency 1 --ignore-busy) > "$OUT/$tag.speed.txt" 2>&1
    tail -12 "$OUT/$tag.speed.txt" | cut -c1-200
    stop
  done
done
echo "== bench35 done"
