#!/usr/bin/env bash
# Lane dc, gate G5/G7 measurements: spec off / dflash / dflash+dcoco, interleaved x ROUNDS, through localeval.
# Run INSIDE one fleetq lease (the server lives on spark1 for the whole lease; localeval sees FLEETQ_JOB and runs in it):
#   fleetq run --host spark1 --expect 150 --mem 26 --quiet gpu,cpu --label "dc: bench" -- scripts/dc/bench.sh [rounds] [configs...]
# Server files live under ~/intarsiaml-dc-out on spark1 (the only dirs this lane may use there).
set -uo pipefail
ROUNDS=${1:-3}; shift || true
CFGS=("${@:-off dflash dcoco}"); [ ${#CFGS[@]} -eq 1 ] && read -r -a CFGS <<< "${CFGS[0]}"
[ -n "${FLEETQ_JOB:-}" ] || { echo "bench.sh: run under fleetq run" >&2; exit 2; }
H=/home/nvidia/.cache/huggingface/hub
BIN='~/intarsiaml-dc-target/release/intarsia'
M8='$(ls -d '$H'/models--Qwen--Qwen3-8B/snapshots/* | head -1)'
DR='$(ls -d '$H'/models--TD-Labs--Qwen3-8B-DCOCO/snapshots/* | head -1)'
PORT=${PORT:-8091}; URL=http://spark-4e77:$PORT/v1
OUT=${OUT:-/opt/code/intarsiaml/.worktrees/dc/docs/results/2026-10-25-dc-dflash-dcoco/raw/bench}
mkdir -p "$OUT"
PID='~/intarsiaml-dc-out/serve.pid'; LOG='~/intarsiaml-dc-out/serve'
stop() { ssh spark1 "if [ -f $PID ]; then p=\$(cat $PID); kill \$p 2>/dev/null; for _ in \$(seq 1 60); do kill -0 \$p 2>/dev/null || break; sleep 1; done; kill -9 \$p 2>/dev/null; rm -f $PID; fi"; }
trap stop EXIT INT TERM
metric() { curl -s "http://spark-4e77:$PORT/metrics" | grep -E "^intarsia_spec_(proposed|accepted)_tokens_total" | tr '\n' ' '; }
for r in $(seq 1 "$ROUNDS"); do
  for cfg in "${CFGS[@]}"; do
    case $cfg in
      off)    SPEC="--spec off" ;;
      dflash) SPEC="--spec dflash --spec-draft $DR" ;;
      dcoco)  SPEC="--spec dflash --spec-draft $DR --dcoco" ;;
    esac
    tag="$cfg-r$r"
    echo "== $tag $(date +%T)"
    ssh spark1 "mkdir -p ~/intarsiaml-dc-out; nohup $BIN serve --backend cuda --model $M8 --model-id Qwen3-8B --port $PORT --host 0.0.0.0 --max-batch 1 --max-seq 8192 --mem-budget 24 $SPEC > $LOG-$tag.log 2>&1 < /dev/null & echo \$! > $PID"
    up=0; for _ in $(seq 150); do curl -sf "http://spark-4e77:$PORT/health" >/dev/null && { up=1; break; }; sleep 2; done
    [ $up = 1 ] || { echo "server $tag did not come up"; ssh spark1 "tail -5 $LOG-$tag.log"; stop; continue; }
    m0=$(metric)
    RECIPE="intarsia serve --backend cuda --model Qwen3-8B --max-batch 1 --max-seq 8192 $SPEC (lane dc, scripts/dc/bench.sh)"
    [ "${QMODE:-gsm8k}" = cat ] || (cd /opt/code/deepeval && uv run localeval speed --url "$URL" --model Qwen3-8B --thinking off --label "dc-$tag" --recipe "$RECIPE" --sizes 128,1024 --gen 128 --reps 3 --concurrency 1,4 --ignore-busy) > "$OUT/$tag.speed.txt" 2>&1
    m1=$(metric)
    if [ "${QMODE:-gsm8k}" = cat ]; then
      # speed-by-category set (recall / code / prose, 27 prompts): real answers instead of the filler sweep
      (cd /opt/code/deepeval && uv run localeval quality --url "$URL" --model Qwen3-8B --thinking off --label "${LABELP:-dc-cat}-$tag" --recipe "$RECIPE" --dataset "${DATASET:-datasets/categories-rcp.jsonl}" --judge none --max-tokens ${CATTOK:-256} --ignore-busy) > "$OUT/$tag.quality.txt" 2>&1
    else
      (cd /opt/code/deepeval && uv run localeval quality --url "$URL" --model Qwen3-8B --thinking off --label "dc-$tag" --recipe "$RECIPE" --bench gsm8k --limit 20 --max-tokens 320 --judge none --ignore-busy) > "$OUT/$tag.quality.txt" 2>&1
    fi
    m2=$(metric)
    echo "metrics before:  $m0" > "$OUT/$tag.spec.txt"; echo "metrics after speed: $m1" >> "$OUT/$tag.spec.txt"; echo "metrics after quality: $m2" >> "$OUT/$tag.spec.txt"
    ssh spark1 "grep -a 'dflash:' $LOG-$tag.log | head -2" >> "$OUT/$tag.spec.txt"
    stop
    tail -3 "$OUT/$tag.quality.txt" | cut -c1-200
  done
done
echo "== bench done"
