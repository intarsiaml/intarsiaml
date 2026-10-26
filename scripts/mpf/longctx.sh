#!/usr/bin/env bash
# Lane mpf: 35B NVFP4 prefill tok/s, TTFT and decode tok/s at 4k/16k/32k prompt tokens through `localeval speed` (after scripts/cmtp/longctx.sh), arms off (--spec off) and
# mtp (--spec mtp --spec-k 2; ARMS overrides: off mtp), interleaved x ROUNDS, one `serve --max-batch 1` per arm. Run INSIDE one fleetq lease:
#   fleetq run --host spark1 --quiet gpu,cpu --mem 50 --expect 40 --label "cmtp: long context" -- scripts/mpf/longctx.sh [rounds]
set -uo pipefail
ROUNDS=${1:-2}
[ -n "${FLEETQ_JOB:-}" ] || { echo "longctx.sh: run under fleetq run" >&2; exit 2; }
BIN=${BIN:-'~/intarsiaml-mpf/target-dev/release/intarsia'}
M35='$(ls -d /home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)'
PORT=${PORT:-8096}; URL=http://spark-4e77:$PORT/v1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
OUT=${OUT:-$REPO/docs/results/2026-10-27-mpf-macro-prefill-mtp/raw/longctx}; mkdir -p "$OUT"
PID='~/intarsiaml-mpf/out/serve-lc.pid'; LOG='~/intarsiaml-mpf/out/serve-lc'
stop() { ssh spark1 "if [ -f $PID ]; then p=\$(cat $PID); kill \$p 2>/dev/null; for _ in \$(seq 1 60); do kill -0 \$p 2>/dev/null || break; sleep 1; done; kill -9 \$p 2>/dev/null; rm -f $PID; fi" < /dev/null; }
trap stop EXIT INT TERM
for r in $(seq 1 "$ROUNDS"); do
  ARMS=${ARMS:-"off mtp"}; order="$ARMS"; (( r % 2 == 0 )) && order=$(echo $ARMS | awk '{for (i=NF;i>=1;i--) printf "%s ", $i}')
  for cfg in $order; do
    case $cfg in off) SPEC="--spec off" ;; auto) SPEC="" ;; mtp*) SPEC="--spec mtp --spec-k ${cfg#mtp}"; [ "$cfg" = mtp ] && SPEC="--spec mtp --spec-k 2" ;; esac
    tag="$cfg-r$r"; echo "== $tag $(date +%T)"
    ssh spark1 "nohup $BIN serve --backend cuda --model $M35 --model-id qwen36-35b --port $PORT --host 0.0.0.0 --max-batch 1 --max-seq 36864 --mem-budget 40 --no-supervisor $SPEC > $LOG-$tag.log 2>&1 < /dev/null & echo \$! > $PID" < /dev/null
    up=0; for _ in $(seq 150); do curl -sf "http://spark-4e77:$PORT/health" >/dev/null && { up=1; break; }; sleep 2; done
    [ $up = 1 ] || { echo "server $tag did not come up"; ssh spark1 "tail -5 $LOG-$tag.log" < /dev/null; stop; continue; }
    RECIPE="intarsia serve --backend cuda --model Qwen3.6-35B-A3B-NVFP4 --max-batch 1 --max-seq 36864 $SPEC (lane mpf, scripts/mpf/longctx.sh)"
    (cd /opt/code/deepeval && uv run localeval speed --url "$URL" --model qwen36-35b --thinking off --label "mpf-lc-$tag" --recipe "$RECIPE" --sizes 4096,16384,32768 --gen 128 --reps ${REPS:-5} --concurrency 1 --ignore-busy) > "$OUT/$tag.speed.txt" 2>&1 < /dev/null
    tail -12 "$OUT/$tag.speed.txt" | cut -c1-200
    ssh spark1 "grep -aE 'default of recipe|speculative|macro' $LOG-$tag.log | head -2 | cut -c1-160" < /dev/null > "$OUT/$tag.serve.txt" 2>&1
    stop
  done
done
echo "== longctx done"
