#!/bin/zsh
# One batched-decode speed arm on endor: start `intarsia serve --max-batch MB --spec off` for the 27B, run localeval speed
# --concurrency at 4k and at 16k (one localeval invocation each, same server), stop the server.
# Inside a fleetq slot: fleetq run --host endor --quiet gpu,cpu --mem 45 -- zsh conc-arm.sh <tag> <bin> <outdir> <maxbatch> <conc-list> [serve args...]
# Env: SIZES (default "4096 16384"), GEN (default 512), REPS (single-stream reps, default 1); env vars of the caller reach the server.
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
TAG=${1:?tag}; BIN=${2:?bin}; OUT=${3:?out}; MB=${4:?maxbatch}; CONC=${5:?conc}; shift 5; mkdir -p $OUT; OUT=${OUT:A}
M=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4; PORT=18473
CMD=($BIN serve --backend metal --model $M --served-model-name intarsia --max-seq 18000 --max-batch $MB --spec off --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT "$@")
$CMD > $OUT/$TAG-server.out 2> $OUT/$TAG-server.err < /dev/null &
SP=$!
trap 'kill $SP 2>/dev/null; sleep 2; kill -9 $SP 2>/dev/null' EXIT
up=0; for i in {1..240}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 1; done
[ $up = 1 ] || { echo "SERVER FAILED" >&2; exit 1; }
cd /opt/code/deepeval
for S in ${=SIZES:-4096 16384}; do
  uv run localeval speed --url http://127.0.0.1:$PORT/v1 --model intarsia --sizes $S --gen ${GEN:-512} --reps ${REPS:-1} --concurrency $CONC \
    --thinking off --label "m27rt-$TAG-$S" --recipe "${CMD[*]} (env: ${INTARSIA_METAL_ATTN_HS:+HS=$INTARSIA_METAL_ATTN_HS }${INTARSIA_METAL_ATTN_HSB:+HSB=$INTARSIA_METAL_ATTN_HSB})" \
    > $OUT/$TAG-$S-localeval.out 2> $OUT/$TAG-$S-localeval.err
  echo "localeval $S exit $?"
done
curl -s -m 5 http://127.0.0.1:$PORT/metrics > $OUT/$TAG-metrics.txt 2>/dev/null
