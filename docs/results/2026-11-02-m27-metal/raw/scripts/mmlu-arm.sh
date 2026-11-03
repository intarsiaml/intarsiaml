#!/bin/zsh
# m27 MMLU arm on endor: start `intarsia serve` for the 27B, run localeval quality (mmlu, 5-shot, 25 per subject = n 1425), stop the server.
# Inside a fleetq slot: fleetq run --host endor --quiet gpu,cpu --mem 28 -- zsh speed-arm.sh <tag> <bin> <outdir> [serve args...]
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
TAG=${1:?tag}; BIN=${2:?bin}; OUT=${3:?out}; shift 3; mkdir -p $OUT; OUT=${OUT:A}
M=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4; PORT=18471
CMD=($BIN serve --backend metal --model $M --served-model-name intarsia --max-seq 20000 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT "$@")
$CMD > $OUT/$TAG-server.out 2> $OUT/$TAG-server.err < /dev/null &
SP=$!
trap 'kill $SP 2>/dev/null; sleep 2; kill -9 $SP 2>/dev/null' EXIT
up=0; for i in {1..240}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 1; done
[ $up = 1 ] || { echo "SERVER FAILED" >&2; exit 1; }
cd /opt/code/deepeval
uv run localeval quality --url http://127.0.0.1:$PORT/v1 --model intarsia --thinking off --structured none --bench mmlu --limit 25 \
  --label "m27-$TAG" --recipe "${CMD[*]}" > $OUT/$TAG-localeval.out 2> $OUT/$TAG-localeval.err
echo "localeval exit $?"
