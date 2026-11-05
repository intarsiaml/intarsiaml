#!/bin/zsh
# One single-stream speed arm: start `intarsia serve --max-batch 1 --spec off` for MODEL, run localeval speed at SIZES, stop the server.
# Inside a fleetq slot (--quiet gpu,cpu). usage: speed-arm.sh TAG BIN OUTDIR MODEL ; env SIZES (required, space separated), GEN (128), REPS (3), UNIM (1: log to unanimis, 0: --no-unim)
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
TAG=${1:?tag}; BIN=${2:?bin}; OUT=${3:?out}; M=${4:?model}; mkdir -p $OUT; OUT=${OUT:A}; PORT=18483
CMD=($BIN serve --backend metal --model $M --served-model-name intarsia --max-seq 12000 --max-batch 1 --spec off --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT)
$CMD > $OUT/$TAG-server.out 2> $OUT/$TAG-server.err < /dev/null &
SP=$!
trap 'kill $SP 2>/dev/null; sleep 2; kill -9 $SP 2>/dev/null' EXIT
up=0; for i in {1..300}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 1; done
[ $up = 1 ] || { echo "SERVER FAILED" >&2; exit 1; }
cd /opt/code/deepeval
NU=(); [ "${UNIM:-1}" = 0 ] && NU=(--no-unim)
for S in ${=SIZES}; do
  SN=${S[1,24]}
  uv run localeval speed --url http://127.0.0.1:$PORT/v1 --model intarsia --sizes $S --gen ${GEN:-128} --reps ${REPS:-3} \
    --thinking off $NU --label "m35peel-$TAG-$SN" --recipe "${CMD[*]}" > $OUT/$TAG-$SN-localeval.out 2> $OUT/$TAG-$SN-localeval.err
  echo "localeval $S exit $?"
done
