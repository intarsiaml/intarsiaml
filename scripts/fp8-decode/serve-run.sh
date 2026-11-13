#!/usr/bin/env bash
# fp8-decode: start `intarsia serve` on spark1 (port 8291), run scripts/fp8-decode/reps.py against it, stop it, keep the logs.
# Run under fleetq (the job's FLEETQ_JOB is passed to the remote side):
#   ARM=name OUT=dir SERVE_ENV='INTARSIA_HOST_PROFILE=1' SERVE_ARGS='--kv-dtype fp8' REPS_ARGS='--sizes 65536 --reps 5' \
#   fleetq run --host spark1 --mem 60 --quiet gpu --expect 20 --label "fp8-decode: ..." -- bash scripts/fp8-decode/serve-run.sh
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}" "${ARM:?}" "${OUT:?}"
H="${BHOST:-spark1}"; HN="${BHOSTNAME:-spark-4e77}"; PORT="${PORT:-8291}"
L="${ABL:-~/intarsia-fp8-decode}"; BIN="${BIN:-$L/bin/intarsia}"
SNAP='~/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5'
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT"
rsh() { ssh -n -o BatchMode=yes "$H" "export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
RLOG="$L/out/serve-$ARM.log"
rsh "mkdir -p $L/out; pkill -f '[s]erve --backend cuda.*--port $PORT' || true"
rsh "cd $L && nohup env ${SERVE_ENV:-} $BIN serve --backend cuda --model $SNAP --port $PORT --max-seq 131072 --mem-budget 40 \
  --max-batch 2 --profile-every 0 --served-model-name guidance ${SERVE_ARGS:-} > $RLOG 2>&1 & echo \$! > $L/out/serve.pid"
# memory guard: kill the server if MemAvailable drops under 10 GB
scp -q -o BatchMode=yes "$HERE/memguard.sh" "$H:${L#\~/}/memguard.sh"
rsh "nohup bash $L/memguard.sh $L/out/serve.pid $L/out/mem-$ARM.log >/dev/null 2>&1 &"
for i in $(seq 1 120); do
  if rsh "grep -q 'listening on' $RLOG"; then break; fi
  if ! rsh "kill -0 \$(cat $L/out/serve.pid)" 2>/dev/null; then echo "server died"; rsh "tail -20 $RLOG"; exit 1; fi
  sleep 2
done
echo "$(date +%T) server up"
set +e
(cd /opt/code/deepeval && uv run --project /opt/code/deepeval python "$HERE/reps.py" --url "http://$HN:$PORT/v1" --model guidance \
  --arm "$ARM" --out "$OUT" ${REPS_ARGS:-}) 2>&1 | grep -v '^\s*$' | tee "$OUT/$ARM.out"
rc=${PIPESTATUS[0]}
set -e
rsh "kill \$(cat $L/out/serve.pid); sleep 3; pkill -9 -f '[s]erve --backend cuda.*--port $PORT' || true"
scp -q -o BatchMode=yes "$H:$RLOG" "$OUT/serve-$ARM.log"
scp -q -o BatchMode=yes "$H:$L/out/mem-$ARM.log" "$OUT/mem-$ARM.log" || true
echo "min MemAvailable GB: $(sort -k2 -n "$OUT/mem-$ARM.log" | head -1)"
exit $rc
