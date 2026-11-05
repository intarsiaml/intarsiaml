#!/bin/zsh
# usage (inside a fleetq slot): ids-run.sh OUTDIR PROMPTS.json LENS MODEL_DIR PORT TAG:BIN [TAG:BIN ...]
# 256 greedy ids per prompt length, solo and behind a decoding row, one server per binary; INTARSIA_CB_TRACE gives the chunk schedule.
set -u
[ -n "${FLEETQ_JOB:-}" ] || exit 2
cd /opt/code/intarsiaml/.worktrees/m35-peel
O=${1:?out}; PR=${2:?prompts}; LENS=${3:?lens}; M=${4:?model}; PORT=${5:?port}; shift 5
mkdir -p $O
for tb in "$@"; do
  t=${tb%%:*}; BIN=${tb#*:}
  INTARSIA_CB_TRACE=1 INTARSIA_CB_NO_REUSE=1 $BIN serve --backend metal --model $M --served-model-name intarsia --max-seq 12000 --max-batch 2 --spec off --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT ${=SERVE_EXTRA:-} > $O/server-$t.out 2> $O/server-$t.err < /dev/null &
  SP=$!
  trap 'kill $SP 2>/dev/null; sleep 2; kill -9 $SP 2>/dev/null' EXIT
  up=0; for i in {1..300}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 1; done
  [ $up = 1 ] || { echo "SERVER FAILED $t"; tail -3 $O/server-$t.err; exit 1; }
  python3 .scratch/ids256.py $PORT $O/ids.json $t $LENS $PR; echo "ids $t exit $?"
  kill $SP 2>/dev/null; sleep 2; kill -9 $SP 2>/dev/null; wait $SP 2>/dev/null
done
trap - EXIT
