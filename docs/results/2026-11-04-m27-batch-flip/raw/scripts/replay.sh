#!/bin/zsh
# usage inside a fleetq slot: replay.sh <bin>
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
cd /opt/code/intarsiaml/.worktrees/m27-flip
BIN=$1; O=$PWD/.scratch/logs/replay1; mkdir -p $O
M=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4; PORT=18475
PR=docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-16384.txt
export INTARSIA_CB_TRACE=1
serve() { # tag extra-args...
  local T=$1; shift
  $BIN serve --backend metal --model $M --served-model-name intarsia --max-seq 12000 --max-batch 2 --spec off --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT "$@" > $O/$T-server.out 2> $O/$T-server.err < /dev/null &
  SP=$!
  local up=0; for i in {1..240}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 1; done
  [ $up = 1 ] || { echo "SERVER FAILED $T"; exit 1; }
}
stop() { kill $SP 2>/dev/null; sleep 2; kill -9 $SP 2>/dev/null; wait $SP 2>/dev/null; }
trap 'stop' EXIT
# S1: no reuse, so every cell is cold
INTARSIA_CB_NO_REUSE=1 serve s1
python3 .scratch/replay.py $PORT $PR $O/s1.json A:solo A:solo A:solo A0:solo:nolp B:behind-q0 B:behind-q0 B:behind-q0 B0:behind-q0:nolp Bq3:behind-q3 Aa:ahead-q0 Aa3:ahead-q3
stop
# S2: no reuse, busy chunk = solo chunk
INTARSIA_CB_NO_REUSE=1 serve s2 --busy-chunk 2048
python3 .scratch/replay.py $PORT $PR $O/s2.json A:solo B:behind-q0 B:behind-q0 Bq3:behind-q3
stop
# S3: default reuse. Warm from each cold path, in both directions.
serve s3
python3 .scratch/replay.py $PORT $PR $O/s3.json Acold:solo Awarm_solo:solo Awarm_behind:behind-q0
stop
serve s4
python3 .scratch/replay.py $PORT $PR $O/s4.json Bcold:behind-q0 Bwarm_solo:solo Bwarm_behind:behind-q0
stop
trap - EXIT
