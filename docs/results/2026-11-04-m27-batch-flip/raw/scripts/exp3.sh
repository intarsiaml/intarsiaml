#!/bin/zsh
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
cd /opt/code/intarsiaml/.worktrees/m27-flip
BIN=$PWD/.scratch/intarsia.head; O=$PWD/.scratch/logs/exp3; mkdir -p $O
M=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4; PORT=18475
PR=docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-16384.txt
export INTARSIA_CB_TRACE=1 INTARSIA_CB_NO_REUSE=1
serve() { local T=$1; shift
  $BIN serve --backend metal --model $M --served-model-name intarsia --max-seq 12000 --max-batch 2 --spec off --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT "$@" > $O/$T-server.out 2> $O/$T-server.err < /dev/null &
  SP=$!
  local up=0; for i in {1..240}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 1; done
  [ $up = 1 ] || { echo "SERVER FAILED $T"; exit 1; }
}
stop() { kill $SP 2>/dev/null; sleep 2; kill -9 $SP 2>/dev/null; wait $SP 2>/dev/null; }
trap 'stop' EXIT
# a. other lengths: tail 3,4,6,9,10,17,33 and controls (tail 64, 128), solo one pass or solo tail >= 256
serve len; python3 .scratch/exp3.py $PORT $PR $O/len.json 259,260,262,265,266,273,289,320,516,1028,1540,2564,1088,1412; stop
# b. threshold: q4 (772) behind a decoding row, last pass t rows (busy chunk 772 - t)
for t in 6 9 10 17 18 33 34; do serve t$t --busy-chunk $((772-t)); python3 .scratch/sweep.py $PORT $PR $O/t$t.json 4 behind; stop; done
# c. opt-in fix candidate: tail peel 16, q4 solo and behind, on one server
INTARSIA_PREFILL_PEEL=16 serve peel16; python3 .scratch/sweep.py $PORT $PR $O/peel16.json 3,4,5 solo,behind; stop
trap - EXIT
