#!/bin/zsh
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
cd /opt/code/intarsiaml/.worktrees/m27-flip
BIN=$1; O=$PWD/.scratch/logs/sweep; mkdir -p $O
M=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4; PORT=18475
PR=docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-16384.txt
export INTARSIA_CB_TRACE=1 INTARSIA_CB_NO_REUSE=1
serve() { local T=$1 MB=$2; shift 2
  $BIN serve --backend metal --model $M --served-model-name intarsia --max-seq 12000 --max-batch $MB --spec off --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT "$@" > $O/$T-server.out 2> $O/$T-server.err < /dev/null &
  SP=$!
  local up=0; for i in {1..240}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 1; done
  [ $up = 1 ] || { echo "SERVER FAILED $T"; exit 1; }
}
stop() { kill $SP 2>/dev/null; sleep 2; kill -9 $SP 2>/dev/null; wait $SP 2>/dev/null; }
trap 'stop' EXIT
# all prompts, both schedules
serve all 2; python3 .scratch/sweep.py $PORT $PR $O/all.json 0,1,2,3,4,5,6,7,8,9 solo,behind; stop
# q4 at other busy chunks, behind a decoding row
for c in 32 64 128 512 771; do serve bc$c 2 --busy-chunk $c; python3 .scratch/sweep.py $PORT $PR $O/bc$c.json 4 behind; stop; done
# the contract's solo reference: q4 alone with --prefill-chunk 256; and the single-sequence engine at its defaults
serve pc256 2 --prefill-chunk 256; python3 .scratch/sweep.py $PORT $PR $O/pc256.json 4 solo; stop
serve mb1 1; python3 .scratch/sweep.py $PORT $PR $O/mb1.json 4 solo; stop
trap - EXIT
