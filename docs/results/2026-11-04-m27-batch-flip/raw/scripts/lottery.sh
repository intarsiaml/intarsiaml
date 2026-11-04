#!/bin/zsh
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
cd /opt/code/intarsiaml/.worktrees/m27-flip
BIN=$PWD/.scratch/intarsia.head; O=$PWD/.scratch/logs/lottery; mkdir -p $O
M=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4; PORT=18475
PR=docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-16384.txt
export INTARSIA_CB_TRACE=1
$BIN serve --backend metal --model $M --served-model-name intarsia --max-seq 12000 --max-batch 2 --spec off --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT > $O/server.out 2> $O/server.err < /dev/null &
SP=$!
trap 'kill $SP 2>/dev/null; sleep 2; kill -9 $SP 2>/dev/null' EXIT
up=0; for i in {1..240}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 1; done
[ $up = 1 ] || { echo "SERVER FAILED"; exit 1; }
python3 .scratch/lottery.py $PORT $PR $O/ids.json 24
