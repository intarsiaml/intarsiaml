#!/bin/zsh
# 35B on Metal: solo chunk 4096, busy chunk 2048, no peel. Exposure check only, no recipe change. Uses exp3.py (top-20 logprobs, solo vs behind).
set -u
[ -n "${FLEETQ_JOB:-}" ] || exit 2
cd /opt/code/intarsiaml/.worktrees/m27-flip
O=$PWD/.scratch/logs/gates; M=/Users/josef/models/Qwen3.6-35B-A3B-NVFP4; PORT=18479
PR=docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-16384.txt
INTARSIA_CB_TRACE=1 INTARSIA_CB_NO_REUSE=1 $PWD/.scratch/intarsia.main serve --backend metal --model $M --served-model-name intarsia --max-seq 12000 --max-batch 2 --spec off --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT > $O/35b2-server.out 2> $O/35b2-server.err < /dev/null &
SP=$!
trap 'kill $SP 2>/dev/null; sleep 2; kill -9 $SP 2>/dev/null' EXIT
up=0; for i in {1..240}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 1; done
[ $up = 1 ] || { echo "SERVER FAILED"; tail -3 $O/35b2-server.err; exit 1; }
BG_MAX=800 BG_SLEEP=1 python3 .scratch/exp3.py $PORT $PR $O/35b-len2.json 2050,2052,2056,2057,2058,2064,2100,3000
