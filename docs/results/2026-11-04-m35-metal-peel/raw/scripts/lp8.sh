#!/bin/zsh
# Qwen3-8B on Metal: top-20 logprobs, solo vs behind (exp3.py of the m27 flip lane), main binary. Inside a fleetq slot.
set -u
[ -n "${FLEETQ_JOB:-}" ] || exit 2
cd /opt/code/intarsiaml/.worktrees/m35-peel
O=$PWD/.scratch/logs/lp8; mkdir -p $O; M=/Users/josef/models/Qwen3-8B; PORT=18486
PR=docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-16384.txt
INTARSIA_CB_TRACE=1 INTARSIA_CB_NO_REUSE=1 $PWD/.scratch/intarsia.main serve --backend metal --model $M --served-model-name intarsia --max-seq 12000 --max-batch 2 --spec off --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT > $O/server.out 2> $O/server.err < /dev/null &
SP=$!
trap 'kill $SP 2>/dev/null; sleep 2; kill -9 $SP 2>/dev/null' EXIT
up=0; for i in {1..240}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 1; done
[ $up = 1 ] || { echo "SERVER FAILED"; tail -3 $O/server.err; exit 1; }
BG_MAX=800 BG_SLEEP=1 python3 docs/results/2026-11-04-m27-batch-flip/raw/scripts/exp3.py $PORT $PR $O/lp.json 258,260,265,266,772,1028
