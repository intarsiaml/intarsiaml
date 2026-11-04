#!/bin/zsh
# attribution: with the NAX form off every MXFP4 matmul is the GEMV, so if the GEMV / NAX split is the cause, solo and behind must be bit-equal
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
cd /opt/code/intarsiaml/.worktrees/m27-flip
BIN=$PWD/.scratch/intarsia.head; O=$PWD/.scratch/logs/exp4; mkdir -p $O
M=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4; PORT=18475
PR=docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-16384.txt
export INTARSIA_CB_TRACE=1 INTARSIA_CB_NO_REUSE=1 INTARSIA_METAL_MXFP4_NAX=0
$BIN serve --backend metal --model $M --served-model-name intarsia --max-seq 12000 --max-batch 2 --spec off --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT > $O/nax0-server.out 2> $O/nax0-server.err < /dev/null &
SP=$!
trap 'kill $SP 2>/dev/null; sleep 2; kill -9 $SP 2>/dev/null' EXIT
up=0; for i in {1..240}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 1; done
[ $up = 1 ] || { echo "SERVER FAILED"; exit 1; }
python3 .scratch/sweep.py $PORT $PR $O/nax0.json 4 solo,behind
python3 .scratch/exp3.py $PORT $PR $O/nax0-len.json 260,1028
kill $SP 2>/dev/null; sleep 2; kill -9 $SP 2>/dev/null; wait $SP 2>/dev/null
# peel 16 and peel 8 on other tail lengths, including the 1-row-avoidance case plen mod 256 == 1 (busy 255 + 2)
unset INTARSIA_METAL_MXFP4_NAX
for p in 16 8; do
INTARSIA_PREFILL_PEEL=$p $BIN serve --backend metal --model $M --served-model-name intarsia --max-seq 12000 --max-batch 2 --spec off --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT > $O/peel$p-server.out 2> $O/peel$p-server.err < /dev/null &
SP=$!
up=0; for i in {1..240}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 1; done
[ $up = 1 ] || { echo "SERVER FAILED"; exit 1; }
python3 .scratch/exp3.py $PORT $PR $O/peel$p-len.json 257,258,259,260,265,266,513,1028
kill $SP 2>/dev/null; sleep 2; kill -9 $SP 2>/dev/null; wait $SP 2>/dev/null
done
# no peel, plen mod 256 == 1 and 2, for comparison
$BIN serve --backend metal --model $M --served-model-name intarsia --max-seq 12000 --max-batch 2 --spec off --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT > $O/nopeel-server.out 2> $O/nopeel-server.err < /dev/null &
SP=$!
up=0; for i in {1..240}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { up=1; break; }; kill -0 $SP 2>/dev/null || break; sleep 1; done
[ $up = 1 ] || { echo "SERVER FAILED"; exit 1; }
python3 .scratch/exp3.py $PORT $PR $O/nopeel-len.json 257,258,513
