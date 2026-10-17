#!/bin/zsh
# relcheck: (a) the B=8 gate scenario exactly as slot_perf.sh (a) of 2026-10-16-metal-cb-m2-closeout (7 streams + 4k at t=5 s, 5 reps,
# 3 solo reps, same limits), (b) the shedding bursts exactly as slot_perf.sh (c) (--max-batch 4 --max-queue 4, traced).
#   usage: slot_b8.sh OUTDIR        (inside a fleetq slot)
source ${0:A:h}/env.sh
O=$1
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
export PORT=8143 INTARSIA_BIN=$HEAD
SERVE_EXTRA="--max-batch 8" $W/tools/cbsoak_serve.sh $O/scenario scenario --streams 7 --reps 5 --solo-reps 3 --max-itl-prefill-p95-ms 150 --max-steady-itl-p95-ms 60 --max-long-ttft-x-solo 2.5
RC=$?
INTARSIA_CB_TRACE=1 SERVE_EXTRA="--max-batch 4 --max-queue 4" $W/tools/cbsoak_serve.sh $O/burst burst --n 16 --prompt-tokens 512 --max-tokens 300 --seed 1 \
  --then burst --n 24 --prompt-tokens 1024 --max-tokens 200 --seed 2
[ $? -gt $RC ] && RC=$?
exit $RC
