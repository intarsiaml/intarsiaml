#!/bin/zsh
# M2 closeout: (a) the gate scenario at B=8 on the closeout binary (7 streams at ~1k + a 4k prompt at t=5 s, 5 reps, chunk 256,
# no trace), (b) a 10 min Poisson run at 0.8 req/s (near saturation, so a queue forms) with 15% cancels and INTARSIA_CB_TRACE=1
# for tools/cbfifo.py, (c) the shedding bursts on --max-batch 4 --max-queue 4 with trace, to read the live queue gauges.
#   usage: slot_perf.sh OUTDIR        env: INTARSIA_BIN (required)
W=/opt/code/intarsiaml-m2close
O=$1
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a slot" >&2; exit 2; }
export PORT=8143
SERVE_EXTRA="--max-batch 8" $W/tools/cbsoak_serve.sh $O/scenario scenario --streams 7 --reps 5 --solo-reps 3 --max-itl-prefill-p95-ms 150 --max-steady-itl-p95-ms 60 --max-long-ttft-x-solo 2.5
RC=$?
INTARSIA_CB_TRACE=1 SERVE_EXTRA="--max-batch 8" $W/tools/cbsoak_serve.sh $O/fifo-soak --rate 0.8 --duration 600 --seed 31 --cancel-frac 0.15 --drain-s 180 --sample-s 5 --window-s 60
[ $? -gt $RC ] && RC=$?
INTARSIA_CB_TRACE=1 SERVE_EXTRA="--max-batch 4 --max-queue 4" $W/tools/cbsoak_serve.sh $O/burst burst --n 16 --prompt-tokens 512 --max-tokens 300 --seed 1 \
  --then burst --n 24 --prompt-tokens 1024 --max-tokens 200 --seed 2
[ $? -gt $RC ] && RC=$?
for d in fifo-soak burst; do python3 $W/tools/cbfifo.py check $O/$d/server.log --allow-open --json $O/$d/fifo.json; done
exit $RC
