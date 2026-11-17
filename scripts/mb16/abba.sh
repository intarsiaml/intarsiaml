#!/bin/zsh
# Lane mb16 speed ABBA (27B, Metal), adapted from scripts/msmallm/abba.sh. Run under fleetq --quiet gpu,cpu.
# Arms: A = main c60d26caf (.base/target) `serve --max-batch 8`; B = this branch `serve --max-batch 16`. Fresh server per visit,
# INTARSIA_CB_TRACE=1 (server step ms). Per visit: warm-up, then cells
#   null4k: 1 stream x 2 requests of a ~4k-token prompt, 32 tokens (4k TTFT, prefill path)
#   s4 / s9 / s12 / s16: N closed-loop streams x 2 short requests (~100 tokens), 256 tokens each. s4 is the null cell (both arms
#   batch 4 rows on the same kernels).
# usage: [COOL=180] abba.sh <outdir> [visits, default "A1 B1 B2 A2"]
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mb16; O=$1; VISITS=${2:-"A1 B1 B2 A2"}; mkdir -p $O
M27=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4; TXT=$W/docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-16384.txt
PORT=18676; GEN=${GEN:-256}; SHORT_CHARS=${SHORT_CHARS:-430}; MAXSEQ=${MAXSEQ:-8192}
th() { notifyutil -g com.apple.system.thermalpressurelevel 2>&1 | awk '{print $2}'; }
# COOL=<s>: before every server start and every cell, wait up to <s> seconds for thermal pressure 0
cool() { local w=0; while [ $(th) != 0 ] && [ $w -lt ${COOL:-0} ]; do sleep 5; w=$((w+5)); done; [ ${COOL:-0} -gt 0 ] && echo "$(date +%s.%N) cooled ${w}s thermal $(th)" >> $1/cells.txt; }
start() { # dir binary extra-args...
  local D=$1 B=$2; shift 2; mkdir -p $D
  INTARSIA_CB_TRACE=1 $B serve --backend metal --model $M27 --served-model-name intarsia --max-seq $MAXSEQ --profile-every 0 --no-supervisor --no-ui \
    --host 127.0.0.1 --port $PORT "$@" > $D/server.out 2> $D/server.err < /dev/null &
  TP=$!
  for i in {1..240}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && return 0; kill -0 $TP 2>/dev/null || break; sleep 1; done
  echo "SERVER FAILED $D"; kill $TP 2>/dev/null; return 1
}
stop() { kill -INT $TP 2>/dev/null; for i in {1..60}; do kill -0 $TP 2>/dev/null || break; sleep 1; done; kill -9 $TP 2>/dev/null; wait $TP 2>/dev/null; }
cell() { # dir tag streams requests chars offset gen
  cool $1
  echo "$(date +%s.%N) start $2 thermal $(th)" >> $1/cells.txt
  python3 $W/scripts/msmallm/load.py --url http://127.0.0.1:$PORT/v1 --text $TXT --streams $3 --requests $4 --chars $5 --offset $6 --gen $7 \
    --tag "$V $2" --out $1/$2.json | tee -a $1/cells.txt
  echo "$(date +%s.%N) end $2 thermal $(th)" >> $1/cells.txt
}
for V in ${=VISITS}; do
  if [ ${V[1]} = A ]; then B=$W/.base/target/release/intarsia; MB=8; else B=$W/.base/bin-mb16/intarsia; MB=16; fi
  D=$O/$V-mb$MB; mkdir -p $D; cool $D; start $D $B --max-batch $MB || continue
  python3 $W/scripts/msmallm/load.py --url http://127.0.0.1:$PORT/v1 --text $TXT --streams 2 --requests 1 --chars 430 --offset 50000 --gen 16 --out $D/warm.json > /dev/null
  cell $D null4k 1 2 17800 20000 32
  for n in 4 9 12 16; do cell $D s$n $n 2 $SHORT_CHARS $((7919 + n*131)) $GEN; done
  stop
  gzip -f $D/server.err
done
echo "## done"
