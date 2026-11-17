#!/bin/zsh
# Lane msm192 TTFT ABBA (27B, Metal, default engine: MTP, max-batch 1). Run under fleetq --quiet gpu,cpu, its own slot.
# Arms: A = target/base/intarsia (origin/main c60d26caf: M = 9..128 on the in-kernel form), B = target/release/intarsia (this branch: per-shape limit up to 192).
# Both with the knob unset. Per visit (fresh server): warm-up, then cells of 8 distinct prompts, 1 stream, 16 tokens, TTFT per request:
#   t100 (null: M <= 128 on both), t140 / t170 / t190 (M = 129..192 rows), null4k (2 requests of about 4k tokens: only the remainder is in the window).
# usage: [COOL=180] abba.sh <outdir> [visits, default "A1 B1 B2 A2"]
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/msm192; O=$1; VISITS=${2:-"A1 B1 B2 A2"}; mkdir -p $O
M27=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4; TXT=$W/docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-16384.txt; PJ=$W/scripts/msm192/prompts.json
PORT=18675; GEN=${GEN:-16}; REQS=${REQS:-8}
th() { notifyutil -g com.apple.system.thermalpressurelevel 2>&1 | awk '{print $2}'; }
cool() { local w=0; while [ $(th) != 0 ] && [ $w -lt ${COOL:-0} ]; do sleep 5; w=$((w+5)); done; [ ${COOL:-0} -gt 0 ] && echo "$(date +%s.%N) cooled ${w}s thermal $(th)" >> $1/cells.txt; }
start() { # dir binary
  local D=$1 BIN=$2; mkdir -p $D
  env X=1 $BIN serve --backend metal --model $M27 --served-model-name intarsia --max-seq 40000 --profile-every 0 --no-supervisor --no-ui \
    --host 127.0.0.1 --port $PORT > $D/server.out 2> $D/server.err < /dev/null &
  TP=$!
  for i in {1..240}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && return 0; kill -0 $TP 2>/dev/null || break; sleep 1; done
  echo "SERVER FAILED $D"; kill $TP 2>/dev/null; return 1
}
stop() { kill -INT $TP 2>/dev/null; for i in {1..60}; do kill -0 $TP 2>/dev/null || break; sleep 1; done; kill -9 $TP 2>/dev/null; wait $TP 2>/dev/null; }
cell() { # dir tag key
  cool $1
  echo "$(date +%s.%N) start $2 thermal $(th)" >> $1/cells.txt
  python3 $W/scripts/msm192/load.py --url http://127.0.0.1:$PORT/v1 --text $PJ --key $3 --streams 1 --requests $REQS --gen $GEN --tag "$V $2" --out $1/$2.json | tee -a $1/cells.txt
  echo "$(date +%s.%N) end $2 thermal $(th)" >> $1/cells.txt
}
for V in ${=VISITS}; do
  BIN=$W/target/base/intarsia; [ ${V[1]} = B ] && BIN=$W/target/release/intarsia
  D=$O/$V; mkdir -p $D; cool $D; start $D $BIN || continue
  python3 $W/scripts/msm192/load.py --url http://127.0.0.1:$PORT/v1 --text $TXT --chars 600 --offset 50000 --streams 1 --requests 2 --gen 16 --out $D/warm.json > /dev/null  # about 160 tokens, a slice no cell uses
  cell $D t100 100; cell $D t140 140; cell $D t170 170; cell $D t190 190
  cool $D; echo "$(date +%s.%N) start null4k thermal $(th)" >> $D/cells.txt
  python3 $W/scripts/msmallm/load.py --url http://127.0.0.1:$PORT/v1 --text $TXT --streams 1 --requests 3 --chars 17800 --offset 20000 --gen $GEN --tag "$V null4k" --out $D/null4k.json | tee -a $D/cells.txt
  stop; gzip -f $D/server.err
done
echo "## done"
