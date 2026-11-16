#!/bin/zsh
# Lane msmallm speed ABBA (27B, Metal). Run under fleetq --quiet gpu,cpu. Arms: A = INTARSIA_METAL_MXFP4_NAXD=0 (the scratch path,
# today's main), B = knob unset (M = 9..128 on the in-kernel decode form); same binary, fresh server per visit.
# Per visit: server 1 = the default engine (MTP, max-batch 1): 1 stream x 4 short requests, then the null cell (4k prompts, 2 requests, TTFT);
#            server 2 = --max-batch 8 (Metal's cap) with INTARSIA_CB_TRACE=1: 4 / 9 / 16 streams x 2 short requests.
# usage: abba.sh <outdir> [visits, default "A1 B1 B2 A2"]
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/msmallm; BIN=$W/target/release/intarsia; O=$1; VISITS=${2:-"A1 B1 B2 A2"}; mkdir -p $O
M27=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4; TXT=$W/docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-16384.txt
PORT=18673; GEN=${GEN:-256}; SHORT_CHARS=${SHORT_CHARS:-430}
th() { notifyutil -g com.apple.system.thermalpressurelevel 2>&1 | awk '{print $2}'; }
start() { # dir env-words extra-args...
  local D=$1 EV=$2; shift 2; mkdir -p $D
  env ${=EV} $BIN serve --backend metal --model $M27 --served-model-name intarsia --max-seq 40000 --profile-every 0 --no-supervisor --no-ui \
    --host 127.0.0.1 --port $PORT "$@" > $D/server.out 2> $D/server.err < /dev/null &
  TP=$!
  for i in {1..240}; do curl -sf -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && return 0; kill -0 $TP 2>/dev/null || break; sleep 1; done
  echo "SERVER FAILED $D"; kill $TP 2>/dev/null; return 1
}
stop() { kill -INT $TP 2>/dev/null; for i in {1..60}; do kill -0 $TP 2>/dev/null || break; sleep 1; done; kill -9 $TP 2>/dev/null; wait $TP 2>/dev/null; }
cell() { # dir tag streams requests chars offset gen
  echo "$(date +%s.%N) start $2 thermal $(th)" >> $1/cells.txt
  python3 $W/scripts/msmallm/load.py --url http://127.0.0.1:$PORT/v1 --text $TXT --streams $3 --requests $4 --chars $5 --offset $6 --gen $7 \
    --tag "$V $2" --out $1/$2.json | tee -a $1/cells.txt
  echo "$(date +%s.%N) end $2 thermal $(th)" >> $1/cells.txt
}
vi=0
for V in ${=VISITS}; do
  vi=$((vi+1)); EV="X=1"; [ ${V[1]} = A ] && EV="INTARSIA_METAL_MXFP4_NAXD=0"
  D=$O/$V-default; start $D "$EV" || continue
  python3 $W/scripts/msmallm/load.py --url http://127.0.0.1:$PORT/v1 --text $TXT --streams 1 --requests 1 --chars 430 --offset 50000 --gen 16 --out $D/warm.json > /dev/null
  cell $D s1 1 4 $SHORT_CHARS 7919 $GEN
  cell $D null4k 1 2 17800 20000 32
  stop
  D=$O/$V-mb8; start $D "$EV INTARSIA_CB_TRACE=1" --max-batch 8 || continue
  python3 $W/scripts/msmallm/load.py --url http://127.0.0.1:$PORT/v1 --text $TXT --streams 2 --requests 1 --chars 430 --offset 50000 --gen 16 --out $D/warm.json > /dev/null
  for n in 4 9 16; do cell $D s$n $n 2 $SHORT_CHARS $((7919 + n*131)) $GEN; done
  stop
  gzip -f $D/server.err
done
echo "## done"
