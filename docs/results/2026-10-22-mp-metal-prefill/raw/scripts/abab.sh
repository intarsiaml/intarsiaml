#!/bin/zsh
# lane MP: interleaved A/B of two intarsia binaries (base/cand, order ABBAAB) in ONE exclusive endor slot, localeval `speed`
# 128 / 1k / 4k / 16k, gen 256, 3 reps, macmon logged per arm, then gate.py (docs/benchmarking.md "Metal speed gate").
#   fleetq run --host endor --exclusive --mem 24 --label "metal speed gate" -- \
#     zsh scripts/metal-speed-gate/sameslot.sh <intarsia binary built with --features metal> <out dir>
# Each arm is bracketed by refbw.py (clock-normalisation reference, baseline in ref-baseline.json); gate.py divides decode by
# the reference/baseline ratio and prints raw, reference and normalised values.
# Server flags as docs/results/2026-10-18-metal-sameslot-main.md, plus --max-seq 20000 / llama.cpp -c 20480 for 16k.
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
BASE=${1:?base binary}; CAND=${2:?cand binary}; OUT=${3:?out dir}
H=/opt/code/wt-mp/scripts/metal-speed-gate
BIN=$CAND
MODEL=${MODEL:-/Users/josef/models/Qwen3-8B}
GGUF=${GGUF:-/Users/josef/models/Qwen3-8B-GGUF/Qwen3-8B-BF16.gguf}
SIZES=${SIZES:-1024,4096,16384}
MACMON=${MACMON:-/Users/josef/bin/macmon}
export INTARSIA_CPU_FREE_MEM_GB=${INTARSIA_CPU_FREE_MEM_GB:-60}
mkdir -p $OUT
SPID=""; MPID=""
trap '[[ -n "$SPID" ]] && kill $SPID 2>/dev/null; [[ -n "$MPID" ]] && kill $MPID 2>/dev/null' EXIT
{ echo "date: $(date -Iseconds)  fleetq job: $FLEETQ_JOB"; echo "base md5: $(md5 -q $BASE) cand md5: $(md5 -q $CAND) cand env: ${CAND_ENV:-none}"; sw_vers | tr '\n' ' '; echo
  echo "top cpu:"; top -l 1 -o cpu -n 6 -stats pid,command,cpu | tail -7; } > $OUT/versions.txt 2>&1

run_engine() { # label port health-path modelid cmd...
  local label=$1 port=$2 hp=$3 mid=$4; shift 4
  echo "##### $label $(date -Iseconds)"
  "$@" > $OUT/$label-server.out 2> $OUT/$label-server.err &
  SPID=$!
  local up=0
  for i in {1..240}; do
    curl -sf http://127.0.0.1:$port$hp >/dev/null 2>&1 && { up=1; break; }
    kill -0 $SPID 2>/dev/null || break
    sleep 1
  done
  [[ $up == 1 ]] || { echo "SERVER FAILED $label"; SPID=""; return 1; }
  REF=$H/refbw.py   # fixed reference workload (clock normalisation), run before and after the arm's localeval
  $REF >> $OUT/$label-ref.jsonl 2>> $OUT/$label-ref.err
  $MACMON pipe -i 250 > $OUT/$label-macmon.jsonl 2>/dev/null & MPID=$!
  ( cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:$port/v1 --model $mid --sizes $SIZES --gen 256 --reps 3 \
      --thinking off --label gate-$label ) > $OUT/$label-localeval.out 2> $OUT/$label-localeval.err
  echo "localeval exit $?"
  $REF >> $OUT/$label-ref.jsonl 2>> $OUT/$label-ref.err
  kill $MPID 2>/dev/null; wait $MPID 2>/dev/null; MPID=""
  local res=$(grep -o 'saved .*result.json' $OUT/$label-localeval.out | tail -1 | sed 's/^saved //')
  [ -n "$res" ] && cp $res $OUT/$label-localeval.json
  kill $SPID 2>/dev/null
  for i in {1..30}; do kill -0 $SPID 2>/dev/null || break; sleep 1; done
  kill -0 $SPID 2>/dev/null && kill -9 $SPID
  wait $SPID 2>/dev/null; SPID=""
  for i in {1..30}; do lsof -iTCP:$port -sTCP:LISTEN -nP >/dev/null 2>&1 || break; sleep 1; done
  sleep 5
}

A=($BASE serve --model $MODEL --backend metal --device 0 --host 127.0.0.1 --port 8093 --served-model-name intarsia-metal --max-seq 20000 --profile-every 0 --no-supervisor --no-ui)
B=(env ${CAND_ENV:-INTARSIA_AB_ARM=cand} $CAND serve --model $MODEL --backend metal --device 0 --host 127.0.0.1 --port 8093 --served-model-name intarsia-metal --max-seq 20000 --profile-every 0 --no-supervisor --no-ui)
for arm in ${=ORDER:-base-1 cand-1 cand-2 base-2 base-3 cand-3}; do
  case $arm in base-*) run_engine $arm 8093 /health intarsia-metal $A ;; cand-*) run_engine $arm 8093 /health intarsia-metal $B ;; esac
done
