#!/bin/bash
# Exclusive endor window for lane pc (prefix store). Runs inside the fleetq reservation (FLEETQ_JOB set by the caller):
#   FLEETQ_JOB=<id> bash docs/results/2026-10-22-pc-prefix-store/scripts/metal-window.sh <step>
# Steps: noop (Qwen3-8B plan show + 256 greedy ids, base vs head, store off), ident (byte-identity test, 8B and 35B),
# dead (35B 4k generate, INTARSIA_PREFILL_SKIP_DEAD unset vs 1, x3 interleaved: ids and prefill ms),
# ttft (35B reuse workloads: record with the store off, then store on / off replays interleaved x3), all.
# Binaries are built beforehand (non-exclusive slot): head $W/target-metal/release/intarsia and the test binary,
# base $S/base/target/release/intarsia (main a92267b0).
set -u
W=/opt/code/wt-pc
S=/private/tmp/claude-501/-opt-code-intarsiaml/f31ccf91-6032-412f-aeab-eea38852047d/scratchpad/pc
R=$W/docs/results/2026-10-22-pc-prefix-store/raw
M8=/Users/josef/models/Qwen3-8B
M35=/Users/josef/models/Qwen3.6-35B-A3B-NVFP4
P=$W/docs/results/raw/2026-10-14-metal-phase-b
HEAD=$W/target-metal/release/intarsia
BASE=$S/base/target/release/intarsia
PORT=8197
mkdir -p $R
step=${1:-all}

noop() {
  O=$R/noop; mkdir -p $O
  for n in base head; do
    B=$BASE; [ $n = head ] && B=$HEAD
    $B plan show --model $M8 --backend metal > $O/$n-plan.out 2> $O/$n-plan.err; echo "$n plan rc=$?"
    for p in short 4k; do
      $B generate --model $M8 --backend metal --max-seq 8192 --temperature 0 --max-tokens 256 --ignore-eos --show-ids \
        --prompt "$(cat $P/prompt-$p.txt)" > $O/$n-gen-$p.out 2> $O/$n-gen-$p.err; echo "$n gen-$p rc=$?"
    done
  done
  cmp -s $O/base-plan.out $O/head-plan.out && echo "plan: identical" || { echo "plan: DIFFERS"; diff $O/base-plan.out $O/head-plan.out | head -20; }
  for p in short 4k; do
    b=$(grep '^ids:' $O/base-gen-$p.err | md5 -q); h=$(grep '^ids:' $O/head-gen-$p.err | md5 -q)
    [ "$b" = "$h" ] && echo "ids $p: identical (${b:0:8})" || echo "ids $p: DIFFER"
  done
}

ident() {
  O=$R/ident; mkdir -p $O
  T=$(ls -t $W/target-metal/release/deps/prefix_store_metal-* | grep -v '\.d$' | head -1)
  for m in $M8 $M35; do
    n=$(basename $m)
    INTARSIA_TEST_REAL_MODEL=1 INTARSIA_METAL_TEST_MODEL=$m $T --nocapture > $O/$n.out 2>&1; echo "ident $n rc=$?"
    grep PREFIX-STORE-METAL $O/$n.out
  done
}

dead() {
  O=$R/dead; mkdir -p $O
  for i in 1 2 3; do
    for v in 0 1; do
      INTARSIA_PREFILL_SKIP_DEAD=$v $HEAD generate --model $M35 --backend metal --max-seq 8192 --temperature 0 --max-tokens 64 \
        --ignore-eos --show-ids --profile-every 0 --prompt "$(cat $P/prompt-4k.txt)" > $O/skip$v-$i.out 2> $O/skip$v-$i.err
      echo "skip=$v rep $i rc=$? $(grep -i 'prefill' $O/skip$v-$i.err | tail -1)"
    done
  done
  for i in 1 2 3; do
    a=$(grep '^ids:' $O/skip0-$i.err | md5 -q); b=$(grep '^ids:' $O/skip1-$i.err | md5 -q)
    [ "$a" = "$b" ] && echo "dead rep $i ids identical (${a:0:8})" || echo "dead rep $i ids DIFFER"
  done
}

serve() { # $1 = on|off, $2 = log
  local extra=""
  [ "$1" = on ] && extra="--prefix-store on"
  $HEAD serve --model $M35 --backend metal --port $PORT --no-supervisor --max-seq 16384 --profile-every 0 --served-model-name m $extra \
    > $2 2>&1 < /dev/null &
  SPID=$!
  for i in $(seq 1 240); do
    curl -sf http://127.0.0.1:$PORT/v1/models > /dev/null 2>&1 && return 0
    sleep 1
  done
  echo "server did not come up"; return 1
}

stop() {
  kill $SPID 2>/dev/null; wait $SPID 2>/dev/null
}

ttft() {
  O=$R/ttft; mkdir -p $O
  REC=$O/recorded
  run() { # $1 label, $2 on|off, $3 record|replay
    serve $2 $O/$1-serve.log || return 1
    if [ "$3" = record ]; then
      # longer answers on the recording run: the history of the replays then holds real answers, not cut-off thinking
      $HEAD bench reuse --url http://127.0.0.1:$PORT --model m --corpus $W/tools/prompts/corpus_docs.txt --max-tokens 512 --record $REC --out $O/$1.json > $O/$1.txt 2>&1
    else
      $HEAD bench reuse --url http://127.0.0.1:$PORT --model m --corpus $W/tools/prompts/corpus_docs.txt --replay $REC --out $O/$1.json > $O/$1.txt 2>&1
    fi
    echo "$1 rc=$?"; cat $O/$1.txt
    curl -s http://127.0.0.1:$PORT/metrics | grep '^intarsia_prefix_store' > $O/$1.metrics
    stop
  }
  [ -d $REC ] || run rec-off off record
  for i in 1 2 3; do
    run on$i on replay
    run off$i off replay
  done
}

date +%T
case $step in
  noop) noop ;;
  ident) ident ;;
  dead) dead ;;
  ttft) ttft ;;
  all) noop; ident; ttft; dead ;;
esac
date +%T
