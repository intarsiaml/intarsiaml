#!/bin/bash
# served A/B through localeval: ab.sh MODE (single | burst | eight)  -- run inside a fleetq slot. Order: M D D M M D.
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside fleetq slot" >&2; exit 2; }
MODE=$1
W=/opt/code/intarsiaml/.worktrees/mnq; O=$W/.lane/ab; PORT=18491
M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4; M8=$HOME/models/Qwen3-8B
binof() { echo $W/.lane/bin/intarsia-$1; }
arm() { # arm round
  local a=$1 r=$2 tag=$MODE-$1-r$2 model=$M35 mb="" sizes=1024,4096,16384 conc=1 gen=128 reps=3 maxseq=20000
  case $MODE in
    burst) mb="--max-batch 16"; sizes=1024; conc=1,4,8,16; gen=256; reps=1; maxseq=8192;;
    burst2) mb="--max-batch 16"; sizes=1024; conc=8,16,4; gen=256; reps=1; maxseq=8192;;
    eight) model=$M8; sizes=4096; reps=3;;
  esac
  local cmd="$(binof $a) serve --backend metal --model $model $mb --served-model-name intarsia --max-seq $maxseq --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT"
  echo "## $tag $(date +%T)"
  if [ "$a" = dev ]; then INTARSIA_METAL_NORM_FP8=1 $cmd > $O/$tag-server.out 2> $O/$tag-server.err & else $cmd > $O/$tag-server.out 2> $O/$tag-server.err & fi
  local sp=$!
  for i in $(seq 1 240); do curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null && break; kill -0 $sp 2>/dev/null || break; sleep 1; done
  (cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:$PORT/v1 --model intarsia --sizes $sizes --gen $gen --reps $reps \
     --concurrency $conc --thinking off --label "mnq-$tag" --recipe "$([ $a = dev ] && echo INTARSIA_METAL_NORM_FP8=1) $cmd" > $O/$tag-localeval.out 2> $O/$tag-localeval.err; echo "localeval exit $?")
  kill $sp; wait $sp 2>/dev/null
  for i in $(seq 1 30); do lsof -iTCP:$PORT -sTCP:LISTEN -nP > /dev/null 2>&1 || break; sleep 1; done
}
if [ "$MODE" = eight ]; then ORDER="main:1 dev:1 dev:2 main:2"; elif [ "$MODE" = burst2 ]; then ORDER="main:1 dev:1 dev:2 main:2 main:3 dev:3 dev:4 main:4 main:5 dev:5"; else ORDER="main:1 dev:1 dev:2 main:2 main:3 dev:3"; fi
for x in $ORDER; do arm ${x%%:*} ${x##*:}; done
echo "## done $(date +%T)"
