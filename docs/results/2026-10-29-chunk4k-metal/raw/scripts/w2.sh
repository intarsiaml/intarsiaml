#!/bin/bash
# chunk4k window 2: served A/B, recipe chunk 4096 (no flag) vs --prefill-chunk 2048, one binary, 3 rounds, arm order rotated.
# single stream: default serve (no --max-batch), sizes 1024,4096,16384 gen 256 reps 3. burst: --max-batch 16, 1024-token prompts, gen 512, 4/8/16 streams (+ the 1-stream row).
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/chunk4k; cd $W; O=$W/.lane/w2; mkdir -p $O
B=$W/.lane/intarsia-metal; M=$HOME/Models/Qwen3.6-35B-A3B-NVFP4; PORT=18491; T0=$SECONDS
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $O/job.log; }
srv() { # tag mode(single|burst) extra-serve-args
  local tag=$1 mode=$2 extra=$3 sizes gen conc ms mb
  if [ $mode = single ]; then sizes=1024,4096,16384; gen=256; reps=3; conc=""; ms=20480; mb=""
  else sizes=1024; gen=512; reps=1; conc="--concurrency 4,8,16"; ms=8192; mb="--max-batch 16"; fi
  local cmd="$B serve --backend metal --model $M $mb --served-model-name intarsia --max-seq $ms --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT $extra"
  $cmd > $O/$tag-server.out 2> $O/$tag-server.err & local sp=$!
  for i in $(seq 1 240); do curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null && break; kill -0 $sp 2>/dev/null || break; sleep 1; done
  grep -a "prefill chunk" $O/$tag-server.err | head -1 | tee -a $O/job.log
  (cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:$PORT/v1 --model intarsia --sizes $sizes --gen $gen --reps $reps $conc \
     --thinking off --label "chunk4k-$tag" --recipe "$cmd" > $O/$tag-localeval.out 2> $O/$tag-localeval.err)
  log "$tag localeval exit $? ($((SECONDS - T0)) s)"
  kill $sp; wait $sp 2>/dev/null
  for i in $(seq 1 30); do lsof -iTCP:$PORT -sTCP:LISTEN -nP > /dev/null 2>&1 || break; sleep 1; done
}
for r in ${ROUNDS:-1 2 3}; do
  if [ $((r % 2)) = 1 ]; then order="c4096 c2048"; else order="c2048 c4096"; fi
  for a in $order; do
    case $a in c4096) X="";; c2048) X="--prefill-chunk 2048";; esac
    srv r$r-$a-single single "$X"
    [ "${NOBURST:-0}" = 1 ] || srv r$r-$a-burst burst "$X"
  done
done
log "done ($((SECONDS - T0)) s)"
