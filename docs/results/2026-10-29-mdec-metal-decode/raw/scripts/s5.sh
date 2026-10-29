#!/bin/bash
# short check 5: served 35B N=4, --max-batch 8 vs 16, 2 interleaved pairs
set -u
[ -n "${FLEETQ_JOB:-}" ] || exit 2
W=/opt/code/intarsiaml/.worktrees/mdec; O=$W/.lane/s5; mkdir -p $O; BIN=$W/.lane/target/release/intarsia; M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4; PORT=18475
srv() { local tag=$1 mb=$2
  local cmd="$BIN serve --backend metal --model $M35 --max-batch $mb --served-model-name intarsia --max-seq 8192 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT"
  $cmd > $O/$tag-server.out 2> $O/$tag-server.err & local sp=$!
  for i in $(seq 1 240); do curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null && break; kill -0 $sp 2>/dev/null || break; sleep 1; done
  (cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:$PORT/v1 --model intarsia --sizes 1024 --gen 512 --reps 1 \
     --concurrency 4 --thinking off --label "mdec-s5-$tag" --recipe "$cmd" > $O/$tag-localeval.out 2> $O/$tag-localeval.err)
  echo "$tag $(grep -E '^\| 4 \|' $O/$tag-localeval.out)"
  kill $sp; wait $sp 2>/dev/null
  for i in $(seq 1 30); do lsof -iTCP:$PORT -sTCP:LISTEN -nP > /dev/null 2>&1 || break; sleep 1; done
}
srv a1-mb8 8; srv a1-mb16 16; srv a2-mb16 16; srv a2-mb8 8
