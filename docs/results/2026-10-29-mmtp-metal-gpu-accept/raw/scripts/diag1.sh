#!/bin/bash
# MMTP diagnostic: does the served engine take the pipelined step? (spec trace, one 4k request with and without min_tokens)
set -u
[ -n "${FLEETQ_JOB:-}" ] || exit 2
W=/opt/code/intarsiaml/.worktrees/mmtp; O=$W/.lane/diag1; mkdir -p $O; PORT=18475
M=$HOME/models/Qwen3.6-35B-A3B-NVFP4
for v in 1 0; do
  env INTARSIA_SPEC_TRACE=1 INTARSIA_METAL_MTP_GPU_ACCEPT=$v $W/target/release/intarsia serve --backend metal --model $M --served-model-name intarsia --max-seq 20000 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT --spec mtp > $O/srv$v.out 2> $O/srv$v.err &
  sp=$!
  for i in $(seq 1 240); do curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null && break; sleep 1; done
  for mt in 256 0; do python3 $W/.lane/an/req.py $PORT $W/.lane/prompts/4ka.txt $mt; done
  kill -INT $sp; wait $sp 2>/dev/null; sleep 2
done
