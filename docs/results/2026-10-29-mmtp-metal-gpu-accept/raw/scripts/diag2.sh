#!/bin/bash
# MMTP diagnostic: served prefix reuse (warm B after A) vs cold B, per arm: mtp knob 1, mtp knob 0, spec off; main binary mtp.
set -u
[ -n "${FLEETQ_JOB:-}" ] || exit 2
W=/opt/code/intarsiaml/.worktrees/mmtp; O=$W/.lane/diag2; mkdir -p $O; PORT=18476
M=$HOME/models/Qwen3.6-35B-A3B-NVFP4; P=$W/.lane/prompts
NEW=$W/target/release/intarsia; MAIN=$W/.lane/main-target/release/intarsia
SUF=" Summarize the above in one paragraph."
up() { env X=1 "$@" > $O/srv-$tag.out 2> $O/srv-$tag.err & sp=$!; for i in $(seq 1 240); do curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null && break; kill -0 $sp 2>/dev/null || break; sleep 1; done; }
down() { kill -INT $sp; wait $sp 2>/dev/null; for i in $(seq 1 30); do lsof -iTCP:$PORT -sTCP:LISTEN -nP > /dev/null 2>&1 || break; sleep 1; done; }
S="serve --backend metal --model $M --served-model-name intarsia --max-seq 20000 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT"
arm() { # name bin spec env
  local name=$1 bin=$2 spec=$3 e=$4
  tag=$name-warm; up $e $bin $S --spec $spec
  python3 $W/.lane/an/req2.py $PORT $P/4ka.txt "" $O/$name-warm-A.txt
  python3 $W/.lane/an/req2.py $PORT $P/4ka.txt "$SUF" $O/$name-warm-B.txt
  down
  tag=$name-cold; up $e $bin $S --spec $spec
  python3 $W/.lane/an/req2.py $PORT $P/4ka.txt "$SUF" $O/$name-cold-B.txt
  down
  cmp -s $O/$name-warm-B.txt $O/$name-cold-B.txt && echo "$name: warm B == cold B" || echo "$name: warm B != cold B DIFFERS"
}
arm mtp1 $NEW mtp INTARSIA_METAL_MTP_GPU_ACCEPT=1
arm mtp0 $NEW mtp INTARSIA_METAL_MTP_GPU_ACCEPT=0
arm mainmtp $MAIN mtp X=1
arm off $NEW off X=1
for a in mtp0 mainmtp off; do cmp -s $O/mtp1-cold-B.txt $O/$a-cold-B.txt && echo "cold B mtp1 == $a" || echo "cold B mtp1 != $a DIFFERS"; done
