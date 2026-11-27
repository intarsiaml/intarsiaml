#!/bin/bash
# endor EXCLUSIVE fleetq slot E, lane ink-metal-dflash2-ready: 2c-sampled (ABBA, sampled T 1.0, abba3.json, default vs dfl5) and 2d (seed stream
# 0..99 at T 1.0 / 0.6 on one engine with prefix reuse: DFlash2 k5 draft vs the own DSpark draft, SAMPLED_DFLASH=0). Information only.
set -u
: "${FLEETQ_JOB:?run under fleetq}"
. $(dirname $0)/lib.sh
O=$S/E; mkdir -p $O; T0=$(date +%s)
RT=$(ls -t $W/target/release/deps/inkling2_sampled_real-* | grep -v '\.d$' | head -1)
echo "slot E start $(date +%T) binary md5 $(md5 -q $B) test binary md5 $(md5 -q $RT) git $(git -C $W rev-parse --short HEAD)"
macmon pipe -i 250 > $O/macmon.jsonl 2>/dev/null &
echo "== 2c-sampled: ABBA, T 1.0, top-k 0, top-p 1, seed 7, 256 tokens, --ignore-eos, 45 s idle"
for tok in dspark@r1 dfl5@r1 dfl5@r2 dspark@r2; do
  arm=${tok%@*}; f=$O/abba-$tok; F=""; [ $arm = dfl5 ] && F=$(flags dfl5)
  sleep 45
  python3 -c 'import time; print(f"{time.time():.3f}")' > $f.t
  $B generate -m $M --backend metal --max-seq 20480 --profile-every 0 --show-ids --max-tokens 256 --ignore-eos --temperature 1.0 --top-k 0 --top-p 1.0 --seed 7 \
    --prompt-ids $S/prompts/abba3.json $F > $f.out 2> $f.err
  rc=$?; python3 -c 'import time; print(f"{time.time():.3f}")' >> $f.t
  echo "$tok rc $rc t+$(( $(date +%s) - T0 ))s | $(grep -aE '^decode\[' $f.err | tr '\n' ' ' | cut -c1-400) | $(grep -aE '^spec: verify' $f.err | cut -c1-80)"
done
echo "== 2d: seed stream, one engine, prefix reuse"
for x in X Y; do
  rm -f $O/seeds-$x.txt; DR=""; [ $x = X ] && DR="INTARSIA_SS_DRAFT=$D INTARSIA_SS_K=5"
  env INTARSIA_TEST_REAL_MODEL=1 INTARSIA_INK_MODEL=$M INTARSIA_INKLING2_BACKEND=metal INTARSIA_SPEC_SAMPLED_DFLASH=0 $DR \
    INTARSIA_SS_SEEDS=0..${SEEDS_N:-20} INTARSIA_SS_TEMPS=1.0,0.6 INTARSIA_SS_TOKENS=16 INTARSIA_SS_OUT=$O/seeds-$x.txt $RT --nocapture > $O/seeds-$x.log 2>&1
  echo "seeds $x rc $? t+$(( $(date +%s) - T0 ))s lines $(wc -l < $O/seeds-$x.txt) | $(grep -aE 'requests in|loaded in' $O/seeds-$x.log | tr '\n' '|' | cut -c1-300)"
done
cmp -s <(sort $O/seeds-X.txt) <(sort $O/seeds-Y.txt) && echo "seeds: X and Y identical" || echo "seeds: X AND Y DIFFER"
echo "slot E end $(date +%T) t+$(( $(date +%s) - T0 ))s"
