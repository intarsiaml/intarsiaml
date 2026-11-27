#!/usr/bin/env bash
# Lane ink-sampled-adapt, HIP window W1, ON tatooine inside window.sh (CAP <= 285 s). Both cards at once.
# Phase 1: greedy md5, 64 tokens, md5.json (6 short + 4k): card 0 main6 (main aee06201c), card 1 ad4 (this branch, merged); --spec off and the recipe
#   default (DSpark k5); no INTARSIA_* variable set.
# Phase 2 (card 0 T 1.0, card 1 T 0.7; seed 11, speed.json, 128 tokens, INTARSIA_SPEC_SAMPLED_DFLASH=1, INTARSIA_SPEC_TRACE=1):
#   calibration fixed k5 (INTARSIA_SPEC_SAMPLED_ADAPT=0), the `drafts` diagnostic, then the policy (=1: checks the offline replay).
# Phase 3: seeded equality, real-model harness at k5, 24 tokens, seeds 0..100 x T 1.0 / 0.6: card 0 drafting off, card 1 the policy
#   (default table, traced); then card 1 the hip tiny suite of this lane.
set -uo pipefail
L=$HOME/intarsiaml-ink-sampled-adapt; T=$L/.scratch/tat; OUT=$L/logs/w1; mkdir -p $OUT/cal
M=/models/Inkling-2-Mini
now() { date '+%F %T'; }
echo "[w1] $(md5sum $L/.scratch/bin/intarsia-ad4 $L/.scratch/bin/intarsia-main6 | tr '\n' ' ')"
G="generate -m $M --backend hip --max-seq 8192 --profile-every 0 --show-ids"
cont() { [ $1 = 1 ] && echo ink-adapt-run2 || echo ink-adapt-run; }
md5() { # card binary
  for sp in off def; do
    local f=$OUT/md5-$2-$sp extra=""
    [ $sp = off ] && extra="--spec off"
    HIP_VISIBLE_DEVICES=$1 INK_CONTAINER=$(cont $1) bash $T/run.sh env /work/.scratch/bin/intarsia-$2 $G --max-tokens 64 --prompt-ids /work/.scratch/prompts/md5.json $extra > $f.out 2> $f.err
    echo "[w1 c$1] md5 $2 $sp rc $? $(now) | $(grep -aE '^decode overall' $f.err) | $(grep -aE '^spec: verify' $f.err)"
  done
}
echo "[w1] md5 phase $(now)"
( md5 0 main6 ) > $OUT/m0.log 2>&1 &
( md5 1 ad4 ) > $OUT/m1.log 2>&1 &
wait
cat $OUT/m0.log $OUT/m1.log
SP="--max-tokens 128 --ignore-eos --prompt-ids /work/.scratch/prompts/speed.json --top-k 0 --top-p 1.0 --seed 11"
cal() { # card temp mode
  local f=$OUT/cal/$3-t$2
  HIP_VISIBLE_DEVICES=$1 INK_CONTAINER=$(cont $1) bash $T/run.sh env INTARSIA_SPEC_SAMPLED_DFLASH=1 INTARSIA_SPEC_SAMPLED_ADAPT=$3 INTARSIA_SPEC_TRACE=1 \
    /work/.scratch/bin/intarsia-ad4 $G $SP --temperature $2 > $f.out 2> $f.err
  echo "[w1 c$1] cal $3 T $2 rc $? $(now) | $(grep -aE '^decode overall' $f.err) | $(grep -aE '^spec: verify' $f.err)"
}
seeds() { # card arm(0|1)
  local f=$OUT/seeds-$2 tr=""
  rm -f $f.txt
  [ $2 = 1 ] && tr="INTARSIA_SPEC_TRACE=1"
  HIP_VISIBLE_DEVICES=$1 INK_CONTAINER=$(cont $1) bash $T/run.sh env INTARSIA_TEST_REAL_MODEL=1 INTARSIA_INK_MODEL=$M INTARSIA_INKLING2_BACKEND=hip \
    INTARSIA_SPEC_SAMPLED_DFLASH=$2 $tr INTARSIA_SS_SEEDS=0..100 INTARSIA_SS_TEMPS=1.0,0.6 INTARSIA_SS_TOKENS=24 INTARSIA_SS_K=5 \
    INTARSIA_SS_BUDGET_S=${SEEDS_BUDGET:-90} INTARSIA_SS_OUT=/work/logs/w1/seeds-$2.txt /work/.scratch/bin/t-ad4-inkling2_sampled_real --nocapture > $f.log 2>&1
  echo "[w1 c$1] seeds arm $2 rc $? $(now) lines $(wc -l < $f.txt) | $(grep -ac '^adapt: token' $f.log) phase changes ($(grep -ac 'Stop {' $f.log) stops, $(grep -ac 'Resume {' $f.log) resumes) | $(grep -aE 'requests in' $f.log)"
}
tiny() { # card
  HIP_VISIBLE_DEVICES=$1 INK_CONTAINER=$(cont $1) bash $T/run.sh env INTARSIA_INKLING2_BACKEND=hip /work/.scratch/bin/t-ad4-inkling2_dspark_sampled_adapt --test-threads=1 > $OUT/tiny-adapt-hip.txt 2>&1
  echo "[w1 c$1] tiny inkling2_dspark_sampled_adapt (hip) rc $? $(now) | $(grep -aE '^test result' $OUT/tiny-adapt-hip.txt)"
}
echo "[w1] calibration + seeds $(now)"
( cal 0 1.0 0; cal 0 1.0 drafts; cal 0 1.0 1; seeds 0 0 ) > $OUT/c0.log 2>&1 &
( cal 1 0.7 0; cal 1 0.7 drafts; cal 1 0.7 1; seeds 1 1; tiny 1 ) > $OUT/c1.log 2>&1 &
wait
cat $OUT/c0.log $OUT/c1.log
if [ -s $OUT/seeds-0.txt ] && [ -s $OUT/seeds-1.txt ]; then
  n=$(comm -12 <(sort $OUT/seeds-0.txt) <(sort $OUT/seeds-1.txt) | wc -l)
  k0=$(cut -d' ' -f1,2 $OUT/seeds-0.txt | sort); k1=$(cut -d' ' -f1,2 $OUT/seeds-1.txt | sort)
  shared=$(comm -12 <(echo "$k0") <(echo "$k1") | wc -l)
  echo "[w1] seeds: $shared shared (T, seed) pairs, $n identical lines"
  for t in 1.0 0.6; do
    a=$(awk -v t=$t '$1==t' $OUT/seeds-0.txt | sort); b=$(awk -v t=$t '$1==t' $OUT/seeds-1.txt | sort)
    echo "[w1] seeds T $t: off $(echo "$a" | grep -c .) on $(echo "$b" | grep -c .) shared $(comm -12 <(echo "$a" | cut -d' ' -f1,2) <(echo "$b" | cut -d' ' -f1,2) | grep -c .) identical $(comm -12 <(echo "$a") <(echo "$b") | grep -c .)"
  done
fi
echo "[w1] done $(now)"
