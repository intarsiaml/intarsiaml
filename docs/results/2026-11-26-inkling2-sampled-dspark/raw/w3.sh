#!/usr/bin/env bash
# Window 3 (lane ink-sampled), ON tatooine inside window.sh. (1) greedy md5 runs, card 0 main (spec off, recipe default), card 1 branch
# (same), 256 tokens over md5.json (6 short + 4k); (2) the real-model law harness, card 0 arm off (seeds 0..), card 1 arm on (shared
# seeds 0..SHARED first, then 1000400.. after W1a's 1000000..1000400), both with a generation budget.
set -uo pipefail
L=$HOME/intarsiaml-ink-sampled; T=$L/.scratch/tat; OUT=$L/logs/w3; mkdir -p $OUT
M=/models/Inkling-2-Mini
G="generate -m $M --backend hip --max-seq 8192 --profile-every 0 --show-ids --max-tokens 256 --prompt-ids /work/.scratch/prompts/md5.json"
now() { date '+%F %T'; }
md5() { # card binary
  local c=ink-sampled-run; [ $1 = 1 ] && c=ink-sampled-run2
  for sp in off def; do
    local f=$OUT/md5-$2-$sp extra=""
    [ $sp = off ] && extra="--spec off"
    HIP_VISIBLE_DEVICES=$1 INK_CONTAINER=$c bash $T/run.sh /work/.scratch/bin/intarsia-$2 $G $extra > $f.out 2> $f.err
    echo "[w3 c$1] md5 $2 $sp rc $? $(now) | $(grep -aE '^decode overall' $f.err) | $(grep -aE '^spec: verify' $f.err)"
  done
}
law() { # card arm seeds
  local c=ink-sampled-run; [ $1 = 1 ] && c=ink-sampled-run2
  rm -f $OUT/law-$2.txt
  HIP_VISIBLE_DEVICES=$1 INK_CONTAINER=$c bash $T/run.sh env INTARSIA_TEST_REAL_MODEL=1 INTARSIA_INK_MODEL=$M INTARSIA_INKLING2_BACKEND=hip \
    INTARSIA_SPEC_SAMPLED_DFLASH=$2 INTARSIA_SS_SEEDS=$3 INTARSIA_SS_TEMPS=1.0,0.6 INTARSIA_SS_TOKENS=6 INTARSIA_SS_BUDGET_S=${LAW_S:-130} \
    INTARSIA_SS_OUT=/work/logs/w3/law-$2.txt /work/.scratch/bin/t-${BR:-br2}-inkling2_sampled_real --nocapture > $OUT/law-$2.log 2>&1
  echo "[w3 c$1] law arm $2 rc $? $(now) lines $(wc -l < $OUT/law-$2.txt 2>/dev/null) | $(grep -aE 'requests in|budget|DFlash-family' $OUT/law-$2.log | tr '\n' ' ')"
}
echo "[w3] start $(now)"
# the inkling2 tiny suite, one test thread (its screened-head tests set INTARSIA_INK_HEAD_SCREEN process-wide; with parallel test
# threads other tests' loads race into the screen and refuse with NoKernel on hip, W1a / W2), branch then main, card 0 alone
for b in ${BR:-br2} main; do
  HIP_VISIBLE_DEVICES=0 bash $T/run.sh env INTARSIA_INKLING2_BACKEND=hip /work/.scratch/bin/t-$b-inkling2 --test-threads=1 > $OUT/tiny-$b.txt 2>&1
  echo "[w3] tiny $b: $(grep -aE 'test result' $OUT/tiny-$b.txt) | failed: $(grep -aE '^test .* FAILED' $OUT/tiny-$b.txt | awk '{print $2}' | tr '\n' ' ')"
done
( md5 0 main; law 0 0 "0..5000" ) > $OUT/c0.log 2>&1 &
( md5 1 ${BR:-br2}; law 1 1 "0..${SHARED:-100},1000400..1005000" ) > $OUT/c1.log 2>&1 &
wait
cat $OUT/c0.log $OUT/c1.log
echo "[w3] done $(now)"
