#!/usr/bin/env bash
# Window 1 (lane ink-sampled), ON tatooine inside window.sh. Card 0: the tiny Inkling-2 suites on hip (branch: the four + the lane's
# inkling2_dspark_sampled; main: inkling2), card 1: the real-model law / seed-equality harness, arm on then arm off.
set -uo pipefail
L=$HOME/intarsiaml-ink-sampled; T=$L/.scratch/tat; OUT=$L/logs/w1; mkdir -p $OUT; rm -f $OUT/tiny-br.txt
N=${LAW_N:-400}; NS=${LAW_SHARED:-150}
now() { date '+%F %T'; }
card0() {
  for t in inkling2 inkling2_dspark inkling2_dspark_cut inkling2_verify_softcap inkling2_dspark_sampled; do
    echo "[w1 c0] tiny branch $t $(now)"
    HIP_VISIBLE_DEVICES=0 bash $T/run.sh env INTARSIA_INKLING2_BACKEND=hip /work/.scratch/bin/t-${BR:-br1}-$t 2>&1 \
      | grep -aE '^test |test result|panicked|random draft|scripted' >> $OUT/tiny-br.txt
  done
  echo "[w1 c0] tiny main inkling2 $(now)"
  HIP_VISIBLE_DEVICES=0 bash $T/run.sh env INTARSIA_INKLING2_BACKEND=hip /work/.scratch/bin/t-main-inkling2 2>&1 \
    | grep -aE '^test |test result|panicked' > $OUT/tiny-main.txt
  echo "[w1 c0] done $(now)"
}
law() { # arm seeds
  rm -f $OUT/law-$1.txt
  HIP_VISIBLE_DEVICES=1 INK_CONTAINER=ink-sampled-run2 bash $T/run.sh env INTARSIA_TEST_REAL_MODEL=1 INTARSIA_INK_MODEL=/models/Inkling-2-Mini \
    INTARSIA_INKLING2_BACKEND=hip INTARSIA_SPEC_SAMPLED_DFLASH=$1 INTARSIA_SS_SEEDS=$2 INTARSIA_SS_TEMPS=1.0,0.6 INTARSIA_SS_TOKENS=6 \
    INTARSIA_SS_OUT=/work/logs/w1/law-$1.txt /work/.scratch/bin/t-${BR:-br1}-inkling2_sampled_real --nocapture > $OUT/law-$1.log 2>&1
  echo "[w1 c1] law arm $1 rc $? $(now) lines $(wc -l < $OUT/law-$1.txt 2>/dev/null) | $(grep -aE 'requests in|loaded in|DFlash-family' $OUT/law-$1.log | tr '\n' ' ')"
}
card1() {
  echo "[w1 c1] law $(now)"
  law 1 "1000000..$((1000000 + N)),0..$NS"
  law 0 "0..$N"
}
card0 > $OUT/c0.log 2>&1 &
card1 > $OUT/c1.log 2>&1 &
wait
cat $OUT/c0.log $OUT/c1.log
grep -aE 'test result|FAILED|panicked' $OUT/tiny-br.txt $OUT/tiny-main.txt
echo "[w1] done $(now)"
