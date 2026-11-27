#!/usr/bin/env bash
# Lane ink-sampled-adapt, HIP window W2, ON tatooine inside window.sh (CAP <= 282 s): sampled speed, three arms, both cards at once;
# card 0 T 1.0, card 1 T 0.7, 3 reps each in a Latin square (off fix pol / fix pol off / pol off fix): 9 processes per card (three arms x
# 4 reps = 12 per card do not fit a 5-min window at ~25 s per process).
#   off = ad4, INTARSIA_SPEC_SAMPLED_DFLASH unset; fix = main6 (main aee06201c, fixed k5) with =1; pol = ad4 with =1 (+ ADAPT=<string>).
#   speed.json, 128 tokens, --ignore-eos --top-k 0 --top-p 1.0 --seed 7, recipe default (DSpark k5, cut 0.5); T 1.0 first.
set -uo pipefail
L=$HOME/intarsiaml-ink-sampled-adapt; T=$L/.scratch/tat; OUT=$L/logs/w2; mkdir -p $OUT/speed
M=/models/Inkling-2-Mini
now() { date '+%F %T'; }
echo "[w2] $(md5sum $L/.scratch/bin/intarsia-ad4 $L/.scratch/bin/intarsia-main6 | tr '\n' ' ') ADAPT=${ADAPT:-<default>}"
G="generate -m $M --backend hip --max-seq 8192 --profile-every 0 --show-ids"
SP="--max-tokens 128 --ignore-eos --prompt-ids /work/.scratch/prompts/speed.json --top-k 0 --top-p 1.0 --seed 7"
arm() { # card name rep
  local f=$OUT/speed/$2-r$3 t=1.0 c=ink-adapt-run pre="-u INTARSIA_SPEC_SAMPLED_DFLASH" bin=ad4
  case $2 in fix*) pre="INTARSIA_SPEC_SAMPLED_DFLASH=1"; bin=main6 ;; pol*) pre="INTARSIA_SPEC_SAMPLED_DFLASH=1 ${ADAPT:+INTARSIA_SPEC_SAMPLED_ADAPT=$ADAPT}" ;; esac
  case $2 in *07) t=0.7 ;; esac
  [ $1 = 1 ] && c=ink-adapt-run2
  date +%s.%N > $f.t0
  HIP_VISIBLE_DEVICES=$1 INK_CONTAINER=$c bash $T/run.sh env $pre /work/.scratch/bin/intarsia-$bin $G $SP --temperature $t > $f.out 2> $f.err
  local rc=$?; date +%s.%N >> $f.t0; mv $f.t0 $f.t
  echo "[w2 c$1] $2 r$3 rc $rc $(now) | $(grep -aE '^decode overall' $f.err) | $(grep -aE '^spec: verify' $f.err)"
}
speed() { # card temp
  local c=$1 tt=$2
  arm $c off$tt 1; arm $c fix$tt 1; arm $c pol$tt 1
  arm $c fix$tt 2; arm $c pol$tt 2; arm $c off$tt 2
  arm $c pol$tt 3; arm $c off$tt 3; arm $c fix$tt 3
}
echo "[w2] speed $(now)"
speed 0 1 > $OUT/c0.log 2>&1 &
speed 1 07 > $OUT/c1.log 2>&1 &
wait
cat $OUT/c0.log $OUT/c1.log
echo "[w2] done $(now)"
