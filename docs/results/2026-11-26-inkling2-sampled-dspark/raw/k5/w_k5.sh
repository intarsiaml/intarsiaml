#!/usr/bin/env bash
# k5 refresh window (lane ink-sampled-k5), ON tatooine inside window.sh (CAP 265 s).
# Phase 1: greedy md5 at 64 tokens over md5.json (6 short prompts + the 4k one), card 0 = main5, card 1 = br5, each spec off and the
#   recipe default (DSpark k5 on hip), NO INTARSIA_SPEC_SAMPLED_DFLASH set.
# Phase 2: sampled speed A/B on br5, both cards at once, same arm at the same time, pairs within one card (as W2 of the k3 run):
#   card 0 = reps 1 2, card 1 = reps 3 4; per temperature ABBA per card (off on | on off); T 1.0 first, T 0.7 last.
#   arm off = variable unset, arm on = INTARSIA_SPEC_SAMPLED_DFLASH=1. 128 tokens, speed.json, seed 7, top-k 0, top-p 1.0.
set -uo pipefail
L=$HOME/intarsiaml-ink-sampled; T=$L/.scratch/tat; OUT=$L/logs/k5; mkdir -p $OUT/speed
M=/models/Inkling-2-Mini
now() { date '+%F %T'; }
echo "[k5] $(md5sum $L/.scratch/bin/intarsia-br5 $L/.scratch/bin/intarsia-main5 | tr '\n' ' ')"
G="generate -m $M --backend hip --max-seq 8192 --profile-every 0 --show-ids"
md5() { # card binary
  local c=ink-sampled-run; [ $1 = 1 ] && c=ink-sampled-run2
  for sp in off def; do
    local f=$OUT/md5-$2-$sp extra=""
    [ $sp = off ] && extra="--spec off"
    HIP_VISIBLE_DEVICES=$1 INK_CONTAINER=$c bash $T/run.sh env -u INTARSIA_SPEC_SAMPLED_DFLASH /work/.scratch/bin/intarsia-$2 $G --max-tokens 64 --prompt-ids /work/.scratch/prompts/md5.json $extra > $f.out 2> $f.err
    echo "[k5 c$1] md5 $2 $sp rc $? $(now) | $(grep -aE '^decode overall' $f.err) | $(grep -aE '^spec: verify' $f.err)"
  done
}
echo "[k5] md5 phase $(now)"
( md5 0 main5 ) > $OUT/m0.log 2>&1 &
( md5 1 br5 ) > $OUT/m1.log 2>&1 &
wait
cat $OUT/m0.log $OUT/m1.log
SP="--max-tokens ${TOKENS:-128} --ignore-eos --prompt-ids /work/.scratch/prompts/speed.json --top-k 0 --top-p 1.0 --seed 7"
arm() { # card name rep
  local f=$OUT/speed/$2-r$3 t=1.0 c=ink-sampled-run pre="-u INTARSIA_SPEC_SAMPLED_DFLASH"
  case $2 in on*) pre="INTARSIA_SPEC_SAMPLED_DFLASH=1" ;; esac
  case $2 in *07) t=0.7 ;; esac
  [ $1 = 1 ] && c=ink-sampled-run2
  date +%s.%N > $f.t0
  HIP_VISIBLE_DEVICES=$1 INK_CONTAINER=$c bash $T/run.sh env $pre /work/.scratch/bin/intarsia-br5 $G $SP --temperature $t > $f.out 2> $f.err
  echo "[k5 c$1] $2 r$3 rc $? $(now) | $(grep -aE '^decode overall' $f.err) | $(grep -aE '^spec: verify' $f.err)"
}
speed() { # card rep_a rep_b   (rep_a odd-style: off first; rep_b reversed)
  local c=$1 a=$2 b=$3
  for tt in 1 07; do arm $c off$tt $a; arm $c on$tt $a; arm $c on$tt $b; arm $c off$tt $b; done
}
echo "[k5] speed $(now)"
speed 0 1 2 > $OUT/c0.log 2>&1 &
speed 1 3 4 > $OUT/c1.log 2>&1 &
wait
cat $OUT/c0.log $OUT/c1.log
echo "[k5] done $(now)"
