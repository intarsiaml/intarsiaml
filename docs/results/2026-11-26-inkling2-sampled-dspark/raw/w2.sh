#!/usr/bin/env bash
# Window 2 (lane ink-sampled), ON tatooine inside window.sh. (1) card 0 alone: the `inkling2` tiny suite, branch then main (W1a ran it
# beside a real-model process on card 1 and both trees failed loads), (2) the speed A/B on both cards at once (card 0 reps 1 2, card 1
# reps 3 4; both cards run the same arm at the same time; pairs are within a card).
set -uo pipefail
L=$HOME/intarsiaml-ink-sampled; T=$L/.scratch/tat; OUT=$L/logs/w2; mkdir -p $OUT/speed
M=/models/Inkling-2-Mini
G="generate -m $M --backend hip --max-seq 8192 --profile-every 0 --show-ids"
SP="--max-tokens ${TOKENS:-128} --ignore-eos --prompt-ids /work/.scratch/prompts/speed.json --top-k 0 --top-p 1.0 --seed 7"
now() { date '+%F %T'; }
echo "[w2] tiny inkling2 alone on card 0 $(now)"
HIP_VISIBLE_DEVICES=0 bash $T/run.sh env INTARSIA_INKLING2_BACKEND=hip /work/.scratch/bin/t-${BR:-br2}-inkling2 > $OUT/tiny-br.txt 2>&1
HIP_VISIBLE_DEVICES=0 bash $T/run.sh env INTARSIA_INKLING2_BACKEND=hip /work/.scratch/bin/t-main-inkling2 > $OUT/tiny-main.txt 2>&1
grep -aE 'test result' $OUT/tiny-br.txt $OUT/tiny-main.txt
arm() { # card name rep
  local f=$OUT/speed/$2-r$3 on=0 t=1.0 c=ink-sampled-run
  case $2 in on*) on=1 ;; esac
  case $2 in *07) t=0.7 ;; esac
  [ $1 = 1 ] && c=ink-sampled-run2
  HIP_VISIBLE_DEVICES=$1 INK_CONTAINER=$c bash $T/run.sh env INTARSIA_SPEC_SAMPLED_DFLASH=$on /work/.scratch/bin/intarsia-${BR:-br2} $G $SP --temperature $t > $f.out 2> $f.err
  echo "[w2 c$1] $2 r$3 rc $? $(now) | $(grep -aE '^decode overall' $f.err) | $(grep -aE '^spec: verify' $f.err)"
}
speed() { # card reps
  local c=$1; shift
  for rep in "$@"; do
    if [ $((rep % 2)) = 1 ]; then order="off1 on1 off07 on07"; else order="on07 off07 on1 off1"; fi
    for a in $order; do arm $c $a $rep; done
  done
}
echo "[w2] speed $(now)"
speed 0 ${C0REPS:-1 2} > $OUT/c0.log 2>&1 &
speed 1 ${C1REPS:-3 4} > $OUT/c1.log 2>&1 &
wait
cat $OUT/c0.log $OUT/c1.log
echo "[w2] done $(now)"
