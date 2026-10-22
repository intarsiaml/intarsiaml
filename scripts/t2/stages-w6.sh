#!/usr/bin/env bash
# Lane T2 window 6 (ON tatooine, radiance down): the discriminating run for section 3 of the results doc. If tp=2 is bitwise
# tp=1 because W4A4 makes every MXFP4 product and sum exact, then with 16-bit activations (`--mxfp4-act none`, W4A16, not the
# declared scheme) tp=2 must NOT be bitwise tp=1. 256 greedy ids short / 4k, and accuracy dumps on P1's W4A16 HF golden
# (compared offline with cmpdump.py, and by --self-golden here).
set -uo pipefail
export O=${1:?out dir}
source "$HOME/t2/lib.sh"
A16="--mxfp4-act none"
ids head 0 27-a16-tp1-short $M27 short $A16
ids head 0,1 27-a16-tp2-short $M27 short $A16 $TP2
ids head 0 27-a16-tp1-4k $M27 4k $A16
ids head 0,1 27-a16-tp2-4k $M27 4k $A16 $TP2
same ids-27-a16-tp1-short.ids ids-27-a16-tp2-short.ids "27B W4A16 tp=2 vs tp=1 short"
same ids-27-a16-tp1-4k.ids ids-27-a16-tp2-4k.ids "27B W4A16 tp=2 vs tp=1 4k"
firstdiff ids-27-a16-tp1-short.ids ids-27-a16-tp2-short.ids "27B W4A16 tp=2 vs tp=1 short"
firstdiff ids-27-a16-tp1-4k.ids ids-27-a16-tp2-4k.ids "27B W4A16 tp=2 vs tp=1 4k"
G16=/work/p1-golden/qwen3.8-27b-mxfp4-hfquark-w4a16-top20.safetensors
for a in tp1 tp2; do
  x=""; g=0; [[ $a == tp2 ]] && { x="$TP2"; g=0,1; }
  # shellcheck disable=SC2086
  run head $g 600 acc-a16-$a "$HUB" accuracy --backend hip --model $M27 --mem-budget 29 --golden $G16 --mode both $A16 $x \
    --write-self-golden /work/t2-out/acc-a16-$a.safetensors --allow-unstamped
  mv "$(T head)/t2-out/acc-a16-$a.safetensors" "$O/" 2>/dev/null
done
PRE=4724b23 self head 0,1 a16-tp2-vs-tp1 acc-a16-tp1.safetensors $A16 $TP2
log done
