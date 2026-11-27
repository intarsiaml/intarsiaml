#!/usr/bin/env bash
# Lane ink-hip-specflip, window D2 (the sampled DFlash2 test, step 3: measurement), ON tatooine inside window.sh. Binary $BR (the PR B
# candidate: the recipe's DFlash2 default, sampled DFlash2 k4 with the D1 table), both cards at once, one process at a time per card,
# #543's W2 protocol: speed-adapt.json, 128 tokens, --ignore-eos --top-k 0 --top-p 1.0 --seed 7, no --spec flag (the default).
# Arms: off = INTARSIA_SPEC_SAMPLED_DFLASH=0 (plain sampling), pol = unset (the PR B default). Card 0 T 1.0, card 1 T 0.7; reps r1..r4 in
# the order off pol / pol off / pol off / off pol. Out: logs/d2/<arm><T>-r<rep>.err (speed3.py).
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-specflip
RAW=$REPO/docs/results/2026-11-26-inkling2-hip-specflip/raw
BR=${BR:?set BR}
OUT=$REPO/logs/d2; mkdir -p "$OUT" "$REPO/.scratch/prompts" "$REPO/.scratch/home"
cp "$RAW/speed-adapt.json" "$REPO/.scratch/prompts/"
M=/models/Inkling-2-Mini
B=/work/.scratch/bin/intarsia-$BR
now() { date '+%T'; }
echo "[d2] start $(now) $BR $(md5sum "$REPO/.scratch/bin/intarsia-$BR" | cut -c1-32) commit $(cat "$REPO/.scratch/bin/intarsia-$BR.commit" 2>/dev/null)"
rc() { local c=$1; shift; env CARD=$c CNAME=ink-hip-specflip-run$c bash "$RAW/run.sh" env HOME=/work/.scratch/home "$@"; }
G="generate -m $M --backend hip --max-seq 8192 --profile-every 0 --show-ids"
SP="--max-tokens 128 --ignore-eos --prompt-ids /work/.scratch/prompts/speed-adapt.json --top-k 0 --top-p 1.0 --seed 7"
arm() { # card arm temp-suffix temp rep
  local f=$OUT/$2$3-r$5 e="-u INTARSIA_SPEC_SAMPLED_DFLASH"
  [ $2 = off ] && e="INTARSIA_SPEC_SAMPLED_DFLASH=0"
  date +%s.%N > $f.t0
  rc $1 env $e $B $G $SP --temperature $4 > $f.out 2> $f.err
  local r=$?; date +%s.%N >> $f.t0; mv $f.t0 $f.t
  echo "[d2 c$1] $2$3 r$5 rc $r $(now) | $(grep -aE '^decode overall' $f.err) | $(grep -aE '^spec: verify' $f.err) | $(grep -aE 'DFlash2 default|DFlash-family drafting|draft length per request' $f.err | tr '\n' ' ' | cut -c1-300)"
}
speed() { # card suffix temp
  arm $1 off $2 $3 1; arm $1 pol $2 $3 1
  arm $1 pol $2 $3 2; arm $1 off $2 $3 2
  arm $1 pol $2 $3 3; arm $1 off $2 $3 3
  arm $1 off $2 $3 4; arm $1 pol $2 $3 4
}
speed 0 1 1.0 > "$OUT/c0.log" 2>&1 &
speed 1 07 0.7 > "$OUT/c1.log" 2>&1 &
wait
cat "$OUT/c0.log" "$OUT/c1.log"
echo "[d2] done $(now)"
